from typing import Tuple, List, Optional, Union, Callable
import time, os
import numpy as np
from tqdm import tqdm
import wandb

from fcsrl.trainer.utils import test_agent, gather_info
from fcsrl.data import Collector


def onpolicy_trainer(
    agent,
    train_collector: Collector,
    test_collector: Collector,
    n_episode_warmup: int,
    max_epoch: int,
    batch_size: int,
    step_per_epoch: int,
    collect_per_step: int,
    train_repeat: int,
    n_episode_test: int,
    threshold: Union[float, Tuple[float, float, int]],
    metric_convert_fn: Optional[Callable],
    save_path: Optional[str],
    stop_fn: Optional[Callable] = None,
):

    global_optim_step = 0 # the times of optimization (for each batch)
    global_length = 0 # the times of env taking action to step
    best_epoch, best_reward, best_cost = -1, -1, 0
    
    total_violation = 0

    wandb.define_metric('Steps')
    wandb.define_metric("*", step_metric="Steps")

    assert n_episode_warmup > 0, "Warmup episode should be > 0."
    print('------ Start warm-up. ------')
    train_collector.collect(n_episode=n_episode_warmup, random=True)
    batch_data, indices = train_collector.sample(0)
    batch_data = agent.process_fn(batch_data, train_collector.replay, indices)
    agent.learn(batch_data, batch_size, train_repeat)
    train_collector.reset_replay()
    print('------ End warm-up. ------')

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        
        # reset constraint threshold by decay.
        if isinstance(threshold, tuple):
            thres_start, thres_end, schedule_epoch = threshold
            if epoch <= schedule_epoch:
                J_cost_thres = thres_start + (thres_end - thres_start) * epoch / schedule_epoch
            else:
                J_cost_thres = thres_end
        else:
            J_cost_thres = threshold
        
        if metric_convert_fn is not None:
            cost_thres = metric_convert_fn(J_cost_thres)
        else:
            cost_thres = J_cost_thres

        agent.cost_limit = cost_thres
        print('cost threshold', J_cost_thres)


        postfix_data = {}
        # train
        agent.train()

        with tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}') as t:
            while t.n < t.total:
                result = train_collector.collect(n_episode=collect_per_step)
                
                # update lambda
                if hasattr(agent, 'update_lagrangian_multiplier'):
                    agent.update_lagrangian_multiplier(result['cost'])

                batch_data, indices = train_collector.sample(0)
                batch_data = agent.process_fn(batch_data, train_collector.replay, indices)
                loss_dict = agent.learn(batch_data, batch_size, train_repeat)
                train_collector.reset_replay()

                _optim_step = 1
                for k in loss_dict.keys():
                    if isinstance(loss_dict[k], list):
                        _optim_step = max(len(loss_dict[k]), _optim_step)
                global_optim_step += _optim_step

                for k, v in loss_dict.items():
                    loss_dict[k] = np.mean(v)
                wandb.log({**loss_dict, "Steps":global_optim_step})
                

                global_length += result['length']
                total_violation += max(0, result['cost'] - J_cost_thres) * result['n_episode']
                
                metric_dict = {
                    'train/reward':result['reward'],
                    'train/cost':result['cost'],
                    'train/total violation': total_violation,
                    'train/lagrangian': agent.lagrg,
                    'Steps': global_length,
                }
                wandb.log(metric_dict)
                
                postfix_data['reward'] = f'{result["reward"]:.2f}'
                postfix_data['cost'] = f'{result["cost"]:.2f}'
                postfix_data['lambda'] = f'{agent.lagrg:.2f}'
                
                t.update(_optim_step)
                t.set_postfix(**postfix_data)

            if t.n <= t.total:
                t.update()

        # test
        result = test_agent(agent, test_collector, n_episode_test)
        if best_epoch == -1 \
            or (best_reward < result['reward'] and result['cost'] < J_cost_thres):
            best_reward, best_cost = result['reward'], result['cost']
            best_epoch = epoch

            # save model
            if save_path:
                _path = os.path.join(save_path, f"epoch_{epoch}")
                os.makedirs(_path, 0o777, exist_ok=True)
                agent.save_model(_path)

        wandb.log({
            'test/reward': result['reward'],
            'test/cost': result['cost'],
            'test/best_reward': best_reward,
            'test/best_cost': best_cost,
            'Steps': global_length,
        })

        print(f'Epoch #{epoch} ================')
        print(f'test_reward: {result["reward"]:.3f}±{np.std(result["reward_list"]):.3f}')
        print(f'test_cost: {result["cost"]:.3f}±{np.std(result["cost_list"]):.3f}')
        print(f'best: reward {best_reward:.3f}, cost {best_cost:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)