import time, os
from typing import Tuple, List, Any
import numpy as np
from tqdm import tqdm
import wandb

from fcsrl.trainer.utils import test_agent, gather_info

def offpolicy_trainer(
    agent, 
    train_collector, 
    test_collector, 
    
    n_episode_warmup,
    max_epoch,
    batch_size, 
    step_per_epoch, 
    collect_per_step, 
    n_episode_test, 
    
    threshold,
    metric_convert_fn,
    
    save_path,
    stop_fn=None,
):
    
    global_step = 0 # step: times of optimization by Gradient Descent
    global_length = 0 # length: times of interacting with Env
    best_epoch, best_reward = -1, -1

    total_violation = 0

    wandb.define_metric('Steps')
    wandb.define_metric("*", step_metric="Steps")

    print('------ Start warm-up. ------')
    train_collector.collect(n_episode=n_episode_warmup, random=True)
    print('------ End warm-up. ------')

    start_time = time.time()

    for epoch in range(1, max_epoch + 1):

        # reset constraint threshold by decay.
        if isinstance(threshold, list):
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
                result = train_collector.collect(n_step=collect_per_step)

                for _ in range(min(result['n_step'] // collect_per_step,
                                t.total - t.n)):
                    global_step += 1
                    
                    batch_data, indices = train_collector.sample(batch_size)
                    batch_data = agent.process_fn(batch_data, train_collector.replay, indices)
                    loss_dict = agent.learn(batch_data)
                    agent.post_process_fn(batch_data, train_collector.replay, indices)

                metric_dict = {}
                if result['length'] > 0:
                    wandb.log({**loss_dict, "Steps":global_step})

                    if hasattr(agent, 'update_lagrangian_multiplier') and epoch >= 1:
                        agent.update_lagrangian_multiplier(result['cost'])

                    # if 'reward_list' in result:
                    #     for r, c, l in zip(result['reward_list'], result['cost_list'], result['length_list']):
                    #         global_length += l
                    #         v = max(0, c - J_cost_thres)
                    #         total_violation += v
                    # else:
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
                
                t.update(1)
                t.set_postfix(**postfix_data)

            if t.n < t.total:
                t.update()

        # test
        result = test_agent(agent, test_collector, n_episode_test)
        wandb.log({
            'test/reward': result['reward'],
            'test/cost': result['cost'],
            'Steps': global_length,
        })

        if best_epoch == -1 or best_reward < result['reward']:
            best_reward = result['reward']
            best_epoch = epoch

            # save model
            if save_path:
                tmp_path = os.path.join(save_path, f"epoch_{epoch}")
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path, 0o777)
                agent.save_model(tmp_path)
        
        print(f'Epoch #{epoch} ================')
        print(f'test_reward: {result["reward"]:.3f}±{np.std(result["reward_list"]):.3f}')
        print(f'test_cost: {result["cost"]:.3f}±{np.std(result["cost_list"]):.3f}')
        print(f'best_reward: {best_reward:.3f} in #{best_epoch}')
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector, best_reward)