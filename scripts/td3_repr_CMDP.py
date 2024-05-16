import os, sys
# from datetime import datetime
import argparse
import yaml
import numpy as np
import gymnasium as gym
import safety_gymnasium as sgym
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import wandb

from fcsrl.agent import TD3LagReprAgent
from fcsrl.trainer import offpolicy_trainer
from fcsrl.data import Collector, ReplayBuffer
from fcsrl.env import SubprocVectorEnv, GoalWrapper, ActionRepeatWrapper
from fcsrl.utils import DeviceConfig, set_seed, BaseNormalizer, MeanStdNormalizer, dict2attr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render_path', type=str)
    parser.add_argument('--hyperparams', type=str, default='hyper_params/TD3Repr_Lag.yaml')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--cudaid', type=int, default=-1)
    parser.add_argument('--repr_type', type=str, default="FCSRL")
    args = parser.parse_args()

    DeviceConfig().select_device(args.cudaid)

    with open(args.hyperparams, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        cfg_dict["env"]["name"] = args.env_name
        cfg_dict["misc"]["seed"] = args.seed
        cfg_dict["misc"]["test"] = args.test
        cfg_dict["network"]["repr_type"] = args.repr_type
    config = dict2attr(cfg_dict)
    
    env_cfg = config.env
    trainer_cfg = config.trainer
    lagrg_cfg = config.Lagrangian
    misc_cfg = config.misc

    env = GoalWrapper(gym.make(env_cfg.name))
    config.network.s_dim = np.prod(env.observation_space.shape) or env.observation_space.n
    config.network.a_dim = np.prod(env.action_space.shape) or env.action_space.n
    config.agent.action_range = [env.action_space.low[0], env.action_space.high[0]]

    train_env_fn = lambda: ActionRepeatWrapper(GoalWrapper(gym.make(env_cfg.name)), 4)
    test_env_fn = lambda: ActionRepeatWrapper(gym.make(env_cfg.name), 4)
    train_envs = SubprocVectorEnv(
        [train_env_fn for _ in range(env_cfg.num_env_train)])
    test_envs = SubprocVectorEnv(
        [test_env_fn for _ in range(env_cfg.num_env_test)])

    # seed
    set_seed(misc_cfg.seed)
    if hasattr(env, "seed"):
        train_envs.seed(misc_cfg.seed)
        test_envs.seed(misc_cfg.seed)

    normalizer = MeanStdNormalizer() if config.agent.obs_normalizer == "MeanStdNormalizer" else BaseNormalizer()
    
    agent = TD3LagReprAgent(config, obs_normalizer=normalizer)
    
    # save model
    # if not os.path.exists(f"{trainer_cfg.model_dir}/{env_cfg.name}"):
    #     os.makedirs(f"{trainer_cfg.model_dir}/{env_cfg.name}", 0o777)
    # save_path = f"{trainer_cfg.model_dir}/{env_cfg.name}/td7_lag_{args.repr_type}"
    save_path = None

    def stop_fn(r):
        return False # r > env.spec.reward_threshold

    if not misc_cfg.test:
        # collector
        train_collector = Collector(agent, train_envs, ReplayBuffer(config.trainer.replay_size), act_space=env.action_space)
        test_collector = Collector(agent, test_envs)

        # logger
        wandb.init(
            project="FCSRL",
            # entity="", use your ID
            group="main",
            name=f"{env_cfg.name}",
            config={
                "env_name": env_cfg.name,
                "seed": misc_cfg.seed,
                "method": f"td3_lag",
                "repr_type": args.repr_type,
            }
        )

        if not lagrg_cfg.schedule_threshold:
            threshold = lagrg_cfg.constraint_threshold
        else:
            threshold = (
                lagrg_cfg.threshold_start,
                lagrg_cfg.threshold_end,
                lagrg_cfg.schedule_epoch,
            )

        # trainer and start training
        if lagrg_cfg.update_by_J:
            metric_convert_fn = None 
        else: 
            raise NotImplementedError("Lagrangian update by Q function is not supported.")
        
        result = offpolicy_trainer(
            agent, 
            train_collector, 
            test_collector,
            trainer_cfg.warmup_episode,
            trainer_cfg.epoch,
            trainer_cfg.batch_size,
            trainer_cfg.grad_step_per_epoch,
            trainer_cfg.collect_len_per_step,
            trainer_cfg.test_episode,
            threshold,
            metric_convert_fn,
            save_path, 
            stop_fn, 
        )

        train_collector.close()
        test_collector.close()
        train_envs.close()
        test_envs.close()
    
    else:
        pass

if __name__ == "__main__":
    main()
