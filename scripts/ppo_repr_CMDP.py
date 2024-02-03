import os
import argparse
import yaml
import numpy as np
import gymnasium as gym
import safety_gymnasium as sgym
import torch
import time
import wandb

from fcsrl.agent import PPOLagReprAgent
from fcsrl.trainer import onpolicy_trainer
from fcsrl.data import Collector, ReplayBuffer
from fcsrl.env import SubprocVectorEnv, GoalWrapper, ActionRepeatWrapper
from fcsrl.utils import Config, set_seed, BaseNormalizer, MeanStdNormalizer, dict2attr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--render_path', type=str)
    parser.add_argument('--hyperparams', type=str, default='hyper_params/PPORepr_Lag.yaml')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--cudaid', type=int, default=-1)
    parser.add_argument('--repr_type', type=str, default="FCSRL")
    args = parser.parse_args()

    Config().select_device(args.cudaid)

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

    # initialize train/test env
    train_envs = SubprocVectorEnv(
        [lambda: ActionRepeatWrapper(GoalWrapper(gym.make(env_cfg.name)), 4) for _ in range(env_cfg.num_env_train)])
    test_envs = SubprocVectorEnv(
        [lambda: ActionRepeatWrapper(gym.make(env_cfg.name), 4) for _ in range(env_cfg.num_env_test)])

    # seed
    set_seed(misc_cfg.seed)
    if hasattr(env, "seed"):
        train_envs.seed(misc_cfg.seed)
        test_envs.seed(misc_cfg.seed)

    normalizer = MeanStdNormalizer() if config.agent.obs_normalizer == "MeanStdNormalizer" else BaseNormalizer()

    agent = PPOLagReprAgent(config, obs_normalizer=normalizer)
    
    save_path = None

    # trainer
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
                "method": "ppo_lag",
                "repr_type": agent.repr_type,
            }
        )

        if not lagrg_cfg.schedule_threshold:
            threshold = lagrg_cfg.constraint_threshold
        else:
            threshold = [
                lagrg_cfg.threshold_start,
                lagrg_cfg.threshold_end,
                lagrg_cfg.schedule_epoch,
            ]

        ## onpolicy_trainer
        result = onpolicy_trainer(
            agent, 
            train_collector, 
            test_collector, 
            trainer_cfg.warmup_episode,
            trainer_cfg.epoch,
            trainer_cfg.batch_size,
            trainer_cfg.step_per_epoch, 
            trainer_cfg.collect_episode_per_step, 
            trainer_cfg.train_repeat,
            trainer_cfg.test_episode,
            threshold,
            None,
            save_path, 
            stop_fn, 
        )

        train_collector.close()
        test_collector.close()

    else:
        pass
        # env = SubprocVectorEnv(
        #     [lambda: ActionRepeatWrapper(gym.make(env_cfg.name)) for _ in range(20)])
        # env.seed(100)
        # save_path = f"{trainer_cfg.model_dir}/{env_cfg.name}/ppo_repr_lag_seed_{misc_cfg.seed}"
        # agent.load_model(save_path)
        # agent.eval()

        # collector = Collector(agent, env, ReplayBuffer(100000))
        # result = collector.collect(n_episode=1, render_path=misc_cfg.render_path)
        # print(f'Final reward: {result["reward"]}, cost: {result["cost"]}, length: {result["length"]}')

if __name__ == "__main__":
    main()