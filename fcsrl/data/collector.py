from typing import Optional, List

import gymnasium as gym
import torch
import time
import numpy as np

from fcsrl.data import Batch, CacheBuffer, VectorCacheBuffer, ReplayBuffer
from fcsrl.env import BaseMultiEnv
from fcsrl.utils import to_numpy, BaseNormalizer

class Collector:

    def __init__(
        self, 
        agent, 
        env, 
        replay: Optional[ReplayBuffer] = None,
        act_space: Optional[gym.Space] = None, 
        store_keywords: Optional[List[str]] = None,
        has_cost: bool = True,
    ):
        """Collector

        args:
            agent: agent of reinforcement learning
            env: environment to collect
            replay: experience pool
            store_keywords: variable to be stored after each step of agent
            has_cost: if the MDP includes cost in `info`
        
        return:
            None
        """
        self.agent = agent
        self.process_fn = agent.process_fn
        self.env = env
        self._multi_env = False
        self.env_num = 1
        if isinstance(env, BaseMultiEnv):
            self._multi_env = True
            self.env_num = len(env)
        self.replay = replay
        self.act_space = act_space
        self.store_keywords = store_keywords
        self.has_cost = has_cost

        self.states = None

        if self._multi_env:
            self._cache_buffer = VectorCacheBuffer(self.env_num)
        else:
            self._cache_buffer = CacheBuffer()
        self._store_data = (self.replay is not None)

        # we run obs normalization before feeding `obs` to the agent
        # but still store unnormalized obs in replay buffer
        if hasattr(self.agent, 'obs_normalizer'):
            self.obs_normalizer = self.agent.obs_normalizer
        else:
            self.obs_normalizer = BaseNormalizer()

        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0.0

        self.reset_env()
    

    def reset_env(self):
        self._obs, _ = self.env.reset()
        self._act = self._rew = self._cost = self._terminate = self._trunc = self._info = None

        # rew is one-step reward, reward is sum of rew
        if self._multi_env:
            self.cum_reward = np.zeros(self.env_num)
            self.cum_cost = np.zeros(self.env_num)
            self.length = np.zeros(self.env_num)
        else:
            self.cum_reward = self.cum_cost = self.length = 0

    def reset_replay(self):
        if self._store_data:
            self._cache_buffer.reset()
            self.replay.reset()


    def _reset_states(self, index=None):
        if hasattr(self.agent, 'reset_states'):
            self.agent.reset_states(self.states, index)


    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs):
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


    def collect(self, n_step=0, n_episode=0, render_path='', random=False):

        assert sum([n_step!=0, n_episode!=0]) == 1, \
            'only 1 of n_step or n_episode should > 0'
        if render_path:
            assert not self._multi_env, 'Vectorized Env is not supported for rendering.'

        start_time = time.time()
        current_step = 0
        current_episode = 0 # np.zeros(self.env_num, dtype=int) if self._multi_env else 0
        reward_list, cost_list, length_list = [], [], []

        if render_path:
            import imageio
            render_imgs = []

        while True:
            # 0. normalize and update normalizer
            # we will **only** update obs normalizer here
            if self.agent.training:
                self.obs_normalizer.unset_read_only()
            obs_normalized = self.obs_normalizer(self._obs)
            self.obs_normalizer.set_read_only()

            # 1. prepare actions, either from agent or from random sampling
            if not self._multi_env:
                obs_normalized = np.array([obs_normalized]) # [1, obs_dim]
            batch_data = Batch(obs=obs_normalized)
            
            if random:
                self._act = np.stack([self.act_space.sample() for _ in range(self.env_num)], axis=0)
            else:
                with torch.no_grad():
                    result, self.states = self.agent(batch_data, states=self.states)
            
                self._act = to_numpy(result.act)
            
            if not self._multi_env:
                self._act = self._act[0]

            # 2. execute `step` in env
            obs_next, self._rew, self._terminate, self._trunc, self._info = self.env.step(self._act)
            
            # store values for `cost` and other `store_keywords`
            self._cost = np.stack([self._info[_]['cost'] for _ in range(self.env_num)]) if self._multi_env else self._info['cost']
            
            _done = np.logical_or(self._terminate, self._trunc)

            self.cum_reward += self._rew
            self.cum_cost += self._cost
            self.length += 1

            if render_path:
                # self.env.render('human')
                frame = self.env.render('rgb_array', camera_id=1, width=300, height=200)
                render_imgs.append(frame)
                time.sleep(0.01)

            # 3. collect results to replay buffer
            if self._multi_env:
                for i in range(self.env_num):
                    store_dict = {}
                    if self.store_keywords:
                        store_dict = {k: self._info[i][k] for k in self.store_keywords}

                    # `terminate` in `info` (if it exists) indicates the goal reaching but the 
                    #   episode of the env has not ended. 
                    # When computing the value function, we should use 
                    #     $V(s) = r + \gamma * (1-info['terminate']) * V(s')$ 
                    #   as Bellman equation.
                    tmp_terminate = self._info[i].get('terminate', self._terminate[i])

                    if self._store_data:
                        self._cache_buffer.add({
                            'obs': self._obs[i],
                            'act': self._act[i],
                            'rew': self._rew[i],
                            'cost': self._cost[i],
                            'terminate': tmp_terminate,
                            'trunc': self._trunc[i],
                            'obs_next': obs_next[i],
                            **store_dict
                        }, i)
                    current_step += 1

                    if _done[i]:
                        current_episode += 1
                        reward_list.append(self.cum_reward[i])
                        cost_list.append(self.cum_cost[i])
                        length_list.append(self.length[i])
                        
                        self.cum_reward[i], self.cum_cost[i], self.length[i] = 0, 0, 0
                        self._reset_states(i)
                        
                        if self._store_data:
                            self.replay.update(self._cache_buffer[i])
                            self._cache_buffer.reset(i)
                
                if sum(_done) > 0:
                    obs_next, _ = self.env.reset(np.where(_done)[0])
                    
                if n_episode != 0 and current_episode >= n_episode:
                    break

            else:
                store_dict = {}
                if self.store_keywords:
                    store_dict = {k: self._info[k] for k in self.store_keywords}
                
                if self._store_data:
                    self._cache_buffer.add({
                        'obs': self._obs,
                        'act': self._act,
                        'rew': self._rew,
                        'cost': self._cost,
                        'terminate': self._terminate,
                        'trunc': self._trunc,
                        'obs_next': obs_next,
                        **store_dict
                    })

                current_step += 1
                if _done:
                    current_episode += 1
                    reward_list.append(self.cum_reward)
                    cost_list.append(self.cum_cost)
                    length_list.append(self.length)

                    self.cum_reward = self.cum_cost = self.length = 0
                    self._reset_states()

                    if self._store_data:
                        self.replay.update(self._cache_buffer)
                        self._cache_buffer.reset()

                    obs_next, _ = self.env.reset()

                    if n_episode != 0 and current_episode >= n_episode:
                        break
            
            if n_step != 0 and current_step >= n_step:
                break

            self._obs = obs_next
        
        self._obs = obs_next

        duration = time.time() - start_time
        # step/s and episode/s
        # step_ps = current_step / duration
        # episode_ps = current_episode / duration

        self.collect_step += current_step
        self.collect_episode += current_episode
        self.collect_time += duration
        
        n_episode = max(1, current_episode)

        if render_path:
            imageio.mimwrite(render_path, render_imgs, 'GIF', fps=60)
        
        return_info = {
            'n_episode': current_episode,
            'n_step': current_step,
            'reward': sum(reward_list) / n_episode,
            'length': sum(length_list),
            'reward_list': reward_list,
            'length_list': length_list
        }
        if self.has_cost:
            return_info['cost'] = sum(cost_list) / n_episode
            return_info['cost_list'] = cost_list
        
        return return_info

    def sample(self, batch_size):
        return self.replay.sample(batch_size) # batch_data, indices
    