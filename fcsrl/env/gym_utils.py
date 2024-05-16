from typing import Any
import gymnasium as gym
import numpy as np
import time

from safety_gymnasium.bases.underlying import VisionEnvConf

class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        next_s, r, terminate, trunc, info = self.env.step(action)
        
        if ('goal_met' in info) and info['goal_met']:
            info['terminate'] = True
        else:
            info['terminate'] = False
        return next_s, r, terminate, trunc, info
    
class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_repeat: int):
        super().__init__(env)
        self.n_repeat = n_repeat

    def step(self, action):
        total_r = 0.0
        total_c = 0.0
        total_term = total_trunc = total_psuedo_term = False

        total_info = {}
        for _ in range(self.n_repeat):
            next_s, r, terminate, trunc, info = self.env.step(action)
            total_r += r
            total_c += info['cost']
            total_term = (total_term or terminate)
            total_trunc = (total_trunc or trunc)
            total_psuedo_term = (total_psuedo_term or info.get("terminate", False))

            if terminate or trunc:
                break
        
        total_info["cost"] = total_c
        total_info["terminate"] = total_psuedo_term
            
        return next_s, total_r, total_term, total_trunc, total_info

class VisionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1.0, (64, 64, 3))
    
    def observation(self, observation):
        VisionEnvConf.vision_size = (64,64)
        obs = self.env.task._obs_vision() / 255
        return obs