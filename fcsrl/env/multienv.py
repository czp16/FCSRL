import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
import time

from fcsrl.env.utils import CloudpickleWrapper

class BaseMultiEnv:
    def __init__(self, env_fns):
        self._env_fns = env_fns
        self.env_num = len(env_fns)

    def __len__(self):
        return self.env_num
    
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def seed(self, seed=None):
        raise NotImplementedError

    def render(self, **kwargs):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError


class VectorEnv(BaseMultiEnv):

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]

    def reset(self, index=None):
        if index is None:
            ret_list = [e.reset() for e in self.envs]
            self._obs = np.stack([r[0] for r in ret_list])
            self._info = np.stack([r[1] for r in ret_list])
        else:
            if np.isscalar(index):
                index = [index]
            for i in index:
                self._obs[i], self._info[i] = self.envs[i].reset()
        
        return self._obs, self._info

    def step(self, action):
        assert len(action) == self.env_num
        result = [e.step(a) for e,a in zip(self.envs, action)]
        self._obs, self._rew, self._terminate, self._trunc, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._terminate = np.stack(self._terminate)
        self._trunc = np.stack(self._trunc)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._terminate, self._trunc, self._info

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            if hasattr(e, 'seed'):
                result.append(e.seed(s))
        return result

    def render(self, **kwargs):
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):
                result.append(e.render(**kwargs))
        return result

    def close(self):
        for e in self.envs:
            e.close()


def worker(parent, p, env_fn_wrapper):
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            elif cmd == 'reset':
                p.send(env.reset())
            elif cmd == 'close':
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseMultiEnv):

    def __init__(self, env_fns, context=None):
        super().__init__(env_fns)
        ctx = mp.get_context(context)
        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[ctx.Pipe() for _ in range(self.env_num)])
        self.processes = [
            ctx.Process(target=worker, args=(
                parent, child, CloudpickleWrapper(env_fn)), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def step(self, action):
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._terminate, self._trunc, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._terminate = np.stack(self._terminate)
        self._trunc = np.stack(self._trunc)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._terminate, self._trunc, self._info

    def reset(self, index=None):
        if index is None:
            for p in self.parent_remote:
                p.send(['reset', None])
            ret_list = [p.recv() for p in self.parent_remote]
            self._obs = np.stack([r[0] for r in ret_list])
            self._info = np.stack([r[1] for r in ret_list])
            return self._obs, self._info
        else:
            if np.isscalar(index):
                index = [index]
            for i in index:
                self.parent_remote[i].send(['reset', None])
            for i in index:
                self._obs[i], self._info[i] = self.parent_remote[i].recv()
            return self._obs, self._info

    def seed(self, seed=None):
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs):
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self):
        if self.closed:
            return
        for p in self.parent_remote:
            p.send(['close', None])
        self.closed = True
        for p in self.processes:
            p.join()