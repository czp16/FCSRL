import os
import types
import numpy as np
import torch
import torch.nn.functional as F

from fcsrl.utils.config import DeviceConfig

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        if x.device != DeviceConfig.DEVICE:
            return x.to(device=DeviceConfig.DEVICE)
        else:
            return x
    # x = np.asarray(x, dtype=float)
    x = torch.as_tensor(x, device=DeviceConfig.DEVICE, dtype=torch.float32)
    return x

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_update(model_old, model_new, tau):
    for o, n in zip(model_old.parameters(), model_new.parameters()):
        o.data.copy_(o.data * (1-tau) + n.data * tau)

# symmetrical log
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

# symmetrical exp
def symexp(y):
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1)

def cosine_sim_loss(x, y):
    x_ = F.normalize(x, p=2.0, dim=-1, eps=1e-3)
    y_ = F.normalize(y, p=2.0, dim=-1, eps=1e-3)
    return (x_ * y_).sum(-1, keepdim=True)

    
class linear_scheduler:
    def __init__(self, start, end, schedule_len):
        self.start = start
        self.end = end
        self.schedule_len = schedule_len
    
    def get_value(self, T):
        assert T >= 0, 'schedule time must be non-negative.'
        if T <= self.schedule_len:
            return self.start + (self.end - self.start) * T / self.schedule_len
        else:
            return self.end

def recusive_update(cfg: object, cfg_dict: dict):
    for k, v in cfg_dict.items():
        if isinstance(v, dict):
            setattr(cfg, k, types.SimpleNamespace())
            recusive_update(getattr(cfg, k), v)
        else:
            setattr(cfg, k, v)

def dict2attr(cfg_dict):
    config = types.SimpleNamespace()
    recusive_update(config, cfg_dict)
    return config

# compute the constraint defined by Q-value according to the constraint defined by return J
# E[\sum_t c(s_t, a_t)] \leq \eps_J ==> E[Q_c(s_t, a_t)] \leq \eps_q
# input \eps_J, output \eps_q
def J_to_q(j, discount, max_episode_len):
    q = j * (1 - discount ** max_episode_len) / (1 - discount) / max_episode_len
    return q

# key function for n step return
# refer to https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/base.py#L349 
def _nstep_return(rew, end_flag, target_q, indices, gamma, n_step):
    gamma_buffer = np.ones(n_step + 1)
    for i in range(1, n_step+1):
        gamma_buffer[i] = gamma_buffer[i-1] * gamma
    target_shape = target_q.shape
    b_size = target_shape[0]

    target_q = target_q.reshape(b_size, -1)
    returns = np.zeros_like(target_q)
    gammas = np.full(indices[0].shape, n_step)

    for n in reversed(range(n_step)):
        current = indices[n]
        gammas[end_flag[current] > 0] = n+1
        returns[end_flag[current] > 0] = 0.0
        returns = rew[current].reshape(b_size, 1) + gamma * returns
    target_q = target_q * gamma_buffer[gammas].reshape(b_size, 1) + returns
    return target_q.reshape(target_shape)
