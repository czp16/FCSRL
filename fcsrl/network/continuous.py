from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcsrl.utils import to_tensor
from fcsrl.network.utils import MLP

LOG_SIG_MAX = 1
LOG_SIG_MIN = -5

class DummyActor(nn.Module):
    def __init__(
        self, 
        a_dim: int,
    ):
        super().__init__()
        self.action_dim = a_dim

    def forward(self, s):
        act = np.random.rand(s.shape[0], self.action_dim)
        act = to_tensor(2 * act - 1.)
        return act


class ActorDeter(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        self.net = MLP(s_dim, hidden_dims, a_dim)

    def forward(self, s, **kwargs):
        s = to_tensor(s)
        logits = self.net(s)
        logits = torch.tanh(logits)
        return logits


class ActorProb(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        self.net = MLP(s_dim, hidden_dims)
        self.mu_net = nn.Linear(hidden_dims[-1], a_dim)
        self.sigma_net = nn.Linear(hidden_dims[-1], a_dim)

    def forward(self, s, **kwargs):
        s = to_tensor(s)
        logits = self.net(s)
        mu = self.mu_net(logits)
        sigma = torch.clamp(self.sigma_net(logits), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)
        return (mu, sigma)


class ActorProb2(nn.Module):
    def __init__(
        self, 
        s_dim, 
        a_dim, 
        hidden_dims
    ):
        super().__init__()
        self.model = []
        last_dim = s_dim
        for next_dim in hidden_dims:
            self.model += [nn.Linear(last_dim, next_dim), nn.ReLU()]
            last_dim = next_dim
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(last_dim, a_dim)
        self.sigma = nn.Parameter(-0.5*torch.ones([1, a_dim]))

    def forward(self, s, **kwargs):
        s = to_tensor(s)
        s = s.view(s.shape[0], -1)
        logits = self.model(s)
        mu = torch.tanh(self.mu(logits))
        sigma = F.softplus(self.sigma).expand(mu.size(0), -1)
        return (mu, sigma)


class Critic(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int = 0, 
        hidden_dims: List[int] = [],
    ):
        super().__init__()
        self.net = MLP(s_dim+a_dim, hidden_dims, 1)

    def forward(self, s, a=None):
        s = to_tensor(s)

        if a is None:
            value = self.net(s)
        else:
            a = to_tensor(a)
            value = self.net(torch.cat([s, a], dim=1))
        return value


class EnsembleCritic(nn.Module):
    def __init__(
        self, 
        ensemble_size: int, 
        s_dim: int, 
        a_dim: int = 0, 
        hidden_dims: List[int] = [],
    ):
        super().__init__()
        self.networks = nn.ModuleList(
            [Critic(s_dim, a_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def forward(self, s, a=None):
        values = []
        for _, net in enumerate(self.networks):
            values.append(net(s, a))
        values = torch.cat(values, dim=1)
        return values

