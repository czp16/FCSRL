import numpy as np
import torch
import torch.nn.functional as F

from fcsrl.utils.misc import symexp, symlog

class PIDLagrangianUpdater:
    def __init__(self, init_lag, K, max_lag=100, min_integral_coef=-0.3):
        super().__init__()
        if np.isscalar(K):
            self.KP, self.KI, self.KD = 0.0, K, 0.0
        else:
            self.KP, self.KI, self.KD = K
            
        self.error_old = 0.0
        self.error_integral = 0.0
        self.lag = init_lag
        self.max_lag = max_lag
        self.min_integral = min_integral_coef / self.KI 
        # s.t. KI * error_int >= min_integral_coef

    def update(self, curr_cost, thres):
        # update cost coef
        error_new = np.mean(curr_cost - thres)
        error_diff = max(0, error_new - self.error_old)
        self.error_integral = max(self.min_integral, self.error_integral + error_new)
        self.error_old = error_new

        self.lag = max(0, 
            self.KP * error_new + 
            self.KI * self.error_integral +
            self.KD * error_diff
        )
        self.lag = min(self.lag, self.max_lag)
    
    def get_lagrg(self):
        return self.lag

class DiscDist:
    def __init__(
        self,
        logits,
        low=-5.0,
        high=5.0,
        n_buckets=63,
        transfwd=symlog,
        transbwd=symexp,
    ):
        self.buckets = torch.linspace(low, high, n_buckets).to(logits.device)
        self.n_buckets = n_buckets
        self.bin_size = (high-low) / (n_buckets-1)
        self.logits = logits
        # self.probs = torch.softmax(logits, -1)
        self.width = (self.buckets[-1] - self.buckets[0]) / len(self.buckets)
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        probs = torch.softmax(self.logits, -1)
        _mean = probs * self.buckets.view(1,1,-1)
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    # def mode(self):
    #     _mode = self.probs * self.buckets
    #     return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x) # (T,B,1)
        buckets = self.buckets.view(1,1,-1)
        below_idx = torch.sum((buckets <= x).long(), dim=-1) - 1 # (T,B)
        above_idx = below_idx + 1
        below_idx = torch.clip(below_idx, 0, self.n_buckets - 1)
        above_idx = torch.clip(above_idx, 0, self.n_buckets - 1)
        
        above_value = self.buckets[above_idx].unsqueeze(-1) # (T,B,1)
        below_value = self.buckets[below_idx].unsqueeze(-1)

        weight_below = (above_value - x) / self.bin_size
        weight_above = (x - below_value) / self.bin_size
        target = (
            F.one_hot(below_idx, num_classes=self.n_buckets) * weight_below
            + F.one_hot(above_idx, num_classes=self.n_buckets) * weight_above
        )
        log_pred = torch.log_softmax(self.logits, dim=-1)
        return (target * log_pred).sum(-1, keepdim=True) # (T,B,1)