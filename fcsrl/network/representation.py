from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcsrl.utils import to_tensor
from fcsrl.network.utils import MLP

LOG_SIG_MAX = 1
LOG_SIG_MIN = -5

# from TD7 (https://github.com/sfujim/TD7/tree/main)
def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

class Encoder(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        z_dim: int, 
        zsa_out_dim: int, 
        hidden_dims: List[int] = [256,256],
    ):
        super().__init__()

        # state encoder
        self.zs_net = MLP(s_dim, hidden_dims, z_dim)
        # state-action encoder
        self.zsa_net = MLP(z_dim+a_dim, hidden_dims, zsa_out_dim)

    
    def zs(self, s):
        s = to_tensor(s)
        zs = self.zs_net(s)
        zs = AvgL1Norm(zs)
        # zs = F.normalize(zs, p=2.0, dim=-1, eps=1e-3)
        return zs


    def zsa(self, zs, a):
        a = to_tensor(a)
        zsa = self.zsa_net(torch.cat([zs, a], 1))
        return zsa
    

class EncodedActorDeter(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        z_dim: int = 256, 
        hidden_dims: List[int] = [256,256],
    ):
        super().__init__()

        self.l0 = nn.Linear(s_dim, hidden_dims[0])
        self.net = MLP(z_dim+hidden_dims[0], hidden_dims, a_dim)
        

    def forward(self, s, zs):
        s = to_tensor(s)
        logit_s = self.l0(s)
        logit_s = AvgL1Norm(logit_s)
        a = self.net(torch.cat([logit_s, zs], 1))
        a = torch.tanh(a)
        return a
    

class EncodedActorProb(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int, 
        z_dim: int, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        self.l0 = nn.Linear(s_dim, hidden_dims[0])
        self.net = MLP(z_dim+hidden_dims[0], hidden_dims)
        self.mu_net = nn.Linear(hidden_dims[-1], a_dim)
        self.sigma_net = nn.Linear(hidden_dims[-1], a_dim)

    def forward(self, s, zs):
        s = to_tensor(s)
        logit_s = self.l0(s)
        logit_s = AvgL1Norm(logit_s)

        logits = self.net(torch.cat([logit_s, zs], 1))
        mu = self.mu_net(logits)
        sigma = torch.clamp(self.sigma_net(logits), LOG_SIG_MIN, LOG_SIG_MAX)
        sigma = torch.exp(sigma)
        return (mu, sigma)
    
    
class EncodedCritic(nn.Module):
    def __init__(
        self, 
        s_dim: int, 
        a_dim: int = 0, 
        z_dim: int = 256, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.l0 = nn.Linear(s_dim+a_dim, hidden_dims[0])
        input_dim = 2*z_dim+hidden_dims[0] if a_dim > 0 else z_dim+hidden_dims[0]
        self.net = MLP(input_dim, hidden_dims, 1)


    def forward(self, s, zs, a=None, zsa=None):
        s = to_tensor(s)
        if a is None:
            sa = s
            z_sa = zs
        else:
            a = to_tensor(a)
            sa = torch.cat([s, a], 1)
            z_sa = torch.cat([zs, zsa], 1)

        logit = self.l0(sa)
        logit = AvgL1Norm(logit)
        logit = torch.cat([logit, z_sa], 1)
        logit = self.net(logit)
        return logit
    
class EnsembleEncodedCritic(nn.Module):
    def __init__(
        self, 
        ensemble_size: int, 
        s_dim: int, 
        a_dim: int = 0, 
        z_dim: int = 256, 
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        self.networks = nn.ModuleList(
            [EncodedCritic(s_dim, a_dim, z_dim, hidden_dims) for _ in range(ensemble_size)]
        )

    def forward(self, s, zs, a=None, zsa=None):
        values = []
        for _, net in enumerate(self.networks):
            values.append(net(s, zs, a, zsa))
        values = torch.cat(values, -1)
        return values
    
class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        a_dim: int,
        z_dim: int,
        hidden_dims: List[int],
        depth: int = 16,
        act: str="SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.post_conv = MLP(self.outdim, hidden_dims, z_dim)

        self.zsa_net = MLP(z_dim+a_dim, hidden_dims, z_dim)

    def zs(self, s, unpermuted=False):
        # s: [B,C,H,W]
        z = to_tensor(s) - 0.5
        if unpermuted:
            z = z.permute(0, 3, 1, 2)
        z = self.layers(z)
        # (B, ...) -> (B, -1)
        z = z.flatten(1)
        z = self.post_conv(z)
        z = AvgL1Norm(z)
        return z
    
    def zsa(self, zs, a):
        a = to_tensor(a)
        zsa = self.zsa_net(torch.cat([zs, a], 1))
        return zsa
    

class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((int(np.ceil(i / s)) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x