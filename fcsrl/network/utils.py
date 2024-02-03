from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: Optional[int] = None,
        norm: str = "LayerNorm",
        activation: str = "SiLU",
    ):
        
        super().__init__()

        norm = getattr(nn, norm)
        activation = getattr(nn, activation)

        layers = []
        last_dim = input_dim
        for next_dim in hidden_dims:
            layers += [
                nn.Linear(last_dim, next_dim, bias=False),
                norm(next_dim),
                activation(),
            ]
            last_dim = next_dim

        if output_dim:
            layers.append(nn.Linear(last_dim, output_dim)) # no activation
        self.layers = nn.Sequential(*layers)
        # TODO: weight init

    def forward(self, input_):
        out_ = self.layers(input_)
        return out_
