from typing import Any, Optional
import numpy as np
from torch import nn

from fcsrl.data import Batch, ReplayBuffer

class BaseAgent(nn.Module):

    def __init__(self):
        super().__init__()

    def train(self, mode: bool = True):
        self.training = mode

    def save_model(self, model_path: str):
        raise NotImplementedError

    def load_model(self, model_path: str):
        raise NotImplementedError
    
    def process_fn(
        self,
        batch: Batch,
        replay: ReplayBuffer,
        indices: np.ndarray,
    ) -> Batch:
        return batch

    def post_process_fn(
        self,
        batch: Batch,
        replay: ReplayBuffer,
        indices: np.ndarray,
    ):
        if hasattr(replay, "update_weight") and hasattr(batch, "weight"):
            replay.update_weight(indices, batch.weight)


    def forward(self, batch: Batch, state: Optional[Any] = None):
        # return batch of (action, ...)
        raise NotImplementedError


    def learn(self, batch: Batch, batch_size: Optional[int] = None):
        raise NotImplementedError


    def sync_weights(self):
        pass