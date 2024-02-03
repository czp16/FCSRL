from torch import nn

class BaseAgent(nn.Module):

    def __init__(self):
        super().__init__()

    def train(self, mode=True):
        self.training = mode

    def save_model(self, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError
    

    def process_fn(self, batch, replay, indices):
        return batch

    def post_process_fn(self, batch, replay, indices):
        if hasattr(replay, "update_weight") and hasattr(batch, "weight"):
            replay.update_weight(indices, batch.weight)


    def forward(self, batch, state=None):
        # return batch of (action, ...)
        raise NotImplementedError


    def learn(self, batch, batch_size=None):
        raise NotImplementedError


    def sync_weights(self):
        pass