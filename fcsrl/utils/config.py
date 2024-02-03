import torch

class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        pass

    def select_device(self, cudaid=None):
        if cudaid is None:
            cudaid = self.hyperparams['misc']['cudaid']
        torch.set_num_threads(4)
        if cudaid >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (cudaid))
        else:
            Config.DEVICE = torch.device('cpu')