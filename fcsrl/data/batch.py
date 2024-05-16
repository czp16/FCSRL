from typing import Union, Optional
import torch
import numpy as np


class Batch(object):
    """Suggested keys: [obs, act, rew, terminate, trunc, obs_next, info]"""

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __len__(self) -> int:
        length = min([
            len(self.__dict__[k]) for k in self.__dict__.keys()
            if self.__dict__[k] is not None])
        return length

    def __getitem__(self, index: Union[int, np.ndarray]):
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                b.update(**{k: self.__dict__[k][index]})
        return b

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def append(self, batch):
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        
        for k in self.__dict__.keys():
            if k not in batch.__dict__.keys():
                if isinstance(self.__dict__[k], np.ndarray):
                    value_shape = self.__dict__[k].shape[1:]
                    batch.__dict__[k] = np.zeros([len(batch), *value_shape])
                elif isinstance(self.__dict__[k], torch.Tensor):
                    value_shape = self.__dict__[k].shape[1:]
                    batch.__dict__[k] = torch.zeros([len(batch), *value_shape]).to(self.__dict__[k].device)
        
        for k in batch.__dict__.keys():
            if batch.__dict__[k] is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = batch.__dict__[k]
            elif isinstance(batch.__dict__[k], np.ndarray):
                self.__dict__[k] = np.concatenate([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], torch.Tensor):
                self.__dict__[k] = torch.cat([
                    self.__dict__[k], batch.__dict__[k]])
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            else:
                s = 'No support for append with type'\
                    + str(type(batch.__dict__[k]))\
                    + 'in class Batch.'
                raise TypeError(s)
                

    def sampler(self, size: Optional[int]=None, permute: bool=True):
        length = self.__len__()
        if size is None:
            size = length
        temp = 0
        if permute:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        while temp < length:
            yield self[index[temp:temp + size]]
            temp += size

    def ensemble_sampler(self, size: Optional[int]=None, num_ensemble: int=1):
        '''
        for bootstrap, yield [num_ensemble, size, ...]
        '''
        length = self.__len__()
        if size is None:
            size = length
        temp = 0
        index = np.random.randint(length, size=(num_ensemble, length))
        while temp < length:
            yield self[index[:,temp:temp+size]]
            temp += size