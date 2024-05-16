from typing import List, Optional, Any, Dict
import numpy as np
from fcsrl.data import Batch

class Storage:
    def __init__(self, size: int):
        self._maxsize = size
        self._size = 0
        self._index = 0
        self._keys = []
        self.reset()

    def __len__(self):
        return self._size

    def __del__(self):
        for k in self._keys:
            v = getattr(self, k)
            del v

    def __getitem__(self, index):
        batch_dict = dict(
            zip(self._keys, [getattr(self,k)[index] for k in self._keys]))
        return batch_dict

    def get_batch(self, index):
        return Batch(**self[index])

    def reset(self):
        self._index = self._size = 0

    def set_placeholder(self, key: str, value: Any):
        if isinstance(value, np.ndarray):
            setattr(self, key, np.zeros((self._maxsize, *value.shape), dtype=np.float32))
        elif isinstance(value, dict):
            setattr(self, key, np.array([{} for _ in range(self._maxsize)]))
        elif np.isscalar(value):
            setattr(self, key, np.zeros((self._maxsize,)))
    
    def add(self, data: Dict[str, Any]):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v)
            getattr(self, k)[self._index] = v

        self._size = min(self._size + 1, self._maxsize)
        self._index = (self._index + 1) % self._maxsize


    def add_list(self, data: Dict[str, Any], length: int):
        assert isinstance(data, dict)

        _tmp_idx = self._index + length

        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v[0])
            
            assert v.shape[0] == length
            
            if _tmp_idx < self._maxsize:
                getattr(self, k)[self._index:_tmp_idx] = v
                to_indices = np.arange(self._index, _tmp_idx)
            else:
                getattr(self, k)[self._index:] = v[:self._maxsize - self._index]
                getattr(self, k)[:_tmp_idx - self._maxsize] = v[self._maxsize - self._index:]
                to_indices = np.concatenate([
                    np.arange(self._index, self._maxsize),
                    np.arange(0, _tmp_idx - self._maxsize)
                ])

        self._size = min(self._size + length, self._maxsize)
        self._index = _tmp_idx % self._maxsize

        return to_indices


    def update(self, buffer):
        i = begin = buffer._index % len(buffer)
        to_indices = []
        
        while True:
            to_indices.append(self._index)
            self.add(buffer[i])
            i = (i+1)% len(buffer)
            if i == begin:
                break

        return np.array(to_indices)


    def sample(self, batch_size: int):
        if batch_size > 0:
            indices = np.random.choice(self._size, batch_size)
        else: # sample all available data when batch_size=0
            indices = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        return self.get_batch(indices), indices


class CacheBuffer(Storage):

    def __init__(self):
        super().__init__(size=0)

    def add(self, data: Dict[str, Any]):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

        self._index += 1
        self._size += 1

    def reset(self):
        self._index = self._size = 0
        for k in self._keys:
            setattr(self, k, [])


class VectorCacheBuffer:
    def __init__(self, num_buffer: int):
        self.num_buffer = num_buffer
        self.buffers = [CacheBuffer() for _ in range(self.num_buffer)]

    def __getitem__(self, buffer_id):
        return self.buffers[buffer_id]

    def add(self, data: Dict, buffer_id: int):
        self.buffers[buffer_id].add(data)

    def reset(self, buffer_ids: Optional[int] = None):
        if buffer_ids is None:
            buffer_ids = np.arange(self.num_buffer)
        if np.isscalar(buffer_ids):
            buffer_ids = [buffer_ids] 
        
        for id in buffer_ids:
            self.buffers[id].reset()


class ReplayBuffer(Storage):
    def __init__(self, size):
        super().__init__(size)

    def prev(self, index):
        '''
        the prev of beginning at the episode is itself.
        '''
        index = (index - 1) % self._size
        terminates = getattr(self, 'terminate')[index].astype(int)
        truncs = getattr(self, 'trunc')[index].astype(int)
        
        end_flag = np.logical_or(terminates, truncs) | (index == self._size - 1)
        return (index + end_flag) % self._size

    def next(self, index):
        '''
        the next of end at the episode is itself.
        '''
        terminates = getattr(self, 'terminate')[index].astype(int)
        truncs = getattr(self, 'trunc')[index].astype(int)
        end_flag = np.logical_or(terminates, truncs) | (index == self._size - 1)
        return (index + (1 - end_flag)) % self._size

    def update(self, buffer: CacheBuffer):
        if len(buffer) == 0 or self._maxsize == 0:
            return np.array([], dtype=int)

        data_dict = {}
        for k in buffer._keys:
            _tmp = getattr(buffer, k)[0]
            if np.isscalar(_tmp) or isinstance(_tmp, np.ndarray):
                data_dict[k] = np.stack(getattr(buffer, k), 0)
            else:
                raise NotImplementedError(f"Not supported data type: {_tmp}")
        
        length = len(buffer)
        return self.add_list(data_dict, length)