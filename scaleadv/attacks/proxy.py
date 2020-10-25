from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from art.config import ART_NUMPY_DTYPE


class Proxy(object):

    def __init__(self, proxy: Callable, n: int, **proxy_kwargs):
        self.proxy = proxy
        self.n = n
        self.proxy_kwargs = proxy_kwargs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f'Only support a single input, but got shape {x.shape}.')
        x = self._augment(x)
        x = np.array(x, dtype=ART_NUMPY_DTYPE)
        return x

    def _augment(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PoolingProxy(Proxy):

    def __init__(self, pooling: nn.Module, n: int):
        super(PoolingProxy, self).__init__(pooling, n)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        batch = torch.as_tensor(x).repeat(self.n, 1, 1, 1)
        return self.proxy(batch)


class NoiseProxy(Proxy):

    def _augment(self, x: np.ndarray) -> np.ndarray:
        batch = np.repeat(x, repeats=self.n, axis=0)
        noise = self.proxy(**self.proxy_kwargs, size=batch.shape)
        return batch + noise
