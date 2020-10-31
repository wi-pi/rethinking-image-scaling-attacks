from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from art.config import ART_NUMPY_DTYPE

from scaleadv.models.scaling import ScaleNet


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

    def __init__(self, pooling: nn.Module, n: int, x_big: np.ndarray, scale: ScaleNet):
        super(PoolingProxy, self).__init__(pooling, n)
        self.x_big = x_big
        self.scale = scale

    def _augment(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            batch = torch.as_tensor(self.x_big).repeat(self.n, 1, 1, 1).cpu()
            batch = self.proxy(batch).cuda()
            batch = self.scale(batch).cpu()
        return batch.detach().clone()


class NoiseProxy(Proxy):

    def _augment(self, x: np.ndarray) -> np.ndarray:
        batch = np.repeat(x, repeats=self.n, axis=0)
        noise = self.proxy(**self.proxy_kwargs, size=batch.shape)
        return batch + noise
