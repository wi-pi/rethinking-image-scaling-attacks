from typing import Sequence, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn

from scaleadv.datasets.imagenet import IMAGENET_STD, IMAGENET_MEAN
from scaleadv.scaling import ScalingAPI

T = TypeVar('T')


class NormalizationLayer(nn.Module):
    """A normalization layer prepends a neural network."""

    PRESET = {
        'imagenet': (IMAGENET_MEAN, IMAGENET_STD)
    }

    @classmethod
    def preset(cls: Type[T], name: str) -> T:
        if name not in cls.PRESET:
            raise ValueError(f'Cannot find preset name "{name}".')
        mean, std = cls.PRESET[name]
        return cls(mean, std)

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super(NormalizationLayer, self).__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32)[:, None, None])
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32)[:, None, None])

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Only support a batch tensor of size (B, C, H, W), but got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'


class ScalingLayer(nn.Module):
    """A simple layer that scales down/up the inputs."""

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScalingLayer, self).__init__()
        self.register_buffer('cl', torch.tensor(cl, dtype=torch.float32))
        self.register_buffer('cr', torch.tensor(cr, dtype=torch.float32))

    @classmethod
    def from_api(cls: Type[T], api: ScalingAPI) -> T:
        return cls(api.cl, api.cr)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr
