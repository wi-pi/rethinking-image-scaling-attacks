from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from src.scaling import ScalingAPI


class NormalizationLayer(nn.Module):
    """A normalization layer prepends a neural network.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super(NormalizationLayer, self).__init__()

        # Register variables
        self.register_buffer('mean', torch.FloatTensor(mean)[..., None, None])
        self.register_buffer('std', torch.FloatTensor(std)[..., None, None])

    @classmethod
    def preset(cls, name: str):
        match name:
            case 'imagenet':
                from src.datasets.imagenet import IMAGENET_MEAN, IMAGENET_STD
                mean, std = IMAGENET_MEAN, IMAGENET_STD
            case _:
                raise ValueError(f'Cannot find preset name "{name}".')

        return cls(mean, std)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f'Only support a batch tensor of size (B, C, H, W), but got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'


class ScalingLayer(nn.Module):
    """A simple layer that scales down/up the inputs using the matrix approximation.
    """

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScalingLayer, self).__init__()
        self.register_buffer('cl', torch.FloatTensor(cl))
        self.register_buffer('cr', torch.FloatTensor(cr))

    @classmethod
    def from_api(cls, api: ScalingAPI):
        return cls(api.cl, api.cr)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr
