from typing import Sequence

import torch
import torch.nn as nn

from scaleadv.datasets.imagenet import IMAGENET_STD, IMAGENET_MEAN


class NormalizationLayer(nn.Module):
    """A normalization layer prepends a neural network.
    """
    PRESET = {
        'imagenet': (IMAGENET_MEAN, IMAGENET_STD)
    }

    @classmethod
    def preset(cls, name: str):
        if name not in cls.PRESET:
            raise ValueError(f'Cannot find preset name "{name}".')
        mean, std = cls.PRESET[name]
        return cls(mean, std)

    @staticmethod
    def make_parameter(x: Sequence[float]):
        x = torch.tensor(x, dtype=torch.float32)[None, :, None, None]
        x = nn.Parameter(x, requires_grad=False)
        return x

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super(NormalizationLayer, self).__init__()
        self.mean = self.make_parameter(mean)
        self.std = self.make_parameter(std)

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Only support a batch tensor of size (B, C, H, W), but got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'
