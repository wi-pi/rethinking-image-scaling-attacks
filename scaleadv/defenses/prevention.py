"""
This module implements prevention-based defenses as pooling layers.
"""
from abc import abstractmethod, ABC
from typing import Optional, Union, Tuple, TypeVar, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

T = TypeVar('T')


class Pooling(nn.Module, ABC):
    """The base class for all pooling based defenses.

    This kind of defense provides one function only:
      * Accept an input batch of shape (batch, channel, height, width).
      * Return a processed batch of the SAME shape.

    Args:
        kernel_size: size of pooling kernel, int or 2-tuple
        stride: pool stride, int or 2-tuple
        padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
        mask: indicate which pixels can be modified

    Note:
      * We provide only the instant defense, no more sampling will be conducted (e.g., on random defenses).
      * It's the attacker's job to compute the expectation of gradients (e.g., by average pooling and sampling).
    """

    # Preferred device to execute the pooling operation.
    preferred_device = torch.device('cuda')

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]],
            padding: Union[int, Tuple[int, int, int, int]],
            mask: Optional[np.ndarray] = None,
    ):
        super(Pooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.mask = 1 if mask is None else torch.tensor(mask).to(self.dev)

    @classmethod
    def like(cls: Type[T], pool: "Pooling") -> T:
        """Return a new pooling layer with the same parameters."""
        return cls(pool.kernel_size, pool.stride, pool.padding, pool.mask)

    @classmethod
    def auto(cls: Type[T], kernel_size: Union[int, Tuple[int, int]], mask: Optional[np.ndarray] = None) -> T:
        """Return a pooling layer with auto determined parameters that fit the kernel size."""
        kh, kw = _pair(kernel_size)
        pt, pl = kh // 2, kw // 2
        pb, pr = kh - pt - 1, kw - pl - 1
        return cls(kernel_size=(kh, kw), stride=1, padding=(pl, pr, pt, pb), mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pooling(x)
        z = y * self.mask + x * (1 - self.mask)
        return z

    @abstractmethod
    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_padding(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding, mode='reflect')

    def unfold(self, x: torch.Tensor) -> torch.Tensor:
        """Unfold a tensor to contiguous view of sub-windows.

        Given a tensor, returns sub-windows around each pixel.
        Input: (batch, channel, height, width)
        Output: (batch, channel, height, width, kernel_h * kernel_w)
        """
        x = self.apply_padding(x)
        for i, (k, s) in enumerate(zip(self.kernel_size, self.stride)):
            x = x.unfold(2 + i, k, s)
        x = x.contiguous().view(*x.shape[:4], -1)
        return x


class NonePooling(Pooling):
    """Nothing."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AveragePooling(Pooling):
    """Replace each pixel by the average of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_padding(x)
        x = F.avg_pool2d(x, self.kernel_size, stride=1)
        return x


class MinPooling(Pooling):
    """Replace each pixel by the minimum of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        return self._unfold(x).min(dim=-1).values


class MaxPool2d(Pooling):
    """Replace each pixel by the maximum of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        return self._unfold(x).max(dim=-1).values


class MedianPooling(Pooling):
    """Replace each pixel by the median of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        med = self.unfold(x).median(dim=-1).values
        return med


class RandomPooling(Pooling):
    """Replace each pixel by a random one of a window.

    TODO: generalize to other random distribution for torch.randint.
    TODO: explore a more efficient implementation.
    """

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = B, C, H, W = x.shape
        # generate index
        idx_c = torch.arange(B * C)[:, None].repeat(1, H * W).flatten()
        idx_h = self.gen_idx(H, self.padding[0], in_shape, expand=True)
        idx_w = self.gen_idx(W, self.padding[2], in_shape, expand=False)
        # padding & take
        x = self.apply_padding(x)
        x = x.view(-1, *x.shape[2:])[idx_c, idx_h, idx_w].view(in_shape)
        return x

    def gen_idx(self, n, gap, shape, expand=False):
        B, C, H, W = shape
        # get index of valid pixels (kernel center)
        idx = torch.arange(n)
        idx = idx[:, None] if expand else idx
        idx = idx + gap + torch.randint(-gap, gap + 1, (B, H, W))
        # duplicate along channel
        idx = idx[:, None, ...].repeat(1, C, 1, 1).flatten()
        return idx
