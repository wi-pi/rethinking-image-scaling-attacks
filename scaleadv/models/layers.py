from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class NormalizationLayer(nn.Module):

    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)[None, :, None, None]
        std = torch.as_tensor(std, dtype=torch.float32)[None, :, None, None]
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Expect a batch tensor of size (B, C, H, W). Got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'


class _Pool2d(nn.Module):

    def __init__(self, kernel_size: int, stride: int, padding: int, mask: Optional[np.ndarray] = None):
        """
        Args:
            kernel_size: size of pooling kernel, int or 2-tuple
            stride: pool stride, int or 2-tuple
            padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
            mask: indicate which pixels can be modified
        """
        super(_Pool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32)


class MedianPool2d(_Pool2d):

    def forward(self, x: torch.Tensor):
        x_raw = x
        x = F.pad(x, self.padding, mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        if self.mask is not None:
            x = x_raw * (1 - self.mask) + x * self.mask
        return x


class RandomPool2d(_Pool2d):

    def _gen_idx(self, n, gap, shape, expand=False):
        B, C, H, W = shape
        # get index of valid pixels (kernel center)
        idx = torch.arange(n)
        idx = idx[:, None] if expand else idx
        idx = idx + gap + torch.randint(-gap, gap + 1, (B, H, W))
        # duplicate along channel
        idx = idx[:, None, ...].repeat(1, C, 1, 1).flatten()
        return idx

    def forward(self, x):
        x_raw = x
        padding = pl, _, pt, _ = self._padding(x)
        in_shape = B, C, H, W = x.shape  # Note this is the shape of x BEFORE padding.
        # generate index
        idx_c = torch.arange(B * C)[:, None].repeat(1, H * W).flatten()
        idx_h = self._gen_idx(H, pt, in_shape, expand=True)
        idx_w = self._gen_idx(W, pl, in_shape, expand=False)
        # padding & take
        x = F.pad(x, padding, mode='reflect')
        x = x.reshape(-1, *x.shape[2:])[idx_c, idx_h, idx_w].reshape(in_shape)
        if self.mask is not None:
            x = x_raw * (1 - self.mask) + x * self.mask
        return x
