"""
This module implements prevention-based defenses as pooling layers.
"""
from abc import abstractmethod, ABC
from typing import Optional, Union, Tuple, TypeVar, Type, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import signal
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
        self.mask = 1 if mask is None else torch.tensor(mask).to(self.preferred_device)
        logger.info(f'Created {self.__class__.__name__}: kernel {kernel_size}, stride {stride}, '
                    f'padding {padding}, mask {mask is not None}.')

    @classmethod
    def like(cls: Type[T], pool: "Pooling") -> T:
        """Return a new pooling layer with the same parameters."""
        return cls(pool.kernel_size, pool.stride, pool.padding, pool.mask)

    @classmethod
    def auto(cls: Type[T], kernel_size: Union[int, Tuple[int, int]], mask: Optional[np.ndarray] = None, **kwargs) -> T:
        """Return a pooling layer with auto determined parameters that fit the kernel size."""
        kh, kw = _pair(kernel_size)
        pt, pl = kh // 2, kw // 2
        pb, pr = kh - pt - 1, kw - pl - 1
        return cls(kernel_size=(kh, kw), stride=1, padding=(pl, pr, pt, pb), mask=mask, **kwargs)

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


class MinPooling(Pooling):
    """Replace each pixel by the minimum of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        return self.unfold(x).min(dim=-1).values


class MaxPooling(Pooling):
    """Replace each pixel by the maximum of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        return self.unfold(x).max(dim=-1).values


class MedianPooling(Pooling):
    """Replace each pixel by the median of a window."""

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        med = self.unfold(x).median(dim=-1).values
        return med


class QuantilePooling(Pooling):
    """Replace each pixel by the median of a window."""

    def __init__(self, *args, **kwargs):
        super(QuantilePooling, self).__init__(*args, **kwargs)

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        # out = x.median(dim=-1).values

        # center quantile
        low = x.quantile(0.3, dim=-1)[..., None]
        med = x.median(dim=-1).values[..., None]
        high = x.quantile(0.7, dim=-1)[..., None]
        ind = (low <= x) * (x <= high) * (1 - (x - med).abs())

        out = (x * ind).sum(dim=-1) / ind.sum(dim=-1)

        return out


class RandomPooling(Pooling, ABC):
    """The base class for all random pooling based defenses.

    Replace each pixel by an (i.i.d) random one of a window.

    Additional Args:
        prob_kwargs: a dict containing args for generating the probability kernel.

    Additional Properties:
        required_prob_kwargs: required args in prob_kwargs.
        prob_kernel: a tensor of shape kernel_size.
    """

    required_prob_kwargs: List[str] = []

    def __init__(self, *args, prob_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        super(RandomPooling, self).__init__(*args, **kwargs)
        logger.info(f'Prob kwargs: {prob_kwargs}')
        self.prob_kwargs = self._check_prob_kwargs(prob_kwargs)
        self.prob_kernel = self._prob_kernel_2d()
        logger.info(f'Use prob kernel:\n {self.prob_kernel.numpy()}')

    def _check_prob_kwargs(self, prob_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        for req in self.required_prob_kwargs:
            if req not in prob_kwargs:
                raise ValueError(f'Parameter "{req}" required, but not found in {prob_kwargs}.')
            setattr(self, req, prob_kwargs[req])
        return prob_kwargs or {}

    @abstractmethod
    def _prob_kernel_1d(self, size) -> np.ndarray:
        raise NotImplementedError

    def _prob_kernel_2d(self) -> torch.Tensor:
        k = list(map(self._prob_kernel_1d, self.kernel_size))
        k = torch.tensor(np.outer(*k), dtype=torch.float32)
        return k / k.sum()

    def randint(self, low: int, high: int, size: Tuple[int, ...]) -> torch.Tensor:
        a = np.arange(low, high)
        p = self._prob_kernel_1d(len(a))
        out = np.random.choice(a, size, p=p)
        return torch.tensor(out, dtype=torch.long)

    def _gen_idx(self, n, gap, shape, expand=False) -> torch.Tensor:
        B, C, H, W = shape
        # get index of valid pixels (kernel center)
        idx = torch.arange(n)
        idx = idx[:, None] if expand else idx
        idx = idx + gap + self.randint(-gap, gap + 1, (B, H, W))
        # duplicate along channel
        idx = idx[:, None, ...].repeat(1, C, 1, 1).flatten()
        return idx

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = B, C, H, W = x.shape
        # generate indices
        idx_c = torch.arange(B * C)[:, None].repeat(1, H * W).flatten()
        idx_h = self._gen_idx(H, self.padding[0], in_shape, expand=True)
        idx_w = self._gen_idx(W, self.padding[2], in_shape, expand=False)
        # padding & take
        x = self.apply_padding(x)
        x = x.view(-1, *x.shape[2:])[idx_c, idx_h, idx_w].view(in_shape)
        return x


class RandomPoolingUniform(RandomPooling):
    required_prob_kwargs = []

    def _prob_kernel_1d(self, size) -> np.ndarray:
        return np.ones(size) / size


class RandomPoolingGaussian(RandomPooling):
    required_prob_kwargs = ['std']

    def _prob_kernel_1d(self, size) -> np.ndarray:
        k = signal.windows.gaussian(size, self.std)
        return k / k.sum()


class RandomPoolingLaplacian(RandomPooling):
    required_prob_kwargs = ['std']

    def _prob_kernel_1d(self, size) -> np.ndarray:
        k = signal.windows.general_gaussian(size, 0.5, self.std)
        return k / k.sum()


# noinspection PyShadowingBuiltins
class SoftMedian(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        output = input.median(dim=-1)
        ctx.save_for_backward(input, output.indices, output.values)
        return output.values

    @staticmethod
    def backward(ctx, grad_output):
        # prepare variables
        input, median_indices, median = ctx.saved_tensors
        median_indices, median, grad_output = map(lambda x: x[..., None], [median_indices, median, grad_output])
        grad_input = torch.zeros_like(input)

        # make grad_input more useful
        # 1. deviation (value) to the median
        deviation_to_median = (input - median).abs()
        # 2. deviation (sorted index) to the median
        deviation_to_median_index = (input.argsort().float() - median_indices).abs()
        # 3. how many pixels do we want to change in each block?
        # from IPython import embed; embed(using=False); exit()
        # nb_pixels = (torch.randn_like(median) * 2).round().abs()
        useful_grad = grad_output  # / deviation_to_median.exp()

        # for grad > 0, we extend grads to RHS of median
        # indicator_has_grad = (grad_output > 0) & (input >= median) & (deviation_to_median_index <= nb_pixels)
        # grad_input += indicator_has_grad * useful_grad

        # for grad < 0, we extend grads to LHS of median
        # indicator_has_grad = (grad_output < 0) & (input <= median) & (deviation_to_median_index <= nb_pixels)
        # grad_input += indicator_has_grad * useful_grad
        grad_input += useful_grad * (deviation_to_median_index <= 5)

        return grad_input


class SoftMedianPooling(Pooling):

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        return SoftMedian.apply(x)


POOLING_MAPS = {
    'none': NonePooling,
    'min': MinPooling,
    'max': MaxPooling,
    'median': MedianPooling,
    'quantile': QuantilePooling,
    'softmedian': SoftMedianPooling,
    'uniform': RandomPoolingUniform,
    'gaussian': RandomPoolingGaussian,
    'laplacian': RandomPoolingLaplacian,
}
if __name__ == '__main__':
    mask = torch.tensor([[0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0]]).reshape(1, 1, 5, 5).float().cuda()
    p = SoftMedianPooling(3, 1, 1, mask).cuda()

    # a = torch.randint(20, (1,1,5,5)).float().cuda().requires_grad_()
    a = torch.tensor([[1, 4, 3, 4, 1],
                      [1, 4, 1, 4, 1],
                      [1, 4, 4, 4, 1],
                      [1, 2, 3, 4, 5],
                      [6, 6, 6, 7, 7]]).reshape(1, 1, 5, 5).float().cuda().requires_grad_()
    b = p(a).cuda()
    (b * mask).sum().backward()
    from IPython import embed;

    embed(using=False);
    exit()
