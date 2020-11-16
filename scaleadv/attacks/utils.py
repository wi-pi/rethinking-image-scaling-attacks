from typing import Union

import numpy as np
import numpy.linalg as LA
import torch
import torch.nn.functional as F


def get_mask_from_cl_cr(cl: np.ndarray, cr: np.ndarray) -> np.ndarray:
    cli, cri = map(LA.pinv, [cl, cr])
    shape = (cli.shape[1], cri.shape[0])
    mask = cli @ np.ones(shape) @ cri
    return mask.round().astype(np.uint8)


def _mask_diff(x: np.ndarray, pooling: "RandomPool2d", n: int):
    assert x.ndim == 4 and x.shape[0] == 1
    x = torch.as_tensor(x, dtype=torch.float32)
    p = pooling(x, n)
    diff = (p - x).permute(2, 3, 0, 1)
    diff = diff[pooling.mask > 0, ...].flatten()
    return diff


def mask_mad(x: np.ndarray, pooling: "RandomPool2d", n=100):
    assert x.ndim == 4 and x.shape[0] == 1
    return _mask_diff(x, pooling, n).abs().mean().cpu().item()


def mask_hist(x: np.ndarray, pooling: "RandomPool2d", n: int = 100, bins: int = 100, min: int = -1, max: int = 1):
    assert x.ndim == 4 and x.shape[0] == 1
    diff = _mask_diff(x, pooling, n)
    hist = torch.histc(diff, bins=bins, min=min, max=max).numpy()
    xs = np.arange(min, max, (max - min) / bins)
    return xs, hist, diff.cpu().numpy()


def estimate_mad(x: Union[np.ndarray, torch.Tensor], k: int):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = F.pad(x, [k // 2] * 4, mode='reflect')
    y = y.unfold(2, k, 1).unfold(3, k, 1)
    d = y - x[..., None, None]
    mad = d.abs().mean().cpu().item()
    return mad
