from math import ceil

import numpy as np
import torch
import torchvision.transforms.functional_pil as F_pil
from PIL.Image import Image as ImageType


class Align(object):
    """Align the input PIL Image to a multiple of the specified size.

    Args:
        align: the size to align, e.g., 224.
        ratio: directly scale to this multiple of `align`, set `None` to align to the nearest larger multiple.
        square: align to a square.
    """

    def __init__(self, align: int, ratio: int | None = None, square: bool = False):
        self.align = align
        self.ratio = ratio
        self.square = square

    def __call__(self, img: ImageType) -> ImageType:
        assert isinstance(img, ImageType)

        # determine output shape
        if self.ratio is not None:
            w = h = self.align * self.ratio
        else:
            w, h = map(self._round_up, img.size)

        # determine output shape if square required
        if self.square:
            w = h = min(w, h)

        return F_pil.resize(img, [w, h])

    def _round_up(self, x: int) -> int:
        return self.align * ceil(x / self.align)


class ToNumpy(object):

    def __init__(self, batch: bool = True):
        self.batch = batch

    def __call__(self, x: torch.Tensor):
        x = x.numpy().astype(np.float32)
        if self.batch:
            x = x[None]
        return x
