from math import ceil
from typing import Optional

import torchvision.transforms.functional as F
from PIL.Image import Image as ImageType


class Align(object):
    """Align the input PIL Image to a multiple of the specified size.

    Args:
        align: the size to align, e.g., 224.
        ratio: directly scale to this multiple of `align`, set `None` to align to the nearest larger multiple.
        square: align to a square.
    """

    def __init__(self, align: int, ratio: Optional[int] = None, square: bool = False):
        self.align = align
        self.ratio = ratio
        self.square = square

    def __call__(self, img: ImageType):
        if self.ratio is not None:
            w = h = self.align * self.ratio
        else:
            w, h = map(self._round_up, img.size)
        if self.square:
            w = h = min(w, h)
        return F.resize(img, (w, h))

    def _round_up(self, x: int):
        return self.align * ceil(x / self.align)
