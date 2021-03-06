import numpy as np
import piq
import torch

from src.defenses.prevention import Pooling, MinPooling
from src.scaling import ScalingAPI


class Detection(object):
    name = 'detection'

    def score(self, x: np.ndarray):
        y = self.reveal(x)
        x, y = x * 255, y * 255
        mse = np.square(x - y).mean()
        ssim = piq.ssim(*map(torch.as_tensor, [x, y]), data_range=255.0).item()
        return mse, ssim

    def reveal(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 4 and x.shape[:2] == (1, 3)
        assert 0 <= x.min() and x.max() <= 1
        y = self._reveal(x)
        assert x.shape == y.shape
        return y

    def _reveal(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented


class Unscaling(Detection):
    name = 'unscaling'

    def __init__(self, scale_down: ScalingAPI, scale_up: ScalingAPI, pooling: Pooling):
        self.scale_down = scale_down
        self.scale_up = scale_up
        self.pooling = pooling

    def _reveal(self, x: np.ndarray) -> np.ndarray:
        # to (1, 3, h, w)[0, 1] tensor && pooling
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
        x = self.pooling(x)

        # from (1, 3, h, w)[0, 1] tensor to (3, h, w)[0, 1] array && scale down && scale up
        x = x[0].cpu().numpy()
        x = self.scale_down(x)
        x = self.scale_up(x)

        # from (h, w, 3)[0, 1] array to (1, 3, h, w)[0, 1] array
        return x[None, ...]


class MinimumFilter(Detection):
    name = 'minimum-filtering'

    def __init__(self):
        self.min_pool = MinPooling(kernel_size=2, stride=1, padding=(1, 0, 1, 0))

    def _reveal(self, x: np.ndarray) -> np.ndarray:
        # to (1, 3, h, w)[0, 1] tensor && pooling
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.min_pool(x)

        # to array
        x = x.cpu().numpy()
        return x
