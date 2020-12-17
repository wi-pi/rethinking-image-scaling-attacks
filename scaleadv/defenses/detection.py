import numpy as np
import piq
import torch
import torchvision.transforms.functional as F
from scaling.ScalingApproach import ScalingApproach

from scaleadv.models.layers import Pool2d, MinPool2d


class Detection(object):
    name = 'detection'

    def score(self, x: np.ndarray):
        y = self._reveal(x)
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

    def __init__(self, scale_down: ScalingApproach, scale_up: ScalingApproach, pooling: Pool2d):
        self.scale_down = scale_down
        self.scale_up = scale_up
        self.pooling = pooling

    def _reveal(self, x: np.ndarray) -> np.ndarray:
        # to (1, 3, h, w)[0, 1] tensor && pooling
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.pooling(x)

        # from (1, 3, h, w)[0, 1] tensor to (h, w, 3)[0, 255] array && scale down && scale up
        x = np.array(F.to_pil_image(x[0]))
        x = self.scale_down.scale_image(x)
        x = self.scale_up.scale_image(x)

        # from (h, w, 3)[0, 255] array to (1, 3, h, w)[0, 1] array
        x = F.to_tensor(x).numpy()[None, ...]
        return x


class MinimumFilter(Detection):
    name = 'minimum filtering'

    def __init__(self):
        self.min_pool = MinPool2d(kernel_size=2, stride=1, padding=(1, 0, 1, 0))

    def _reveal(self, x: np.ndarray) -> np.ndarray:
        # to (1, 3, h, w)[0, 1] tensor && pooling
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.min_pool(x)

        # to array
        x = x.cpu().numpy()
        return x
