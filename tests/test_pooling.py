from typing import Tuple, Union

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as F
from PIL import Image

from scaleadv.defenses.prevention import Pooling, NonePooling, MedianPooling

Tuple2 = Union[int, Tuple[int, int]]
Tuple3 = Tuple[int, int, int]


def _load_img(name: str) -> np.ndarray:
    pic = Image.open(name)
    arr = F.to_tensor(pic)[None, ...].cuda()
    return arr


class TestPooling(object):
    batch_list = [1, 2, 4]
    shape_list = [(3, 20, 20), (3, 30, 50), (3, 224, 224)]
    kernel_list = [3, 4, 5, 6, 7]
    cls_list = [NonePooling, MedianPooling]

    @pytest.mark.parametrize('batch', batch_list)
    @pytest.mark.parametrize('shape', shape_list)
    @pytest.mark.parametrize('kernel', kernel_list)
    @pytest.mark.parametrize('cls', cls_list)
    def test_consistency(self, batch: int, shape: Tuple3, kernel: Tuple2, cls: Pooling):
        x = torch.rand(batch, *shape).cuda()
        pooling = cls.auto(kernel)
        y = pooling(x)
        assert x.shape == y.shape
        assert x.dtype == y.dtype
        assert x.device == y.device


class TestPoolingOnAttack(object):
    source = _load_img('./testcases/scaling-attack/source.png')
    attack = _load_img('./testcases/scaling-attack/attack.png')

    cls_list = [MedianPooling]
    kernel_list = [3, 5, 7, 9]

    @pytest.mark.parametrize('cls', cls_list)
    @pytest.mark.parametrize('kernel', kernel_list)
    def test_effectiveness(self, cls: Pooling, kernel: Tuple2):
        pooling = cls.auto(kernel)
        mse_attack = torch.square(self.source - self.attack).mean().item()
        mse_defense = torch.square(self.source - pooling(self.attack)).mean().item()
        assert mse_defense < mse_attack * 0.5
