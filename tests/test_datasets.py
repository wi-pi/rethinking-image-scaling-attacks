import pytest
import torch
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

from src.datasets import get_imagenet, get_celeba
from src.datasets.transforms import Align


class TestImageNet(object):

    def test_val_size(self):
        ds = get_imagenet('val')
        assert len(ds) == 50000


class TestCelebA(object):

    def test_attr_one(self):
        ds = get_celeba('test', attrs=['Young'], transform=T.ToTensor())
        for _, y in Subset(ds, list(range(10))):
            assert isinstance(y, int)

        loader = DataLoader(ds, batch_size=5)
        for _, y in loader:
            assert y.shape == (5,)
            break

    def test_attr_multi(self):
        ds = get_celeba('test', attrs=['Eyeglasses', 'Young'], transform=T.ToTensor())
        for _, y in Subset(ds, list(range(10))):
            assert isinstance(y, torch.Tensor)
            assert len(y) == 2

        loader = DataLoader(ds, batch_size=5)
        for _, y in loader:
            assert y.shape == (5, 2)
            break


class TestAlignment(object):
    align_list = [224, 555, 33]
    ratio_list = [2, 3, 4]

    @pytest.mark.parametrize('align', align_list)
    def test_auto_align(self, align):
        ds = get_imagenet('val', transform=Align(align))
        for x, _ in Subset(ds, list(range(10))):
            assert x.size[0] % align == 0
            assert x.size[1] % align == 0

    @pytest.mark.parametrize('align', align_list)
    @pytest.mark.parametrize('ratio', ratio_list)
    def test_ratio_align(self, align, ratio):
        ds = get_imagenet('val', transform=Align(align, ratio))
        for x, _ in Subset(ds, list(range(10))):
            assert x.size[0] == x.size[1] == align * ratio
