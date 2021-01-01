import pytest
from torch.utils.data import Subset

from scaleadv.datasets import get_imagenet
from scaleadv.datasets.transforms import Align


class TestImageNet(object):
    ratio_list = [2, 3, 4, 5, 6]
    ratio_size = [6555, 599, 185, 160, 170]

    def test_val_size(self):
        ds = get_imagenet('val')
        assert len(ds) == 50000

    @pytest.mark.parametrize('ratio,size', zip(ratio_list, ratio_size))
    def test_ratio_size(self, ratio, size):
        ds = get_imagenet(f'val_{ratio}')
        assert len(ds) == size


class TestAlignment(object):
    align_list = [224, 555, 33]
    ratio_list = [2, 3, 4]

    @pytest.mark.parametrize('align', align_list)
    def test_auto_align(self, align):
        trans = Align(align)
        ds = get_imagenet('val', transform=trans)
        for x, _ in Subset(ds, list(range(10))):
            assert x.size[0] % align == 0
            assert x.size[1] % align == 0

    @pytest.mark.parametrize('align', align_list)
    @pytest.mark.parametrize('ratio', ratio_list)
    def test_ratio_align(self, align, ratio):
        trans = Align(align, ratio)
        ds = get_imagenet('val', transform=trans)
        for x, _ in Subset(ds, list(range(10))):
            assert x.size[0] == x.size[1] == align * ratio
