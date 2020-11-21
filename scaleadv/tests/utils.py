import os
from collections import OrderedDict
from math import ceil
from typing import Tuple

import numpy as np
import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from lpips import LPIPS
from prettytable import PrettyTable

from scaleadv.models.layers import MedianPool2d, RandomPool2d
from scaleadv.models.scaling import ScaleNet


def resize_to_224x(img: Image.Image, scale: int = 0, square: bool = False):
    w, h = img.size
    if scale:
        w = h = 224 * scale
    else:
        w = 224 * ceil(w / 224)
        h = 224 * ceil(h / 224)
    if square:
        w = h = min(w, h)
    return img.resize((w, h))


class Evaluator(object):
    DIFF_FIELDS = ['Y', 'Y_MED', 'Y_RND', 'L-INF', 'L-2', 'PSNR', 'SSIM', 'MS-SSIM', 'LPIPS']

    def __init__(self, scale_net: ScaleNet, class_net: nn.Module, pooling_args: Tuple, nb_samples: int = 200):
        self.scale_net = scale_net
        self.class_net = class_net
        self.pooling_med = MedianPool2d(*pooling_args)
        self.pooling_rnd = RandomPool2d(*pooling_args)
        self.nb_samples = nb_samples
        self.lpips = LPIPS(net='alex', verbose=False).cuda()

    @staticmethod
    def _check(x: np.ndarray) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
        assert x.dtype == np.float32
        assert 0 <= x.min() and x.max() <= 1
        return torch.as_tensor(x, dtype=torch.float32)

    def summary(self, stats: OrderedDict, y_src: int, y_adv: int, tag: str):
        tab = PrettyTable([tag] + self.DIFF_FIELDS)
        # generate each row
        for k, data in stats.items():
            row = [k]
            # predict (normal)
            p = data['Y']
            row.append(p)
            # predict (median)
            p = data['Y_MED']
            p = '-' if p is None else p
            row.append(p)
            # predict (random)
            p = data['Y_RND']
            p = '-' if p is None else f'{np.mean(p == y_src):.2%} / {np.mean(p == y_adv):.2%}'
            row.append(p)
            # distance metrics
            for field in self.DIFF_FIELDS[3:]:
                if field in data:
                    row.append(f'{data[field].cpu().item():.3f}')
                else:
                    row.append('-')
            tab.add_row(row)
        return tab

    def predict(self, x: torch.Tensor, scale: bool, pooling: str = None):
        """Predict a given image with pooling support.

        Args:
            x: input image tensor of shape [1, 3, H, W].
            scale: True if input image is large.
            pooling: 'med', 'rnd', None for Median/Random/No pooling operation.

        Returns:
            np.ndarray containing predicted labels (multiple for random pooling).
        """
        if pooling is not None:
            assert scale, 'Cannot pool if no scale.'
        with torch.no_grad():
            if pooling is not None:
                if pooling == 'med':
                    func, n = self.pooling_med, 1
                elif pooling == 'rnd':
                    func, n = self.pooling_rnd, self.nb_samples
                else:
                    raise NotImplementedError
                x = func(x, n)
            if scale:
                x = self.scale_net(x)
            pred = self.class_net(x).argmax(1).cpu()
        return pred.numpy()

    def eval(
            self,
            src: np.ndarray,
            adv: np.ndarray,
            att: np.ndarray,
            summary: bool = False,
            tag: str = '',
            save: str = '',
            y_adv: int = None
    ):
        # Check params & to tensors
        src = self._check(src).cuda()
        adv = self._check(adv).cuda()
        att = self._check(att).cuda()

        # Generate reference images
        src_big = src
        src_inp = self.scale_net(src)
        adv_big = nn.functional.interpolate(adv, size=src_big.shape[2:], mode='bilinear')
        adv_inp = adv

        # Compute labels
        y_src = self.predict(src_inp, scale=False, pooling=None).item()
        if y_adv is None:
            y_adv = self.predict(adv_inp, scale=False, pooling=None).item()

        # Compute adv from att
        att_non_inp = self.scale_net(att)
        att_med_inp = self.scale_net(self.pooling_med(att, 1))
        att_rnd_inp = self.scale_net(self.pooling_rnd(att, 1))

        # Evaluation
        stats = OrderedDict({
            'SRC': self.eval_one(ref=src_big, x=src_big, scale=True),
            'ADV': self.eval_one(ref=src_inp, x=adv_inp, scale=False),
            'BASE': self.eval_one(ref=src_big, x=adv_big, scale=True),
            'ATT': self.eval_one(ref=src_big, x=att, scale=True),
            'ATT-INP-NON': self.eval_one(ref=src_inp, x=att_non_inp, scale=False),
            'ATT-INP-MED': self.eval_one(ref=src_inp, x=att_med_inp, scale=False),
            'ATT-INP-RND': self.eval_one(ref=src_inp, x=att_rnd_inp, scale=False),
        })

        if summary:
            print(self.summary(stats, y_src, y_adv, tag))

        if save:
            # self.save_images(src, tag=f'{tag}.SRC', root=save)
            self.save_images(att, tag=f'{tag}.ATT', root=save)
            F.to_pil_image(adv_inp.cpu()[0]).save(f'{save}/{tag}.ADV.inp.png')
            F.to_pil_image(adv_big.cpu()[0]).save(f'{save}/{tag}.ADV.big.png')

        return stats

    def eval_one(self, ref: torch.Tensor, x: torch.Tensor, scale: bool):
        """Evaluate one image.

        Args:
            ref: which referenced image this input should compare
            x: the input image to be evaluated
            scale: True if the input image is large.

        Returns:
            OrderedDict for evaluate results.
        """
        stats = OrderedDict({
            # predict (normal)
            'Y': self.predict(x, scale=scale, pooling=None).item(),
            # predict (pooling)
            'Y_MED': self.predict(x, scale=True, pooling='med').item() if scale else None,
            'Y_RND': self.predict(x, scale=True, pooling='rnd') if scale else None,
            # distance
            'L-INF': torch.norm(x - ref, p=np.inf),
            'L-2': torch.norm(x - ref, p=2),
            # similarity
            'PSNR': piq.psnr(x, ref, data_range=1),
            'SSIM': piq.ssim(x, ref, data_range=1),
            'MS-SSIM': piq.multi_scale_ssim(x, ref, data_range=1),
            'LPIPS': self.lpips(x * 2 - 1, ref * 2 - 1),
        })
        return stats

    def save_images(self, x: torch.Tensor, tag: str, root: str = '.'):
        """Save all processed images derived from x.

        Args:
            x: the input image to be processed.
            tag: the tag for saved filenames.
            root: root directory to save images.

        Returns:
            None
        """
        os.makedirs(root, exist_ok=True)
        fname = f'{root}/{tag}'

        # big images
        imgs = {
            'plain': x,
            'median': self.pooling_med(x),
            'random': self.pooling_rnd(x),
        }

        # get input and save
        for k, v in imgs.items():
            big = v.cpu()
            inp = self.scale_net(v.cuda()).cpu()
            F.to_pil_image(big[0]).save(f'{fname}.{k}.big.png')
            F.to_pil_image(inp[0]).save(f'{fname}.{k}.inp.png')
