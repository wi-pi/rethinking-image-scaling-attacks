import os
from collections import OrderedDict
from typing import Dict, Any, Optional

import numpy as np
import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from prettytable import PrettyTable

from scaleadv.attacks.core import ScaleAttack
from scaleadv.defenses import POOLING_MAPS
from scaleadv.models import ScalingLayer
from scaleadv.scaling import ScalingAPI


class Evaluator(object):
    nb_samples = 100
    fields = ['Y', 'Y_MED', 'Y_RND', 'Linf', 'L2', 'MSE', 'SSIM']

    def __init__(self, scaling_api: ScalingAPI, class_network: nn.Module, nb_samples: Optional[int] = None):
        # Init network
        self.scaling_layer = ScalingLayer.from_api(scaling_api).cuda()
        self.class_network = class_network
        if nb_samples is not None:
            self.nb_samples = nb_samples

        # Init test pooling
        args = (round(scaling_api.ratio) * 2 - 1, scaling_api.mask)
        self.pooling = {k: POOLING_MAPS[k].auto(*args).cuda() for k in ['none', 'median', 'uniform']}

    @staticmethod
    def _check_inputs(x: np.ndarray) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
        assert x.dtype == np.float32
        assert 0 <= x.min() and x.max() <= 1
        return torch.as_tensor(x, dtype=torch.float32)

    def predict(self, x: torch.Tensor, pooling: str, scaling: bool, n: int = 1) -> np.ndarray:
        with torch.no_grad():
            if scaling:
                x = x.repeat(n, 1, 1, 1)
                x = self.pooling[pooling](x)
                x = self.scaling_layer(x)
            pred = self.class_network(x).argmax(1).cpu()
        return pred.numpy()

    def eval(self, src: np.ndarray, adv: np.ndarray, att: np.ndarray, y_src: int, y_adv: Optional[int] = None,
             tag: str = 'test', show: bool = False, save: str = ''):
        # Check inputs
        src = self._check_inputs(src).cuda()
        adv = self._check_inputs(adv).cuda()
        att = self._check_inputs(att).cuda()

        # Generate reference images
        src_big = src
        src_inp = self.scaling_layer(src)
        adv_big = ScaleAttack.baseline(src_big.cpu().numpy(), src_inp.cpu().numpy(), adv.cpu().numpy())
        adv_big = self._check_inputs(adv_big).cuda()
        adv_inp = adv

        # Evaluate
        stats = OrderedDict({
            'src': self.eval_one(ref=src_big, x=src_big, scaling=True),
            'adv': self.eval_one(ref=src_inp, x=adv_inp, scaling=False),
            'base': self.eval_one(ref=src_big, x=adv_big, scaling=True),
            'att': self.eval_one(ref=src_big, x=att, scaling=True),
        })

        # Output
        if show:
            if y_adv is None:
                y_adv = self.class_network(adv).argmax(1).cpu().item()
            self.show(tag, stats, y_src, y_adv)

        if save:
            os.makedirs(save, exist_ok=True)
            self.save(save, tag, 'src', src_big)
            self.save(save, tag, 'adv', adv_inp, scaling=False)
            self.save(save, tag, 'base', adv_big)
            self.save(save, tag, 'att', att)

        return stats

    def eval_one(self, ref: torch.Tensor, x: torch.Tensor, scaling: bool) -> Dict[str, Any]:
        ref = torch.clamp(ref, 0, 1)
        x = torch.clamp(x, 0, 1)

        stats = OrderedDict({
            # pred (normal)
            'Y': self.predict(x, pooling='none', scaling=scaling).item(),
            # pred (pooling)
            'Y_MED': self.predict(x, pooling='median', scaling=True).item() if scaling else None,
            'Y_RND': self.predict(x, pooling='uniform', scaling=True, n=self.nb_samples) if scaling else None,
            # distance
            'Linf': torch.norm(x - ref, p=np.inf).cpu().item(),
            'L2': torch.norm(x - ref, p=2).cpu().item(),
            # similarity
            'MSE': torch.square(x - ref).mean().cpu().item() * 255 ** 2,
            'SSIM': piq.ssim(x.cpu(), ref.cpu(), data_range=1).cpu().item(),
        })
        return stats

    def show(self, tag: str, stats: Dict[str, Any], y_src: int, y_adv: int):
        tab = PrettyTable([tag] + self.fields)
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
            for field in self.fields[3:]:
                if field in data:
                    row.append(f'{data[field]:.3f}')
                else:
                    row.append('-')
            tab.add_row(row)
        print(tab)

    def save(self, save: str, tag: str, name: str, x: torch.Tensor, scaling: bool = True):
        save = save.rstrip('/')

        # save origin
        if not scaling:
            F.to_pil_image(x.cpu()[0]).save(f'{save}/{tag}.{name}.png')
            return

        # save pooling
        for pk, pf in self.pooling.items():
            y = pf(x)
            F.to_pil_image(y.cpu()[0]).save(f'{save}/{tag}.{name}.{pk}.big.png')
            y = self.scaling_layer(y)
            F.to_pil_image(y.cpu()[0]).save(f'{save}/{tag}.{name}.{pk}.inp.png')
