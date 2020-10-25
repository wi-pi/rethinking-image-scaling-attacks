"""
This module implements Scaling attack with variants.
1. Common attack, with cross-entropy support.
2. Adaptive attack, against both deterministic and non-deterministic defenses.
"""
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from art.config import ART_NUMPY_DTYPE
from tqdm import trange

from scaleadv.datasets.imagenet import create_dataset
from scaleadv.models.layers import Pool2d
from scaleadv.models.scaling import ScaleNet
from scaleadv.tests.utils import resize_to_224x

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999


class ScaleAttack(object):

    def __init__(
            self,
            scale_net: ScaleNet,
            pooling: Optional[Pool2d] = None,
            class_net: Optional[nn.Module] = None,
            max_iter: int = 1000,
            lr: float = 0.01,
            nb_samples: int = 1,
            early_stop: bool = True,
            tol=1e-6):
        """
        Create a `ScaleAttack` instance.

        Args:
            scale_net: scaling network
            pooling: pooling layer as a defense
            class_net: classification network
            max_iter: maximum number of iterations.
            lr: learning rate
            nb_samples: number of samples if pooling is not deterministic
            early_stop: stop optimization if loss has converged
            tol: tolerance when converting to tanh space
        """
        if pooling is not None and nb_samples < 1:
            raise ValueError(f'Expect at least one sample of the pooling layer, but got {nb_samples}.')

        self.scale_net = scale_net
        self.pooling = pooling
        self.class_net = class_net
        self.max_iter = max_iter
        self.lr = lr
        self.nb_samples = nb_samples
        self.early_stop = early_stop
        self.tol = tol

    def generate(self,
                 src: np.ndarray,
                 tgt: np.ndarray,
                 use_pooling: bool = False,
                 use_ce: bool = False,
                 y_tgt: int = None):
        # Check params
        for x in [src, tgt]:
            assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
            assert x.dtype == np.float32

        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()

        # Prepare attack
        var = torch.autograd.Variable(src.clone().detach(), requires_grad=True)
        var.data = torch.atanh((var.data * 2 - 1) * (1 - self.tol))
        optimizer = torch.optim.Adam([var], lr=self.lr)
        if use_ce:
            y_tgt = torch.LongTensor([y_tgt]).repeat((self.nb_samples,)).cuda()

        # Start attack
        with trange(self.max_iter, desc='ScaleAttack') as pbar:
            prev_loss = np.inf
            for i in pbar:
                # Get attack image (big)
                att = (var.tanh() + 1) * 0.5
                # Get defensed image (big)
                att_def = att
                if use_pooling:
                    att_def = att_def.cpu().repeat(self.nb_samples, 1, 1, 1)
                    att_def = self.pooling(att_def).cuda()
                # Get scaled image (small)
                inp = self.scale_net(att)
                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                if use_ce:
                    pred = self.class_net(inp)
                    loss['CLS'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                total_loss = sum(loss.items(), start=torch.zeros(1))
                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                pbar.set_postfix(stats)
                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        break
                    prev_loss = total_loss

        # Convert to numpy
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        inp = np.array(inp.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att, inp


if __name__ == '__main__':
    import torchvision.transforms as T
    from scaling.ScalingGenerator import ScalingGenerator
    from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
    from scaling.SuppScalingLibraries import SuppScalingLibraries

    # load data
    dataset = create_dataset(transform=None)
    src, _ = dataset[5000]
    tgt, _ = dataset[1000]
    src = resize_to_224x(src)
    src, tgt = map(np.array, [src, tgt])

    # load scaling
    lib = SuppScalingLibraries.CV
    algo = SuppScalingAlgorithms.LINEAR
    scaling = ScalingGenerator.create_scaling_approach(src.shape, (224, 224, 4), lib, algo)
    tgt = scaling.scale_image(tgt)

    # load attack
    src = np.array(src / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    tgt = np.array(tgt / 255, dtype=np.float32).transpose((2, 0, 1))[None, ...]
    scaling_net = ScaleNet(scaling.cl_matrix, scaling.cr_matrix).eval().cuda()
    attack = ScaleAttack(scaling_net)
    att, inp = attack.generate(src, tgt)

    # save figs
    f = T.Compose([lambda x: x[0], torch.tensor, T.ToPILImage()])
    for n in ['src', 'tgt', 'att', 'inp']:
        var = locals()[n]
        f(var).save(f'TEST.ScaleAttack.{n}.png')
