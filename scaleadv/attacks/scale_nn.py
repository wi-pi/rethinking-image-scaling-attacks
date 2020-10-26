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

from scaleadv.models.layers import Pool2d, RandomPool2d
from scaleadv.models.scaling import ScaleNet

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
            lam_inp: float = 1.0,
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
            lam_inp: lambda for loss_inp
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
        self.lam_inp = lam_inp
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
                    if isinstance(self.pooling, RandomPool2d):
                        att_def = att_def.cpu()
                    att_def = att_def.repeat(self.nb_samples, 1, 1, 1)
                    att_def = self.pooling(att_def).cuda()

                # Get scaled image (small)
                inp = self.scale_net(att_def)

                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean()
                if use_ce:
                    pred = self.class_net(inp)
                    loss['CLS'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                total_loss = loss['BIG'] + self.lam_inp * loss['INP'] + loss.get('CLS', 0)

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                if use_ce:
                    with torch.no_grad():
                        if pred.shape[0] == 1:
                            stats['PRED'] = pred.argmax(1)[0].cpu().item()
                        else:
                            acc = (pred.argmax(1) == 100).float().mean().cpu().item()
                            stats['PRED-100'] = f'{acc:.2%}'
                            acc = (pred.argmax(1) == 200).float().mean().cpu().item()
                            stats['PRED-200'] = f'{acc:.2%}'
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
