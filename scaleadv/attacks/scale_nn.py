from collections import OrderedDict
from typing import Optional

import numpy as np
import piq
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from art.config import ART_NUMPY_DTYPE
from torch.autograd import Variable
from tqdm import trange

from scaleadv.models.layers import Pool2d, LaplacianPool2d
from scaleadv.models.scaling import ScaleNet

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999
TANH_TOLERANCE = 1 - 1e-6


class ScaleAttack(object):
    """This class implements Scaling attack with several variants.
    1. Common Attack
       Hide an arbitrary small image (possibly adversarial) into a large image.
    2. Adaptive Attack
       Like 1, but is robust to deterministic and non-deterministic defenses.
    3. Optimal Attack
       Generate a high-resolution adversarial image.

    Args:
        scale_net: scaling network of type `ScaleNet`.
        class_net: classification network of type `nn.Module`.
        pooling: pooling layer (defense) of type `Pool2d` (optional).

    Keyword Args:
        lr: step size for scaling attack.
        max_iter: maximum number of iterations for scaling attack.
        lam_inp: extra multiplier for L2 penalty of input space loss.
        nb_samples: number of samples to approximate EE(pooling).
        early_stop: stop optimization if loss has converged.
    """

    def __init__(
            self,
            scale_net: ScaleNet,
            class_net: nn.Module,
            pooling: Optional[Pool2d] = None,
            lr: float = 0.01,
            max_iter: int = 1000,
            lam_inp: float = 1.0,
            nb_samples: int = 1,
            early_stop: bool = True,
    ):
        if nb_samples < 1:
            raise ValueError(f'Expect at least one sample, but got {nb_samples}.')

        self.scale_net = scale_net
        self.class_net = class_net
        self.pooling = pooling
        self.lr = lr
        self.max_iter = max_iter
        self.lam_inp = lam_inp
        self.nb_samples = nb_samples
        self.early_stop = early_stop

    @staticmethod
    def img_to_tanh(x: torch.Tensor) -> torch.Tensor:
        x = (x * 2 - 1) * TANH_TOLERANCE
        x = torch.atanh(x)
        return x

    @staticmethod
    def tanh_to_img(x: torch.Tensor) -> torch.Tensor:
        x = (x.tanh() + 1) * 0.5
        return x

    def predict(self, x: torch.Tensor, scale: bool = False, pooling: bool = False, n: int = 1) -> np.ndarray:
        """Predict big/small image with pooling support.
        Args:
            x: input image of shape [1, 3, H, W].
            scale: True if input image needs to be scaled.
            pooling: True if you want to apply pooling before scaling.
            n: number of samples for the pooling layer.

        Returns:
            np.ndarray containing predicted labels (multiple for n > 1).
        """
        x = torch.tensor(x).cuda()
        with torch.no_grad():
            if pooling:
                assert scale, 'Cannot apply pooling without scaling.'
                x = self.pooling(x, n)
            if scale:
                x = self.scale_net(x)
            pred = self.class_net(x).argmax(1).cpu()

        return pred.numpy()

    def generate(
            self,
            src: np.ndarray,
            tgt: np.ndarray,
            adaptive: bool = False,
            mode: str = 'sample',
            test_freq: int = 0,
            include_self: bool = False,
    ) -> np.ndarray:
        """Run scale-attack with given source and target images.

        Args:
            src: large source image, of shape [1, 3, H, W].
            tgt: small target image, of shape [1, 3, h, w].
            adaptive: True if run adaptive-attack against predefined pooling layer.
            mode: how to approximate the random pooling, only 'sample' and 'worst' supported now.
            test_freq: full test per `test` iterations, set 0 to disable it.
            include_self: True if you want the attack image is adversarial without pooling.

        Returns:
            np.ndarray: final large attack image

        Notes:
            1. 'worst' returns the worst result by up-sampling with linear interpolation.
               this solves both median and random defenses with "2\beta" kernel width.
        """
        # Check params
        for x in [src, tgt]:
            assert x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 3
            assert x.dtype == np.float32
        if adaptive is True:
            assert mode in ['sample', 'average', 'worst'], f'Unsupported adaptive mode "{mode}".'

        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        tgt = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        factor = np.sqrt(1. * src.numel() / tgt.numel())

        # Return worst case result
        if adaptive and mode == 'worst':
            x = nn.functional.interpolate(tgt, src.shape[2:], mode='bilinear')
            x = np.array(x.cpu(), dtype=ART_NUMPY_DTYPE)
            return x

        # Get predicted labels
        y_src = self.predict(src, scale=True).item()
        y_tgt = self.predict(tgt, scale=False).item()

        # Prepare attack vars
        var = Variable(torch.zeros_like(src), requires_grad=True)
        var.data = self.img_to_tanh(src.data)

        # Prepare attack optimizer
        optimizer = torch.optim.Adam([var], lr=self.lr)
        lam_inp = self.lam_inp
        best_att, best_att_l2 = None, np.inf

        # Start attack
        prev_loss = np.inf
        desc = 'ScaleAttack' + (' (adaptive)' if adaptive else '')
        with trange(self.max_iter, desc=desc) as pbar:
            for i in pbar:
                # Get attack image (big)
                att = self.tanh_to_img(var)

                # Get defensed image (big)
                att_def = att
                if adaptive:
                    if mode == 'sample':
                        att_def = self.pooling(att_def, self.nb_samples)
                        if include_self:
                            att_def = torch.cat([att_def, att])  # include unpooling as well
                    else:
                        raise NotImplementedError

                # Get scaled image (small)
                inp = self.scale_net(att_def)

                # Compute loss
                loss = OrderedDict()
                loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['INP'] = (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean() * factor
                total_loss = loss['BIG'] + lam_inp * loss['INP']

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                with torch.no_grad():
                    pred = self.predict(inp)
                    if pred.shape[0] == 1:
                        stats['PRED'] = pred.item()
                    else:
                        for y in [y_src, y_tgt]:
                            stats[f'PRED-{y}'] = f'{np.mean(pred == y):.2%}'
                pbar.set_postfix(stats)

                # Update direction according to predictions
                if np.mean(pred == y_tgt) > 0.95:
                    lam_inp = 1
                    if loss['BIG'] < best_att_l2:
                        best_att, best_att_l2 = att.detach().clone(), loss['BIG']
                else:
                    lam_inp = self.lam_inp

                # Test
                if test_freq and i % test_freq == 0:
                    pred = self.predict(att, scale=True, pooling=True, n=self.nb_samples)
                    for y in [y_src, y_tgt]:
                        print(f'Test {y}: {np.mean(pred == y):.2%}')
                    F.to_pil_image(att[0].cpu().detach()).save(f'ADV-{i:03d}.png')

                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        if stats['PRED'] == y_tgt:
                            break
                    prev_loss = total_loss

        # Convert to numpy
        att = best_att
        att = np.array(att.detach().cpu(), dtype=ART_NUMPY_DTYPE)
        return att

    def generate_optimal(
            self,
            src: np.ndarray,
            target: int,
            lam_ce: int = 2,
    ) -> np.ndarray:
        """Run scale-attack on the entire pipeline with (optional) given pooling result.

        Args:
            src: large source image, of shape [1, 3, H, W].
            target: targeted class number.
            lam_ce: weight for CE loss.

        Returns:
            np.ndarray: generated large attack image.

        Todo:
            1. Add other regularization like Shadow Attack.
            2. Save best-adv like `generate`.
        """
        # Check params
        assert src.ndim == 4 and src.shape[0] == 1 and src.shape[1] == 3
        assert src.dtype == np.float32
        assert self.pooling is not None, 'Optimal attack does not support none pooling.'

        # Convert to tensor
        src = torch.as_tensor(src, dtype=torch.float32).cuda()
        y_src = self.predict(src, scale=True).item()

        # Prepare attack vars
        var = Variable(torch.zeros_like(src), requires_grad=True)
        var.data = self.img_to_tanh(src.data)

        # Prepare attack optimizer
        optimizer = torch.optim.Adam([var], lr=self.lr)

        # Start attack
        prev_loss = np.inf
        desc = 'ScaleAttack (optimal)'
        with trange(self.max_iter, desc=desc) as pbar:
            for i in pbar:
                # Get attack image (big)
                att = self.tanh_to_img(var)

                # Get defensed image (big)
                if i % 50 == 0:
                    if isinstance(self.pooling, LaplacianPool2d):
                        self.pooling.fresh_dist(att)
                att_def = self.pooling(att, n=self.nb_samples)

                # Get scaled image (small)
                inp = self.scale_net(att_def)

                # Get prediction logits
                pred = self.class_net(inp)
                y_tgt = torch.LongTensor([target]).repeat(pred.shape[0]).cuda()

                # Compute loss
                loss = OrderedDict()
                # loss['BIG'] = (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean()
                loss['PSNR'] = piq.psnr(src, att)
                loss['CE'] = nn.functional.cross_entropy(pred, y_tgt, reduction='mean')
                total_loss = 80 - loss['PSNR'] + lam_ce * loss['CE']

                # Optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                pred = pred.argmax(1).cpu().numpy()
                if pred.shape[0] == 1:
                    stats['PRED'] = pred.item()
                else:
                    for y in [y_src, target]:
                        stats[f'PRED-{y}'] = f'{np.mean(pred == y):.2%}'
                pbar.set_postfix(stats)

                # Early stop
                if self.early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        if stats['PRED'] == y_tgt:
                            break
                    prev_loss = total_loss

        # Convert to numpy
        att = att.detach().cpu().numpy().astype(ART_NUMPY_DTYPE)
        return att
