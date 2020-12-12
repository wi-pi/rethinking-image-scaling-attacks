from collections import OrderedDict
from typing import Optional, Union, Dict, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from art.attacks import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from torch.autograd import Variable
from tqdm import trange

from scaleadv.models.layers import Pool2d, LaplacianPool2d, NonePool2d, CheapRandomPool2d, RandomPool2d
from scaleadv.models.scaling import ScaleNet, FullScaleNet
from scaleadv.models.utils import AverageGradientClassifier, ReducedCrossEntropyLoss

EARLY_STOP_ITER = 200
EARLY_STOP_THRESHOLD = 0.999
TANH_TOLERANCE = 1 - 1e-6

# Adaptive attack modes
RANDOM_APPROXIMATION = {
    'sample': RandomPool2d,
    'cheap': CheapRandomPool2d,
    'laplace': LaplacianPool2d,
}

ART_ATTACK = TypeVar('ART_ATTACK', bound=EvasionAttack)


class ScaleAttack(object):
    """This class implements Scaling attack with several variants.
    All variants supports adaptive mode against defenses.

    1. Hide
       Hide a small image (maybe adversarial) into a large image.
    2. Generate
       Generate a HR adversarial image.

    Args:
        scale_net: scaling network of type `ScaleNet`.
        class_net: classification network of type `nn.Module`.
        pooling: the defense to be bypassed of type `Pool2d` (optional).
    """

    def __init__(
            self,
            scale_net: ScaleNet,
            class_net: nn.Module,
            pooling: Optional[Pool2d] = None,
    ):
        self.scale_net = scale_net
        self.class_net = class_net
        self.pooling = NonePool2d() if pooling is None else pooling

    @staticmethod
    def img_to_tanh(x: torch.Tensor) -> torch.Tensor:
        x = (x * 2 - 1) * TANH_TOLERANCE
        x = torch.atanh(x)
        return x

    @staticmethod
    def tanh_to_img(x: torch.Tensor) -> torch.Tensor:
        x = (x.tanh() + 1) * 0.5
        return x

    @staticmethod
    def baseline(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(tgt, dtype=torch.float32).cuda()
        x = nnf.interpolate(x, src.shape[2:], mode='bilinear')
        x = np.array(x.cpu(), dtype=ART_NUMPY_DTYPE)
        return x

    def predict(
            self,
            x: Union[np.ndarray, torch.Tensor],
            scale: bool = False,
            pooling: bool = False,
            n: int = 1
    ) -> np.ndarray:
        """Predict big/small image with pooling support.

        Args:
            x: input image of shape [1, 3, H, W].
            scale: True if input image needs to be scaled.
            pooling: True if you want to apply pooling before scaling.
            n: number of samples for the pooling layer.

        Returns:
            np.ndarray containing predicted labels (multiple for n > 1).
        """
        x = torch.as_tensor(x, dtype=torch.float32).cuda()
        with torch.no_grad():
            if pooling:
                assert scale, 'Cannot apply pooling without scaling.'
                x = self.pooling(x, n)
            if scale:
                x = self.scale_net(x)
            pred = self.class_net(x).argmax(1).cpu()

        return pred.numpy()

    def _check_mode(self, mode: Optional[str], nb_samples: int):
        if mode is not None:
            assert isinstance(self.pooling, RandomPool2d), 'Only support approximation for RandomPool2d.'
            assert mode in RANDOM_APPROXIMATION.keys(), f'Unsupported approximation mode "{mode}".'
            if nb_samples == 1:
                raise UserWarning(f'Random defense only used {nb_samples} samples.')
        if mode is None:
            if nb_samples > 1:
                raise UserWarning(f'Non-random defense used {nb_samples} samples.')

    def _check_input(self, x: np.ndarray) -> torch.Tensor:
        assert isinstance(x, np.ndarray)
        assert x.ndim == 4 and x.shape[0:2] == (1, 3) and x.dtype == ART_NUMPY_DTYPE
        return torch.as_tensor(x, dtype=torch.float32).cuda()

    def _get_attack_pooling(self, mode: Optional[str], src: Optional[torch.Tensor] = None) -> Pool2d:
        pooling = self.pooling
        if mode is not None:
            pooling = RANDOM_APPROXIMATION[mode].from_pooling(pooling)
            if mode == 'laplace':
                assert src is not None, 'Need src image to estimate Laplacian distribution.'
                pooling.fresh_dist(src, lam=1.0)
        return pooling

    def hide(
            self,
            src: np.ndarray,
            tgt: np.ndarray,
            lr: float = 0.01,
            step: int = 1000,
            lam_inp: float = 1,
            mode: Optional[str] = None,
            nb_samples: int = 1,
            attack_self: bool = False,
            tgt_label: int = None,
            test_freq: int = 0,
            early_stop: bool = True,
    ) -> np.ndarray:
        """Hide a small image (maybe adversarial) into a large image.

        Args:
            src: large source image of shape [1, 3, H, W].
            tgt: small target image of shape [1, 3, h, w].
            lr: learning rate.
            step: max iterations.
            lam_inp: weight for input space penalty.
            mode: how to approximate the random pooling, see `RANDOM_APPROXIMATION`.
            nb_samples: how many samples to approximate the random pooling.
            attack_self: True if include non-defense source image in the loop.
            tgt_label: target label for the adv-example (for test only)
            test_freq: run full test per `test_freq` iterations, set 0 to disable it.
            early_stop: stop if loss converges.

        Returns:
            np.ndarray: final large attack image
        """
        # Check params & convert to tensors
        src = self._check_input(src)
        tgt = self._check_input(tgt)
        self._check_mode(mode, nb_samples)

        # Get reference labels
        y_src = self.predict(src, scale=True).item()
        y_tgt = self.predict(tgt, scale=False).item() if tgt_label is None else tgt_label

        # Prepare attack vars
        var = Variable(torch.zeros_like(src), requires_grad=True)
        var.data = self.img_to_tanh(src.data)

        # Prepare pooling layer to be attacked
        pooling = self._get_attack_pooling(mode, src)

        # Prepare attack optimizer
        factor = np.sqrt(1. * src.numel() / tgt.numel())
        optimizer = torch.optim.Adam([var], lr=lr)

        # Start attack
        prev_loss = np.inf
        with trange(step, desc=f'ScaleAdv-Hide ({mode})') as pbar:
            for i in pbar:
                # forward
                att = self.tanh_to_img(var)
                att_def = pooling(att, n=nb_samples)
                if attack_self:
                    att_def = torch.cat([att_def, att])
                inp = self.scale_net(att_def)

                # loss
                loss = OrderedDict({
                    'BIG': (src - att).reshape(att.shape[0], -1).norm(2, dim=1).mean(),
                    'INP': (tgt - inp).reshape(inp.shape[0], -1).norm(2, dim=1).mean() * factor,
                })
                total_loss = loss['BIG'] + lam_inp * loss['INP']

                # optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # logging
                loss['TOTAL'] = total_loss
                stats = OrderedDict({k: f'{v.cpu().item():.3f}' for k, v in loss.items()})
                pred = self.predict(inp, scale=False, pooling=False, n=1)
                if pred.shape[0] == 1:
                    stats['PRED'] = pred.item()
                else:
                    for y in [y_src, y_tgt]:
                        stats[f'PRED-{y}'] = f'{np.mean(pred == y):.2%}'
                pbar.set_postfix(stats)

                # full test
                if test_freq and i % test_freq == 0:
                    pred = self.predict(att, scale=True, pooling=True, n=nb_samples)
                    for y in [y_src, y_tgt]:
                        print(f'Test {y}: {np.mean(pred == y):.2%}')
                    # F.to_pil_image(att[0].cpu().detach()).save(f'ADV-{i:03d}.png')

                # early stop
                if early_stop and i % EARLY_STOP_ITER == 0:
                    if total_loss > prev_loss * EARLY_STOP_THRESHOLD:
                        break
                    prev_loss = total_loss

        # Convert to np
        att = att.detach().cpu().numpy().astype(ART_NUMPY_DTYPE)
        return att

    def generate(
            self,
            x_src: np.ndarray,
            y_src: int,
            attack_cls: Type[ART_ATTACK],
            attack_args: Dict,
            y_tgt: Optional[int] = None,
            mode: Optional[str] = None,
            nb_samples: int = 1,
            nb_classes: int = 1000,
            verbose: bool = True,
    ) -> np.ndarray:
        """Generate a HR adversarial image.

        Args:
            x_src: large source image of shape [1, 3, H, W].
            y_src: label of source image.
            attack_cls: class of adv attacker
            attack_args: kw args of adv attacker
            y_tgt: target label, set to None for un-targeted attack.
            mode: how to approximate the random pooling, see `RANDOM_APPROXIMATION`.
            nb_samples: how many samples to approximate the random pooling.
            nb_classes: total number of classes.
            verbose: output losses

        Returns:
            np.ndarray: final large attack image
        """
        # Check params & convert to tensors
        x_src = self._check_input(x_src)
        self._check_mode(mode, nb_samples)

        # For un-targeted attack, we set y_tgt to y_src
        if y_tgt is None:
            y_tgt = y_src

        # Prepare pooling layer to be attacked
        pooling = self._get_attack_pooling(mode, x_src)

        # Load networks
        full_net = FullScaleNet(self.scale_net, self.class_net, pooling, n=1)
        classifier = AverageGradientClassifier(full_net, ReducedCrossEntropyLoss(), tuple(x_src.shape[1:]), nb_classes,
                                               nb_samples=nb_samples, verbose=verbose, y_cmp=[y_src, y_tgt],
                                               clip_values=(0, 1))

        # Attack
        attack = attack_cls(classifier, **attack_args)
        y_tgt = np.eye(nb_classes, dtype=np.int)[None, y_tgt]
        att = attack.generate(x=x_src.cpu(), y=y_tgt)

        return att
