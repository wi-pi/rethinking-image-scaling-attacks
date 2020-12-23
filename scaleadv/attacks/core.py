from typing import Type, Dict, Any, TypeVar

import numpy as np
import torch
import torch.nn as nn
from art.attacks import EvasionAttack
from torch.autograd import Variable
from tqdm import trange

from scaleadv.attacks.utils import PyTorchClassifierFull
from scaleadv.defenses import Pooling
from scaleadv.models import ScalingLayer, FullNet
from scaleadv.scaling import ScalingAPI

ART_ATTACK = TypeVar('ART_ATTACK', bound=EvasionAttack)


def img_to_tanh(x: torch.Tensor) -> torch.Tensor:
    x = (x * 2 - 1) * (1 - 1.e-6)
    x = torch.atanh(x)
    return x


def tanh_to_img(x: torch.Tensor) -> torch.Tensor:
    x = (x.tanh() + 1) * 0.5
    return x


class ScaleAttack(object):
    """This class implements the Scaling Attack with two invariants.

    0. Baseline
       Directly scale up the perturbation.
    1. Hide
       Hide a target image (maybe adversarial) into a large source image.
    2. Generate
       Generate a HR adversarial image.

    Args:
        scaling_api: the API for scaling lib/alg to attack.
        pooling_layer: the defense to bypass.
        class_network: the final classification network.
    """

    def __init__(self, scaling_api: ScalingAPI, pooling_layer: Pooling, class_network: nn.Module):
        # Init network
        self.pooling_layer = pooling_layer
        self.scaling_layer = ScalingLayer.from_api(scaling_api).cuda()
        self.class_network = class_network

        # Init art's proxy
        full_net = FullNet(pooling_layer, self.scaling_layer, class_network)
        kwargs = dict(loss=nn.CrossEntropyLoss(), nb_classes=1000, clip_values=(0, 1))
        self.classifier_big = PyTorchClassifierFull(full_net, input_shape=(3,) + scaling_api.src_shape, **kwargs)
        self.smart_pooling = self.classifier_big.smart_pooling

    @staticmethod
    def _check_inputs(x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError(f'Only support numpy array, but got {type(x)}.')
        if x.ndim != 4 or x.shape[:2] != (1, 3):
            raise ValueError(f'Only support 4d array of shape (1, 3, h, w), but got {x.shape}.')
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError(f'Only support floating array, but got {x.dtype}.')
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def baseline(src: np.ndarray, inp: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        api = ScalingAPI(src.shape[-2:], tgt.shape[-2:], lib='cv', alg='area')
        eps = (tgt - inp)[0].transpose((1, 2, 0)) * 255
        eps = api.backend.scale(eps, src.shape[-2:]).transpose((2, 0, 1)) / 255
        out = np.clip(src + eps, 0, 1)
        return out

    def hide(self, src: np.ndarray, tgt: np.ndarray, src_label: int, tgt_label: int,
             max_iter: int = 100, lr: float = 0.01, weight: float = 2.0,
             nb_sample: int = 20, verbose: bool = True) -> np.ndarray:
        # Check inputs
        src = self._check_inputs(src).cuda()
        tgt = self._check_inputs(tgt).cuda()

        # Prepare attack
        var = Variable(torch.zeros_like(src), requires_grad=True)
        var.data = img_to_tanh(src.data)
        opt = torch.optim.Adam([var], lr=lr)

        with trange(max_iter, desc='Scaling Attack - Hide', disable=not verbose) as pbar:
            for _ in pbar:
                # forward
                att = tanh_to_img(var)
                att_pooling = self.classifier_big.smart_pooling(att, nb_sample)
                inp = self.scaling_layer(att_pooling)

                # loss
                loss_src = torch.square(src - att).mean() * 255 ** 2
                loss_tgt = torch.square(tgt - inp).mean() * 255 ** 2
                loss_total = torch.sum(loss_src + weight * loss_tgt)

                # opt
                opt.zero_grad()
                loss_total.backward()
                opt.step()

                # test
                pred = self.classifier_big.predict(att.detach().cpu().repeat(nb_sample, 1, 1, 1)).argmax(1)
                acc_src = np.mean(pred == src_label)
                acc_tgt = np.mean(pred == tgt_label)
                pbar.set_postfix({
                    'src': f'{loss_src.cpu().item():.3f}',
                    'tgt': f'{loss_tgt.cpu().item():.3f}',
                    'tot': f'{loss_total.cpu().item():.3f}',
                    f'pred-{src_label}': f'{acc_src:.2%}',
                    f'pred-{tgt_label}': f'{acc_tgt:.2%}',
                })
                if acc_tgt > 0.99:
                    break

        att = att.detach().cpu().numpy()
        return att

    def generate(self, src: np.ndarray, src_label: int,
                 attack_cls: Type[ART_ATTACK], attack_kwargs: Dict[str, Any],
                 nb_samples: int = 20) -> np.ndarray:
        # Check inputs
        src = self._check_inputs(src)

        # Prepare Attack
        attack = attack_cls(self.classifier_big, **attack_kwargs)
        prev_nb = self.smart_pooling.nb_samples
        self.smart_pooling.nb_samples = nb_samples

        # Attack
        att = attack.generate(src, np.array([src_label]))
        self.smart_pooling.nb_samples = prev_nb

        return att
