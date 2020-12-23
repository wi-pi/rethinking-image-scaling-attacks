"""
This module implements the utilities of adaptive scaling attacks.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torch.autograd import Variable

from scaleadv.defenses.prevention import Pooling
from scaleadv.defenses.prevention import RandomPooling


class SmartPooling(nn.Module):
    """A proxy for pooling layers that can approximate the gradient efficiently.

    Note:
        The counter `nb_iter` is not thead-safe and is not appropriate for parallelism.
    """

    def __init__(self, pooling_layer: Pooling, nb_samples: int = 20, nb_flushes: int = 30):
        super(SmartPooling, self).__init__()
        self.pooling_layer = pooling_layer
        self.nb_samples = nb_samples
        self.nb_flushes = nb_flushes

        self.nb_iter = 0
        self.is_random = False
        if isinstance(pooling_layer, RandomPooling):
            self.cache = None
            self.is_random = True
            self.register_buffer('prob_kernel', pooling_layer.prob_kernel[None, None, ...])

    def forward(self, x: torch.Tensor, nb_samples: Optional[int] = None) -> torch.Tensor:
        if not self.is_random:
            self.nb_iter += 1
            return self.pooling_layer(x)

        # expectation part
        x = self.pooling_layer.apply_padding(x)
        x = nnf.conv2d(x.transpose(0, 1), self.prob_kernel).transpose(0, 1)

        # noise part
        if self.nb_iter % self.nb_flushes == 0:
            x_rep = x.repeat(nb_samples or self.nb_samples, 1, 1, 1)
            self.cache = self.pooling_layer(x_rep).data - x_rep.data

        x = torch.clamp(x + self.cache, 0, 1)

        self.nb_iter += 1
        return x


class PyTorchClassifierFull(PyTorchClassifier):
    """A wrapper for the full network with smart pooling.

    This class has the same outcome as PyTorchClassifier, except that:
      1. provides a smart_pooling proxy.
      2. computes gradients with the smart pooling.
    """

    def __init__(self, *args, verbose: bool = True, **kwargs):
        super(PyTorchClassifierFull, self).__init__(*args, **kwargs)
        self.smart_pooling = SmartPooling(self.model.pooling).cuda()
        self.verbose = verbose

    def loss_gradient_framework(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        x_var = Variable(x, requires_grad=True)
        y_cat = torch.argmax(y)

        x_att = self.smart_pooling(x_var)  # we only use the smart version to compute gradients.
        x_att = self.model.scaling(x_att)
        logits = self.model.backbone(x_att)

        loss = self._loss(logits, y_cat.repeat(logits.shape[0]))
        if self.verbose:
            logger.info(f'[{self.smart_pooling.nb_iter}] loss = {loss.cpu().item():.5f}')

        self._model.zero_grad()
        loss.backward()
        return x_var.grad
