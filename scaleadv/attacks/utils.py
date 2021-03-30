"""
This module implements the utilities of adaptive scaling attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from art.estimators.classification import PyTorchClassifier
from loguru import logger
from torch.autograd import Variable

import numpy as np
from tqdm import trange

from scaleadv.defenses.prevention import Pooling
from scaleadv.defenses.prevention import RandomPooling


def img_to_tanh(x: torch.Tensor) -> torch.Tensor:
    x = (x * 2 - 1) * (1 - 1.e-6)
    x = torch.atanh(x)
    return x


def tanh_to_img(x: torch.Tensor) -> torch.Tensor:
    x = (x.tanh() + 1) * 0.5
    return x


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
            if self.nb_samples == 1:
                logger.warning(f'The pooling layer {type(pooling_layer).__name__} only got 1 sample.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_random:
            self.nb_iter += 1
            return self.pooling_layer(x)

        # expectation part
        xp = self.pooling_layer.apply_padding(x)
        xp = nnf.conv2d(xp.transpose(0, 1), self.prob_kernel).transpose(0, 1)
        x = xp * self.pooling_layer.mask + x * (1 - self.pooling_layer.mask)  # NOTE: this mask is important!

        # noise part
        if self.nb_iter % self.nb_flushes == 0:
            x_rep = x.repeat(self.nb_samples, 1, 1, 1)
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

    def __init__(self, *args, nb_samples: int, nb_flushes: int, verbose: bool = True, **kwargs):
        super(PyTorchClassifierFull, self).__init__(*args, **kwargs)
        self.smart_pooling = SmartPooling(self.model.pooling, nb_samples, nb_flushes).cuda()
        self.verbose = verbose

    def loss_gradient(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
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

    # For different versions of art.
    loss_gradient_framework = loss_gradient


class SignOPT_ModelAdaptor(object):

    def __init__(self, model: PyTorchClassifier):
        self.model = model

    def predict_label(self, x: torch.Tensor):
        x = x.clamp(0, 1)
        # x = torch.round(x * 255.0) / 255.0
        p = self.model.predict(x.cpu().numpy()).argmax(1)
        return torch.tensor(p).cuda()


def improve_theta(preprocess: nn.Module, x0, theta, gg):
    # get HR boundary point
    hr_bnd = x0 + torch.tensor(theta * gg, dtype=torch.float).cuda()

    # get LR pivot point
    lr_pivot = preprocess(hr_bnd)

    # try to find the best HR that projects to the same LR pivot
    var = Variable(torch.zeros_like(x0), requires_grad=True)
    var.data = img_to_tanh(x0.data)
    opt = torch.optim.Adam([var], lr=0.01)
    for i in range(200):
        x1 = tanh_to_img(var)
        x1_lr = preprocess(x1)
        loss_src = torch.square(x0 - x1).mean() * 255 ** 2
        loss_tgt = torch.square(lr_pivot - x1_lr).mean() * 255 ** 2
        loss_total = loss_src + 2.0 * loss_tgt
        opt.zero_grad()
        loss_total.backward()
        opt.step()

    # now we have a better HR point
    # print((x0-x1).norm(2), (lr_pivot - x1_lr).norm(2))

    # get the better theta
    theta_new = x1 - x0
    gg_new = torch.norm(theta_new)
    theta_new = theta_new / gg_new

    return theta_new.detach().cpu().numpy(), gg_new.detach().cpu().numpy().item()


def inverse_preprocess(preprocess: nn.Module, x0: torch.Tensor, delta: torch.Tensor, w1=1.0,w2=10.0,T=2000,tau=100) -> torch.Tensor:
    """
    Map LR-space perturbation delta to the HR-space.
    Find delta_hr, such that preprocess(x0 + delta_hr) ≈ preprocess(x0) + delta

    Args:
        preprocess: the function maps inputs from the HR-space to the LR-space.
        x0: the HR clean image.
        delta: the LR perturbation.

    Returns:
        Delta: the HR perturbation, which maps to the LR perturbation delta.

    Solving:
        Delta* = arg min MSE{pre(x0+Delta) - (pre(x0) + x1_noise)} + c*MSE(Delta)
    """
    assert isinstance(preprocess, nn.Module)
    assert x0.ndim == 4 and x0.shape[0] == 1
    assert delta.ndim == 4 and delta.shape[0] == 1

    # get LR-space mount point
    x1 = (preprocess(x0) + delta).detach()

    # solve for var=x0+Delta*
    var_hr = Variable(x0.detach().clone(), requires_grad=True)
    opt = torch.optim.Adam([var_hr], lr=0.01)
    with trange(T, disable=False) as pbar:
        prev = np.inf
        for i in pbar:
            var_lr = preprocess(var_hr)
            loss_hr = torch.square(x0 - var_hr).mean() * 255 ** 2
            loss_lr = torch.square(x1 - var_lr).mean() * 255 ** 2
            loss_total = w1 * loss_hr + w2 * loss_lr
            opt.zero_grad()
            loss_total.backward()
            opt.step()

            pbar.set_postfix({
                'hr': f'{loss_hr.cpu().item():.3f}',
                'lr': f'{loss_lr.cpu().item():.3f}',
                'tot': f'{loss_total.cpu().item():.3f}',
            })

            if i % tau == 0:
                if loss_total > prev * 0.99999:
                    break
                prev = loss_total

    # get HR-space direction
    delta_hr = (var_hr - x0).detach()
    return delta_hr


def inverse_preprocess_np(preprocess: nn.Module, x0: np.ndarray, delta: np.ndarray, w1=1.0,w2=10.0,T=2000,tau=100) -> np.ndarray:
    if preprocess is None:
        return delta
    x0 = torch.as_tensor(x0, dtype=torch.float).cuda()
    delta = torch.as_tensor(delta, dtype=torch.float).cuda()
    res = inverse_preprocess(preprocess, x0, delta,w1,w2,T,tau)
    return res.cpu().numpy()
