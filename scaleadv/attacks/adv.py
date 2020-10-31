"""
This module implements several attacks' variants.
1. Attack x, compute perturbation on x.
2. Attack x, compute perturbation on fixed proxy(x).

Note:
    1. For PGD, option 2 is implemented by hacking the estimator's `loss_gradient_framework` function.
       - This might be inaccurate due to projection on proxy(x) instead of x.
       - This might be affected by future updates of PGD's official implementation.

References:
    - ShadowAttack
      https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/shadow_attack.py
"""
from typing import Optional

import numpy as np
from art.attacks.evasion import ProjectedGradientDescentPyTorch, ShadowAttack
from art.config import ART_NUMPY_DTYPE
from tqdm import trange

from scaleadv.attacks.proxy import Proxy


def loss_gradient_framework_average(super_call):
    def wrapper(x, y, **kwargs):
        grad = super_call(x, y, **kwargs)
        return grad.mean(0).repeat(x.shape[0], 1, 1, 1)

    return wrapper


class IndirectPGD(ProjectedGradientDescentPyTorch):

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, proxy: Optional[Proxy] = None) -> np.ndarray:
        """
        Generate adversarial example by attacking a proxy.
        Args:
            x: An array with the original inputs, shape is (1, C, H, W).
            y: Target values.
            proxy: A callable that returns a batch samples from x.

        Returns:
            An array holding the adversarial example.
        """
        if proxy is None:
            return super(IndirectPGD, self).generate(x, y)
        if x.shape[0] != 1:
            raise ValueError(f'IndirectPGD only supports a single input, but got {x.shape[0]} inputs.')

        # Set up proxies
        x_proxy = proxy(x)
        y_proxy = y.repeat(x_proxy.shape[0], axis=0)

        # Hack gradient
        grad_fn = self.estimator.loss_gradient_framework
        self.estimator.loss_gradient_framework = loss_gradient_framework_average(grad_fn)

        # Generate adversarial examples
        x_proxy_adv = super(IndirectPGD, self).generate(x_proxy, y_proxy)
        perturbation = x_proxy_adv[0] - x_proxy[0]
        x_adv = np.clip(x + perturbation, *self.estimator.clip_values).astype(ART_NUMPY_DTYPE)

        # Recover gradient
        self.estimator.loss_gradient_framework = grad_fn

        # Inner test
        pred = self.estimator.predict(x_proxy_adv).argmax(1)
        print(f'adv-100: {np.mean(pred == 100):.2%}')
        print(f'adv-200: {np.mean(pred == 200):.2%}')

        return x_adv


class IndirectShadowAttack(ShadowAttack):

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, proxy: Optional[Proxy] = None) -> np.ndarray:
        """
        Generate adversarial example by attacking a proxy (not necessarily Normal Noise)
        Args:
            x: An array with the original inputs, shape is (1, C, H, W).
            y: Target values.
            proxy: A callable that returns a batch samples from x.

        Returns:
            An array holding the adversarial example.
        """
        if proxy is None:
            return super(IndirectShadowAttack, self).generate(x, y)
        if x.shape[0] != 1:
            raise ValueError(f'Indirect attack only supports a single input, but got {x.shape[0]} inputs.')

        # Set up proxies
        x_proxy = proxy(x)
        y_proxy = y.repeat(x_proxy.shape[0], axis=0)

        # Initialize perturbation
        perturbation = (
                np.random.uniform(
                    low=self.estimator.clip_values[0], high=self.estimator.clip_values[1], size=x.shape
                ).astype(ART_NUMPY_DTYPE)
                - (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2
        )

        for _ in trange(self.nb_steps, desc="Shadow attack"):
            gradients_ce = np.mean(
                self.estimator.loss_gradient(x=x_proxy + perturbation, y=y_proxy, sampling=False)
                * (1 - 2 * int(self.targeted)),
                axis=0,
                keepdims=True,
            )
            gradients = gradients_ce - self._get_regularisation_loss_gradients(perturbation)
            perturbation += self.learning_rate * gradients

        x_p = x + perturbation
        x_adv = np.clip(x_p, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]).astype(
            ART_NUMPY_DTYPE
        )
        return x_adv
