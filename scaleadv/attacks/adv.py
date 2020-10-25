"""
This module implements the Projected Gradient Descent attack's invariants.
1. Attack x, compute perturbation on x.
2. Attack x, compute perturbation on fixed augment(x).

Note:
    Option 2 is implemented by hacking the estimator's `loss_gradient_framework` function.
    This might be inaccurate due to projection on proxy(x) instead of x.
"""
from typing import Optional, Callable

import numpy as np
from art.attacks.evasion import ProjectedGradientDescentPyTorch, ShadowAttack


def loss_gradient_framework_average(super_call):
    def wrapper(x, y, **kwargs):
        grad = super_call(x, y, **kwargs)
        return grad.mean(0).repeat(x.shape[0], 1, 1, 1)

    return wrapper


class IndirectPGD(ProjectedGradientDescentPyTorch):

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, proxy: Optional[Callable] = None) -> np.ndarray:
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
        x_adv = np.clip(x + perturbation, *self.estimator.clip_values)

        # Recover gradient
        self.estimator.loss_gradient_framework = grad_fn

        return x_adv
