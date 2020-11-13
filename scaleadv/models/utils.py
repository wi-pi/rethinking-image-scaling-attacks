from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier


class ReducedCrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            targets = targets.repeat(inputs.shape[0])
        return super(ReducedCrossEntropyLoss, self).forward(inputs, targets)


class AverageGradientClassifier(PyTorchClassifier):

    def __init__(self, *args, nb_samples: int, verbose: bool, y_cmp: List[int], **kwargs):
        super(AverageGradientClassifier, self).__init__(*args, **kwargs)
        self.nb_samples = nb_samples
        self.verbose = verbose
        self.y_cmp = y_cmp
        self.cnt = 0

    def loss_gradient_framework(self, x: "torch.Tensor", y: "torch.Tensor", **kwargs) -> "torch.Tensor":
        import torch  # lgtm [py/repeated-import]

        # Check label shape
        y = self.reduce_labels_framework(y)

        # Convert the inputs to Variable
        x = torch.autograd.Variable(x, requires_grad=True)

        # Compute the gradient and return
        self._model._model.n = self.nb_samples
        model_outputs = self._model(x)
        self._model._model.n = 1
        loss = self._loss(model_outputs[-1], y)

        # Logging
        if self.verbose:
            self.cnt += 1
            p = model_outputs[-1].detach().cpu().numpy().argmax(1)
            stats = OrderedDict({'LOSS': f'{loss.cpu().item():.3f}'})
            for y in self.y_cmp:
                stats[f'PRED-{y}'] = f'{np.mean(p == y):.2%}'
            print(self.cnt, ' '.join([f'{k}: {v}' for k, v in stats.items()]))

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()
        grads = x.grad
        assert grads.shape == x.shape  # type: ignore

        return grads  # type: ignore
