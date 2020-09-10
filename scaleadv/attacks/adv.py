import torch
from art.attacks.attack import EvasionAttack

class AdvAttack(object):

    def __init__(self, attack: EvasionAttack):
        if not isinstance(attack, EvasionAttack):
            raise TypeError(f'Expect an attack method of type EvasionAttack. Got {type(attack)}.')
        self.attack = attack

    def generate(self, x: torch.Tensor):
        self._validate(x)
        return self.attack.generate(x)

    def _validate(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError(f'Expect a torch tensor. Got {type(x)}.')

        if x.ndimension() != 4:
            raise ValueError(f'Tensor should be (B, C, H, W). Got {x.size()}.')

        shape = self.attack.estimator.input_shape
        if x.shape[1:] != shape:
            raise ValueError(f'Single input shape should be {shape}. Got {x.shape[1:]}.')

        min_, max_ = self.attack.estimator.clip_values
        if x.min() < min_ or max_ < x.max():
            raise ValueError(f'Input value is not within the range of ({min_}, {max_}). Got ({x.min().item()}, {x.max().item()}).')

