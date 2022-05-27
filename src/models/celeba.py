import torch
import torch.nn as nn
from loguru import logger
from torchvision.models import resnet34

CELEBA_MODEL_PATH = {
    'nature': 'static/models/celeba-res34.pth',
}

class SingleToBinary(nn.Module):
    """A simple (non-differentiable) module that converts one neuron to binary sigmoid outputs.
    """

    def __init__(self, index: int, thr: float = 0.5):
        super().__init__()
        self.index = index
        self.thr = thr
        self.register_backward_hook(self._backward)

    def forward(self, x: torch.Tensor):
        is_pos = (torch.sigmoid(x[:, self.index]) > self.thr).float()
        is_neg = 1 - is_pos
        return torch.stack([is_neg, is_pos], dim=-1)

    def _backward(self, module, grad_input, grad_output):
        raise Warning(f'{self.__class__.__class__} is not differentiable.')


def celeba_resnet34(num_classes: int, binary_label: int | None = None, ckpt: str | None = None):
    model = resnet34(num_classes=num_classes)

    if ckpt is not None:
        ckpt = CELEBA_MODEL_PATH['nature']
        logger.info(f'Loading weight file from "{ckpt}".')
        ckpt = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(ckpt)

    if binary_label is not None:
        take_one = SingleToBinary(binary_label)
        model = nn.Sequential(model, take_one)

    return model
