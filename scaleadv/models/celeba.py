from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet34


class TakeOne(nn.Module):

    def __init__(self, index: int, thr: float = 0.5):
        super().__init__()
        self.index = index
        self.thr = thr

    def forward(self, x: torch.Tensor):
        is_pos = (torch.sigmoid(x[:, self.index]) > self.thr).float()
        is_neg = 1 - is_pos
        return torch.stack([is_neg, is_pos], dim=-1)


def celeba_resnet34(num_classes: int, binary_label: Optional[int] = None, ckpt: Optional[str] = None):
    model = resnet34(num_classes=num_classes)
    if ckpt is not None:
        ckpt = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(ckpt)

    if binary_label is not None:
        take_one = TakeOne(binary_label)
        model = nn.Sequential(model, take_one)

    return model
