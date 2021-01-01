import torch.nn as nn

from scaleadv.defenses.prevention import Pooling
from .layers import ScalingLayer


class FullNet(nn.Module):

    def __init__(self, pooling: Pooling, scaling: ScalingLayer, backbone: nn.Module):
        super(FullNet, self).__init__()
        self.pooling = pooling
        self.scaling = scaling
        self.backbone = backbone

    def forward(self, x):
        x = self.pooling(x)
        x = self.scaling(x)
        y = self.backbone(x)
        return y
