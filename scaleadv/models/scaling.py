import numpy as np
import torch
import torch.nn as nn

from scaleadv.models.layers import Pool2d


class ScaleNet(nn.Module):

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScaleNet, self).__init__()
        self.cl = nn.Parameter(torch.as_tensor(cl.copy(), dtype=torch.float32), requires_grad=False)
        self.cr = nn.Parameter(torch.as_tensor(cr.copy(), dtype=torch.float32), requires_grad=False)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr


class FullScaleNet(nn.Module):

    def __init__(self, scale_net: ScaleNet, class_net: nn.Module, pooling: Pool2d, n: int):
        super(FullScaleNet, self).__init__()
        self.scale_net = scale_net
        self.class_net = class_net
        self.pooling = pooling
        self.n = n

    def forward(self, x):
        if self.n > 0:
            x = x.to(self.pooling.dev).repeat(self.n, 1, 1, 1)
            x = self.pooling(x).cuda()
        x = self.scale_net(x)
        y = self.class_net(x)
        return y
