import numpy as np
import torch
import torch.nn as nn


class ScaleNet(nn.Module):

    def __init__(self, cl: np.ndarray, cr: np.ndarray):
        super(ScaleNet, self).__init__()
        self.cl = nn.Parameter(torch.as_tensor(cl.copy(), dtype=torch.float32), requires_grad=False)
        self.cr = nn.Parameter(torch.as_tensor(cr.copy(), dtype=torch.float32), requires_grad=False)

    def forward(self, inp: torch.Tensor):
        return self.cl @ inp @ self.cr
