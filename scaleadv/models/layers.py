import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):

    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)[None, :, None, None]
        std = torch.as_tensor(std, dtype=torch.float32)[None, :, None, None]
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Expect a batch tensor of size (B, C, H, W). Got {x.size()}.')
        x = (x - self.mean) / self.std
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'
