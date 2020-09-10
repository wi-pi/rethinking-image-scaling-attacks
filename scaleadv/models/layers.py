import torch
import torch.nn as nn
import torchvision.transforms as T

class NormalizationLayer(nn.Module):

    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        if x.ndimension() != 4:
            raise ValueError(f'Expect a batch tensor oof siez (B, C, H, W). Got {x.size()}.')

        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return x

    def __repr__(self):
        return f'NormalizationLayer(mean={self.mean}, std={self.std})'

