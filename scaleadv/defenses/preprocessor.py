from typing import Optional

import numpy as np
import torch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from scaleadv.defenses import MedianPooling
from scaleadv.scaling import ScalingAPI


class MedianFilteringPyTorch(PreprocessorPyTorch):

    def __init__(self, scaling_api: ScalingAPI):
        super().__init__(apply_fit=False, apply_predict=True)
        self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.median_filter = MedianPooling.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = self.median_filter(x)
        return x, y

