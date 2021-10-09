from typing import Optional, Tuple

import numpy as np
import torch
from art.defences.preprocessor.preprocessor import Preprocessor, PreprocessorPyTorch
from torchvision.transforms.functional import to_pil_image, to_tensor

from scaleadv.defenses import MedianPooling, RandomPooling
from scaleadv.scaling import ScalingAPI


class MedianFilteringExact(PreprocessorPyTorch):

    def __init__(self, scaling_api: ScalingAPI):
        super().__init__(apply_fit=False, apply_predict=True)
        self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.median_filter = MedianPooling.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = self.median_filter(x)
        return x, y


class MedianFilteringBPDA(PreprocessorPyTorch):

    def __init__(self, scaling_api: ScalingAPI):
        super().__init__(apply_fit=False, apply_predict=True)
        self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
        self.median_filter = MedianPooling.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = self.median_filter(x)
        return x, y

    def estimate_forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        return x


class EoTRandomFiltering(PreprocessorPyTorch):

    def __init__(self, scaling_api: ScalingAPI, nb_samples: int, nb_flushes: int):
        super().__init__(apply_fit=False, apply_predict=True)
        self.random_filter = RandomPooling.auto(round(scaling_api.ratio) * 2 - 1, scaling_api.mask)
        self.nb_samples = nb_samples
        self.nb_flushes = nb_flushes

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = self.random_filter(x)
        return x, y


class SaveAndLoad(Preprocessor):

    def __init__(self):
        super().__init__(apply_fit=False, apply_predict=True)

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x = to_pil_image(torch.tensor(x)[0])
        x = to_tensor(x).numpy()[None]
        return x, y


class SaveAndLoadPyTorch(PreprocessorPyTorch):

    def __init__(self):
        super().__init__(apply_fit=False, apply_predict=True)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        # x = x[0]
        x = x.mul(255).byte()
        x = x.div(255).float()
        # x = x[None]
        return x, y