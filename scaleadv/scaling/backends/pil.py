"""
This module implements the PIL scaling backend.
"""
import numpy as np
from PIL import Image

from .core import ScalingBackend, ScalingAlg, ShapeType


class ScalingBackendPIL(ScalingBackend):
    algorithms = {
        ScalingAlg.NEAREST: Image.NEAREST,
        ScalingAlg.LINEAR: Image.LINEAR,
        ScalingAlg.CUBIC: Image.CUBIC,
        ScalingAlg.LANCZOS: Image.LANCZOS,
        ScalingAlg.AREA: Image.BOX,
    }

    def _scale(self, x: np.ndarray, shape: ShapeType) -> np.ndarray:
        y = x

        # PIL only supports uint8 for RGB mode.
        if y.ndim == 3:
            y = np.round(y).astype(np.uint8)

        # Scale
        y = Image.fromarray(y).resize(shape, resample=self.alg)
        y = np.array(y)

        # Round back to input dtype
        if np.issubdtype(x.dtype, np.integer):
            y = np.round(y)

        return y.astype(x.dtype)
