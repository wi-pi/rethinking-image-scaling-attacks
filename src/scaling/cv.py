"""
This module implements the OpenCV scaling backend.
"""
import cv2 as cv
import numpy as np

from src.scaling.backend import ScalingBackend, Shape
from src.scaling.enum import ScalingAlg


class ScalingBackendCV(ScalingBackend):
    algorithms = {
        ScalingAlg.NEAREST: cv.INTER_NEAREST,
        ScalingAlg.LINEAR: cv.INTER_LINEAR,
        ScalingAlg.CUBIC: cv.INTER_CUBIC,
        ScalingAlg.LANCZOS: cv.INTER_LANCZOS4,
        ScalingAlg.AREA: cv.INTER_AREA,
    }

    def _scale(self, x: np.ndarray, shape: Shape) -> np.ndarray:
        x = cv.resize(x, shape, interpolation=self.alg)
        return x
