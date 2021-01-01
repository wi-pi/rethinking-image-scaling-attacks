from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Any

import numpy as np
from loguru import logger

ShapeType = Tuple[int, int]


class ScalingLib(Enum):
    CV = 1
    PIL = 2

    @classmethod
    def names(cls):
        return [x.name.lower() for x in cls]


class ScalingAlg(Enum):
    NEAREST = 1
    LINEAR = 2
    CUBIC = 3
    LANCZOS = 4
    AREA = 5

    @classmethod
    def names(cls):
        return [x.name.lower() for x in cls]


class ScalingBackend(ABC):
    """Backend interface for scaling libraries & algorithms.

    This backend only accepts:
      * shape: gray (w, h) or rgb (w, h, 3)
      * value: [0, 255]
      * type: any

    Note:
      * We explicitly restrict the value range to [0, 255],
        because some backends (e.g., PIL) do not support range [0, 1].
      * Be aware of the acceptable shape is (w, h), not (h, w).
    """
    algorithms: Dict[ScalingAlg, Any] = NotImplemented

    @staticmethod
    def create(lib: ScalingLib, alg: ScalingAlg) -> "ScalingBackend":
        """Create implemented backends.

        We need lazy import to avoid unnecessary importing overheads.
        """
        if lib == ScalingLib.CV:
            from . import cv
            return cv.ScalingBackendCV(alg)
        if lib == ScalingLib.PIL:
            from . import pil
            return pil.ScalingBackendPIL(alg)
        raise NotImplementedError(f'Does not support backend "{lib}".')

    @staticmethod
    def _check_input(x: np.ndarray):
        if x.ndim not in [2, 3]:
            raise ValueError(f'Only support 2/3 dimensions, but got {x.ndim}.')
        if x.ndim == 3 and x.shape[-1] != 3:
            raise ValueError(f'Only support channel last RGB data, but got {x.shape}.')
        if x.max() <= 1:
            logger.warning(f'The input image may not be in the correct range [0, 255], got [{x.min()}, {x.max()}].')

    def __init__(self, alg: ScalingAlg):
        if alg not in self.algorithms.keys():
            raise NotImplementedError(f'{self.__class__.__name__} does not support algorithm "{alg}".')
        self.alg = self.algorithms[alg]

    def scale(self, x: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Scale a given image array to the given shape.

        Args:
            x: input image of shape (w, h) or (w, h, 3), and range [0, 255].
            shape: a tuple indicating target shape (w, h).

        Return:
            scaled image array with the same dtype.
        """
        self._check_input(x)
        x = self._scale(x, shape)
        return x

    @abstractmethod
    def _scale(self, x: np.ndarray, shape: ShapeType) -> np.ndarray:
        raise NotImplementedError
