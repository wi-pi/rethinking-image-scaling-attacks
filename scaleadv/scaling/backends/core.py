"""
This module implements core components of unified scaling backend.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple, Any, Type

import numpy as np
from loguru import logger

"""Define type of image shape
"""
ShapeType = Tuple[int, int]


class LowercaseNameMixin(object):
    """Mixin to provide a lowercase name list for Enum (for implementation's convenience).
    """

    @classmethod
    def names(cls: Type[Enum]):
        return [x.name.lower() for x in cls if isinstance(x, Enum)]


class ScalingLib(LowercaseNameMixin, Enum):
    """Enumeration of supported scaling libraries.
    """
    CV = 1
    PIL = 2


class ScalingAlg(LowercaseNameMixin, Enum):
    """Enumeration of supported scaling algorithms.
    """
    NEAREST = 1
    LINEAR = 2
    CUBIC = 3
    LANCZOS = 4
    AREA = 5


class ScalingBackend(ABC):
    """A unified backend for scaling libraries & algorithms.

    The backend only accepts:
      * shape: GRAY (w, h) or RGB (w, h, 3)
      * value: [0, 255]
      * type: any

    Note:
      * Value range is restricted to [0, 255] because some backends (PIL) do not support normalized range [0, 1].
      * Acceptable shape is (w, h), not (h, w).
    """

    # Mapping from our unified algorithm enum to lib-specific algorithm enum,
    # should be implemented by subclasses.
    algorithms: Dict[ScalingAlg, Any] = {}

    def __init__(self, alg: ScalingAlg):
        if alg not in self.algorithms.keys():
            raise NotImplementedError(f'{self.__class__.__name__} does not support algorithm "{alg}".')
        self.alg = self.algorithms[alg]

    def scale(self, x: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Scale an image to the specified shape.

        Args:
            x: input image of shape (w, h) or (w, h, 3), and range [0, 255].
            shape: a tuple indicating target shape (w, h).

        Return:
            Scaled image with the same dtype.
        """
        logger.debug(f'Scaling image ({x.dtype}) from {x.shape} to {shape}.')
        self._validate_image(x)
        return self._scale(x, shape)

    @staticmethod
    def create(lib: ScalingLib, alg: ScalingAlg) -> "ScalingBackend":
        """Factory function to create backend instance.
        """
        # use lazy import to avoid importing overheads.
        if lib == ScalingLib.CV:
            from . import cv
            return cv.ScalingBackendCV(alg)
        if lib == ScalingLib.PIL:
            from . import pil
            return pil.ScalingBackendPIL(alg)
        raise NotImplementedError(f'Backend "{lib}" not implemented.')

    @abstractmethod
    def _scale(self, x: np.ndarray, shape: ShapeType) -> np.ndarray:
        """The actual implementation of `self.scale(...)`.
        """
        raise NotImplementedError

    @staticmethod
    def _validate_image(x: np.ndarray):
        if x.ndim not in [2, 3]:
            raise ValueError(f'Only support 2 or 3 dimensions, but got {x.ndim}.')
        if x.ndim == 3 and x.shape[-1] != 3:
            raise ValueError(f'Only support channel-last RGB data of shape (w, h, 3), but got {x.shape}.')
        if np.max(x) <= 1:
            logger.warning(f'The input image may not be in the correct range [0, 255], got [{np.min(x)}, {np.max(x)}].')
