"""
This module implements core components of unified scaling backend.
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from loguru import logger

from scaleadv.scaling.enum import ScalingAlg, ScalingLib

# type alias
Shape = tuple[int, int]


class ScalingBackend(ABC):
    """A unified abstract backend for scaling libraries, such as CV and PIL.

    The backend requires inputs to have:
      * shape: GRAY (w, h) or RGB (w, h, 3)
      * value: [0, 255]
      * type: any

    Notes:
      * Value range is restricted to [0, 255] because some backends (PIL) do not support normalized range [0, 1].
      * Acceptable shape is (w, h), not (h, w).
    """

    # Mapping from our unified algorithm enum to lib-specific algorithm enum,
    # should be implemented by subclasses.
    algorithms: dict[ScalingAlg, Any] = {}

    def __init__(self, alg: ScalingAlg):
        if alg not in self.algorithms.keys():
            raise NotImplementedError(f'{self.__class__.__name__} does not support algorithm "{alg}".')
        self.alg = self.algorithms[alg]

    def scale(self, x: np.ndarray, shape: Shape) -> np.ndarray:
        """Scale an image to the specified shape.

        Args:
            x: input image of shape (w, h) or (w, h, 3), and range [0, 255].
            shape: a tuple indicating target shape (w, h).

        Return:
            Scaled image with the same dtype.
        """
        logger.debug(f'Scaling {x.dtype} image from {x.shape} to {shape}.')
        if x.ndim not in [2, 3]:
            raise ValueError(f'Only support 2 or 3 dimensions, but got {x.ndim}.')
        if x.ndim == 3 and x.shape[-1] != 3:
            raise ValueError(f'Only support channel-last RGB data of shape (w, h, 3), but got {x.shape}.')
        if np.max(x) <= 1:
            logger.warning(f'The input image may not be in the correct range [0, 255], got [{np.min(x)}, {np.max(x)}].')

        # do the real scaling
        x = self._scale(x, shape)

        return x

    @abstractmethod
    def _scale(self, x: np.ndarray, shape: Shape) -> np.ndarray:
        """The actual implementation of `self.scale(...)`.
        """
        raise NotImplementedError


def get_backend(lib: ScalingLib, alg: ScalingAlg) -> ScalingBackend:
    """Retrieve scaling backend given the lib and alg.
    """
    # use lazy import to avoid importing overheads.
    if lib == ScalingLib.CV:
        from scaleadv.scaling import cv
        return cv.ScalingBackendCV(alg)

    if lib == ScalingLib.PIL:
        from scaleadv.scaling import pil
        return pil.ScalingBackendPIL(alg)

    raise NotImplementedError(f'Backend "{lib}" not implemented.')
