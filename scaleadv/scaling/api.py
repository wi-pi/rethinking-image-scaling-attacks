"""
This module implements scaling backends as unified API.
"""
from typing import Optional, Union, Tuple

import numpy as np
import scipy.linalg as LA
from loguru import logger

from .backends import ShapeType, ScalingLib, ScalingAlg, ScalingBackend

# Mappings from str to enum.
_str_to_lib = {x.name: x for x in ScalingLib}
_str_to_alg = {x.name: x for x in ScalingAlg}


class ScalingAPI(object):
    """Numpy API for scaling algorithms.

    This API only accepts channel-first tensor-like arrays (as most usages are from tensors):
      * shape: gray (1, h, w) or rgb (3, h, w)
      * value: [0, 1]
      * dtype: np.floating

    The input will be converted to backends' acceptable arrays (as most backends accept channel-last arrays):
      * shape: gray (w, h) or rgb (w, h, 3)
      * value: [0, 255]
      * dtype: any

    Note:
      * Be aware of the shape's difference: array's (h, w) vs pic's (w, h)

    Usage:
      >>> api = ScalingAPI((448, 488,), (224, 224), 'cv', 'linear')
      >>> x_big = np.random.uniform(0, 1, size=(3, 448, 448))
      >>> x_small = api(x_big)
    """

    def __init__(
            self,
            src_shape: ShapeType,
            tgt_shape: ShapeType,
            lib: Union[ScalingLib, str],
            alg: Union[ScalingAlg, str],
            verbose: bool = True
    ):
        # Convert str to enums
        if isinstance(lib, str):
            lib = lib.upper()
            if lib not in _str_to_lib:
                raise ValueError(f'Scaling library {lib} not supported.')
            lib = _str_to_lib[lib]

        if isinstance(alg, str):
            alg = alg.upper()
            if alg not in _str_to_alg:
                raise ValueError(f'Scaling algorithm {alg} not supported.')
            alg = _str_to_alg[alg]

        self.src_shape = src_shape
        self.tgt_shape = tgt_shape
        self.lib = lib
        self.alg = alg
        self.backend = ScalingBackend.create(lib, alg)
        self.ratio = self._get_ratio(src_shape, tgt_shape)
        self.cl, self.cr = self._get_matrix()
        self.mask = self._get_mask()
        if verbose:
            logger.info(f'Create scaling api: src {src_shape}, tgt {tgt_shape}, lib {lib}, alg {alg}.')

    def __call__(self, x: np.ndarray, shape: Optional[ShapeType] = None):
        self._check_input(x)
        x = self._to_backend(x)
        x = self.backend.scale(x, (shape or self.tgt_shape)[::-1])
        x = self._from_backend(x)
        return x

    def _check_input(self, x: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f'Only support 3 dimensions, but got {x.ndim}.')
        if x.shape[0] not in [1, 3]:
            raise ValueError(f'Only support 1 or 3 channels, but got {x.shape[0]} channels.')
        if not issubclass(x.dtype.type, np.floating):
            raise ValueError(f'Only support np.floating types, but got {x.dtype}.')
        if np.min(x) < 0 or np.max(x) > 1:
            raise Warning(f'Image should be normalized to [0, 1], but got [{np.min(x)}, {np.max(x)}].')
        if x.shape[1:] != self.src_shape:
            raise ValueError(f'Image shape does not match, expecting {self.src_shape} but got {x.shape[1:]}')
        return x

    @staticmethod
    def _to_backend(x: np.ndarray) -> np.ndarray:
        # From: shape (c, h, w), value [0, 1]
        # To: shape (h, w, c), value [0, 255]
        x = x * 255
        x = x.transpose((1, 2, 0))
        if x.shape[-1] == 1:
            x = np.squeeze(x, -1)
        return x

    @staticmethod
    def _from_backend(x: np.ndarray) -> np.ndarray:
        # From: shape (h, w, c), value [0, 255]
        # To: shape (c, h, w), value [0, 1]
        x = np.atleast_3d(x)  # view x with at least 3 dims
        x = x.transpose((2, 0, 1))
        x = x / 255
        return x

    @staticmethod
    def _get_ratio(src_shape: ShapeType, tgt_shape: ShapeType) -> float:
        # Calculate ratios at each dim and choose the smallest one
        ratios = [x / y for x, y in zip(src_shape, tgt_shape)]
        return min(ratios)

    def _get_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        def infer(size, width, height):
            x = np.identity(size) * 255
            x = self.backend.scale(x, (width, height))
            return x / 255

        cl = infer(self.src_shape[0], self.src_shape[0], self.tgt_shape[0])
        cr = infer(self.src_shape[1], self.tgt_shape[1], self.src_shape[1])
        cl = cl / cl.sum(axis=1, keepdims=True)
        cr = cr / cr.sum(axis=0, keepdims=True)
        return cl, cr

    def _get_mask(self) -> np.ndarray:
        cli, cri = map(LA.pinv, [self.cl, self.cr])
        mask = cli @ np.ones(self.tgt_shape) @ cri
        mask = mask.round().astype(np.uint8)
        return mask

    def __repr__(self):
        return f'ScalingAPI(lib={self.lib}, alg={self.alg}, src={self.src_shape}, tgt={self.tgt_shape})'
