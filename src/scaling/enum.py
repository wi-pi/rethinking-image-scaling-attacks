from enum import Enum


class ScalingLib(Enum):
    """Enumeration of supported scaling libraries.
    """
    CV = 1
    PIL = 2


class ScalingAlg(Enum):
    """Enumeration of supported scaling algorithms.
    """
    NEAREST = 1
    LINEAR = 2
    CUBIC = 3
    LANCZOS = 4
    AREA = 5


str_to_lib = {x.name.lower(): x for x in ScalingLib if isinstance(x, Enum)}
str_to_alg = {x.name.lower(): x for x in ScalingAlg if isinstance(x, Enum)}
