import abc
import types

import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as np
    import numpy.typing as cpt


class Norm(abc.ABC):

    def __init__(self, xp: types.ModuleType):
        self._xp = xp

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        raise NotImplementedError


class SimpleNorm(Norm):
    """simple norms that are not differentiable, but where the prox operator of the convex dual is simple"""

    @abc.abstractmethod
    def prox_convex_dual(self, x: npt.NDArray | cpt.NDArray,
                         sigma: float) -> npt.NDArray | cpt.NDArray:
        """proximal operator of the convex dual of the norm"""
        raise NotImplementedError


class SmoothNorm(Norm):
    """simple norms that are not differentiable, but where the prox operator of the convex dual is simple"""

    @abc.abstractmethod
    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of the norm"""
        raise NotImplementedError


class L2L1Norm(SimpleNorm):
    """sum of pointwise Eucliean norms (L2L1 norm)"""

    def __init__(self, xp: types.ModuleType):
        super(SimpleNorm, self).__init__(xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return self.xp.linalg.norm(x, axis=0).sum()

    def prox_convex_dual(self, x: npt.NDArray | cpt.NDArray,
                         sigma: float) -> npt.NDArray | cpt.NDArray:
        gnorm = self._xp.linalg.norm(x, axis=0)
        r = x / self._xp.clip(gnorm, 1, None)

        return r


class SquaredL2Norm(SimpleNorm, SmoothNorm):
    """squared L2 norm times 0.5"""

    def __init__(self, xp: types.ModuleType):
        super(SimpleNorm, self).__init__(xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return 0.5 * (x**2).sum()

    def prox_convex_dual(self, x: npt.NDArray | cpt.NDArray,
                         sigma: float) -> npt.NDArray | cpt.NDArray:
        return x / (1 + sigma)

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return x