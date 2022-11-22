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
    """abstract base class for a norm"""

    def __init__(self, xp: types.ModuleType):
        self._xp = xp

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """proximal operator of the convex dual of the norm"""
        raise NotImplementedError


class SmoothNorm(Norm):
    """simple norms that are not differentiable, but where the prox operator of the convex dual is simple"""

    @abc.abstractmethod
    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of the norm"""
        raise NotImplementedError


class L2L1Norm(Norm):
    """sum of pointwise Eucliean norms (L2L1 norm)"""

    def __init__(self, xp: types.ModuleType):
        super().__init__(xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return self.xp.linalg.norm(x, axis=0).sum()

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        gnorm = self._xp.linalg.norm(x, axis=0)
        r = x / self._xp.clip(gnorm, 1, None)

        return r


class SquaredL2Norm(SmoothNorm):
    """squared L2 norm times 0.5"""

    def __init__(self, xp: types.ModuleType):
        super().__init__(xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return 0.5 * (x**2).sum()

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        return x / (1 + sigma)

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return x


class Distance(abc.ABC):
    """abstract base class for a distance (metric) between two vectors x and y"""

    def __init__(self, y: npt.NDArray | cpt.NDArray, xp: types.ModuleType):
        self._y = y
        self._xp = xp

    @property
    def y(self) -> npt.NDArray | cpt.NDArray:
        return self._y

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        """calculate the distance vector x (and y)"""
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """proximal operator of the convex dual of the distance"""
        raise NotImplementedError


class SmoothDistance(Distance):

    @abc.abstractmethod
    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of the distance with respect to first argument"""
        raise NotImplementedError


class SquaredL2NormDistance(SmoothDistance):

    def __init__(self, y: npt.NDArray | cpt.NDArray, xp: types.ModuleType):
        super().__init__(y, xp)
        self._sql2norm = SquaredL2Norm(xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return self._sql2norm(x - self.y)

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return x - self.y

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        return (x - sigma * self.y) / (1 + sigma)


class NegativePoissonLogLikelihood(SmoothDistance):

    def __init__(self, y: npt.NDArray | cpt.NDArray, xp: types.ModuleType):
        super().__init__(y, xp)

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return (x - self.y * self.xp.log(x)).sum()

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return 1 - (self.y / x)

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        return 0.5 * (x + 1 - self.xp.sqrt((x - 1)**2 + 4 * sigma * self.y))
