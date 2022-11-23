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


class Functional(abc.ABC):
    """abstract base class for a norm"""

    def __init__(self, xp: types.ModuleType):
        self._xp = xp

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        raise NotImplementedError


class FunctionalWithDualProx(Functional):

    @abc.abstractmethod
    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """proximal operator of the convex dual of the functional"""
        raise NotImplementedError


class FunctionalWithProx(Functional):

    @abc.abstractmethod
    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """proximal operator of the functional"""
        raise NotImplementedError


class SmoothFunctional(Functional):
    """smooth functional with gradient"""

    @abc.abstractmethod
    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of the functional"""
        raise NotImplementedError


class BoundIndicatorFunctional(FunctionalWithProx):

    def __init__(self,
                 xp: types.ModuleType,
                 lb: float | None = None,
                 ub: float | None = None):
        super().__init__(xp)

        if lb is None:
            self._lb = -self.xp.inf
        else:
            self._lb = lb

        if ub is None:
            self._ub = self.xp.inf
        else:
            self._ub = ub

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        res = 0

        if x.max() > self._ub:
            res = self.xp.inf
        if x.min() < self._lb:
            res = self.xp.inf

        return res

    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        return self.xp.clip(x, self._lb, self._ub)


class L2L1Norm(FunctionalWithDualProx):
    """sum of pointwise Eucliean norms (L2L1 norm)"""

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return self.xp.linalg.norm(x, axis=0).sum()

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        gnorm = self._xp.linalg.norm(x, axis=0)
        r = x / self._xp.clip(gnorm, 1, None)

        return r


class SquaredL2Norm(SmoothFunctional, FunctionalWithDualProx):
    """squared L2 norm times 0.5"""

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


class SquaredL2NormDistance(SmoothFunctional, FunctionalWithDualProx):

    def __init__(self, y: npt.NDArray | cpt.NDArray, xp: types.ModuleType):
        super().__init__(xp)
        self._y = y
        self._sql2norm = SquaredL2Norm(xp)

    @property
    def y(self) -> npt.NDArray | cpt.NDArray:
        return self._y

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return self._sql2norm(x - self._y)

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return x - self._y

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        return (x - sigma * self._y) / (1 + sigma)


class NegativePoissonLogLikelihood(SmoothFunctional, FunctionalWithDualProx):

    def __init__(self, y: npt.NDArray | cpt.NDArray, xp: types.ModuleType):
        super().__init__(xp)
        self._y = y

    @property
    def y(self) -> npt.NDArray | cpt.NDArray:
        return self._y

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        return (x - self._y * self.xp.log(x)).sum()

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return 1 - (self._y / x)

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        return 0.5 * (x + 1 - self.xp.sqrt(
            (x - 1)**2 + 4 * sigma * self._y)).astype(x.dtype)
