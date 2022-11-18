import types
import numpy.typing as npt

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt

import pyparallelproj.operators as operators


class GaussianImageBasedResolutionModel(operators.LinearOperator):
    """Image based resolution model using a shift invariant Gaussian convolution"""

    def __init__(self, input_shape: tuple[int, ...],
                 sigma: float | tuple[float, ...], xp: types.ModuleType,
                 ndi: types.ModuleType) -> None:

        super().__init__(input_shape, input_shape, xp)

        self._sigma = sigma
        self._ndi = ndi

    @property
    def sigma(self) -> float | tuple[float, ...]:
        return self._sigma

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self._ndi.gaussian_filter(x, self.sigma)

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self._ndi.gaussian_filter(y, self.sigma)