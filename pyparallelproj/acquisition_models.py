import numpy as np
import numpy.typing as npt

import pyparallelproj.operators as operators
import pyparallelproj.petprojectors as petprojectors

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class PETAcquisitionModel(operators.LinearSubsetOperator):

    def __init__(self,
                 projector: petprojectors.PETProjector,
                 attenuation_factors: npt.NDArray | cpt.NDArray,
                 sensitivity_factors: npt.NDArray | cpt.NDArray,
                 image_based_resolution_model: None
                 | operators.LinearOperator = None):

        self._projector = projector
        self._image_based_resolution_model = image_based_resolution_model

        super().__init__(projector.image_shape, projector.output_shape,
                         projector.coincidence_descriptor.scanner.xp,
                         projector.subsetter)

        if self._projector.tof:
            self._attenuation_factors = self.xp.expand_dims(
                attenuation_factors, -1)
            self._sensitivity_factors = self.xp.expand_dims(
                sensitivity_factors, -1)
        else:
            self._attenuation_factors = attenuation_factors
            self._sensitivity_factors = sensitivity_factors

    @property
    def projector(self) -> petprojectors.PETProjector:
        return self._projector

    @property
    def attenuation_factors(self) -> npt.NDArray | cpt.NDArray:
        return self._attenuation_factors

    @property
    def sensitivity_factors(self) -> npt.NDArray | cpt.NDArray:
        return self._sensitivity_factors

    @property
    def image_based_resolution_model(self) -> None | operators.LinearOperator:
        return self._image_based_resolution_model

    # abstract methods to be implemented

    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        return self.projector.get_subset_shape(subset)

    def forward_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.image_based_resolution_model is not None:
            x = self.image_based_resolution_model.forward(x)

        x_forward = self.sensitivity_factors[
            subset_inds] * self.attenuation_factors[
                subset_inds] * self.projector.forward_subset(x, subset_inds)

        return x_forward

    def adjoint_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        y_back = self.projector.adjoint_subset(
            self.sensitivity_factors[subset_inds] *
            self.attenuation_factors[subset_inds] * y_subset, subset_inds)

        if self.image_based_resolution_model is not None:
            y_back = self.image_based_resolution_model.forward(y_back)

        return y_back