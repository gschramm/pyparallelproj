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


class PETAcquisitionModel(operators.LinearOperator):

    def __init__(self,
                 projector: petprojectors.PETProjector,
                 attenuation_factors: npt.NDArray | cpt.NDArray,
                 sensitivity_factors: npt.NDArray | cpt.NDArray,
                 image_based_resolution_model=None):

        self._projector = projector
        self._image_based_resolution_model = image_based_resolution_model

        super().__init__(projector.image_shape, projector.output_shape,
                         projector.coincidence_descriptor.scanner.xp)

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
    def image_based_resolution_model(self) -> None:
        return self._image_based_resolution_model

    def forward_subset(self,
                       image: npt.NDArray | cpt.NDArray,
                       subset: int = 0,
                       lors=None) -> npt.NDArray | cpt.NDArray:

        if lors is None:
            lors = self.projector.subsetter.get_subset_indices(subset)

        img_fwd_subset = self.sensitivity_factors[
            lors] * self.attenuation_factors[
                lors] * self.projector.forward_subset(image, lors=lors)

        return img_fwd_subset

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        image_forward = self.xp.zeros(self.output_shape, dtype=self.xp.float32)

        for subset in range(self.projector.subsetter.num_subsets):
            lors = self.projector.subsetter.get_subset_indices(subset)
            image_forward[lors] = self.forward_subset(x, lors=lors)

        return image_forward

    def adjoint_subset(
            self,
            y_subset: npt.NDArray | cpt.NDArray,
            subset: int = 0,
            lors: None | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:

        if lors is None:
            lors = self.projector.subsetter.get_subset_indices(subset)

        back_image = self.projector.adjoint_subset(
            self.sensitivity_factors[lors] * self.attenuation_factors[lors] *
            y_subset,
            lors=lors)

        return back_image

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        back_image = self.xp.zeros(self.input_shape, dtype=self.xp.float32)

        for subset in range(self.projector.subsetter.num_subsets):
            lors = self.projector.subsetter.get_subset_indices(subset)
            back_image += self.adjoint_subset(y[lors], subset=subset)

        return back_image
