import numpy as np
import numpy.typing as npt

import pyparallelproj.operators as operators
import pyparallelproj.petprojectors as petprojectors
import pyparallelproj.subsets as subsets

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class PETAcquisitionModel(operators.LinearListmodeSubsetOperator):

    def __init__(self,
                 projector: petprojectors.PETProjector,
                 multiplicative_corrections: npt.NDArray | cpt.NDArray,
                 multiplicative_correction_list: None | npt.NDArray
                 | cpt.NDArray = None,
                 image_based_resolution_model: None
                 | operators.LinearOperator = None):

        self._projector = projector
        self._image_based_resolution_model = image_based_resolution_model

        super().__init__(projector.image_shape, projector.output_shape,
                         projector.coincidence_descriptor.scanner.xp,
                         projector.subsetter)

        if self._projector.tof:
            self._multiplicative_corrections = self.xp.expand_dims(
                multiplicative_corrections, -1)
        else:
            self._multiplicative_corrections = multiplicative_corrections

        self._multiplicative_correction_list = multiplicative_correction_list

    @property
    def projector(self) -> petprojectors.PETProjector:
        return self._projector

    @property
    def multiplicative_corrections(self) -> npt.NDArray | cpt.NDArray:
        return self._multiplicative_corrections

    @property
    def multiplicative_correction_list(
            self) -> None | npt.NDArray | cpt.NDArray:
        return self._multiplicative_correction_list

    @multiplicative_correction_list.setter
    def multiplicative_correction_list(
            self, value: None | npt.NDArray | cpt.NDArray) -> None:
        self._multiplicative_correction_list = value

    @property
    def image_based_resolution_model(self) -> None | operators.LinearOperator:
        return self._image_based_resolution_model

    @property
    def listmode_subsetter(self) -> subsets.Strided1DSubsetter:
        return self._projector.listmode_subsetter

    @listmode_subsetter.setter
    def listmode_subsetter(self, value: subsets.Strided1DSubsetter) -> None:
        self.projector.listmode_subsetter = value

    @property
    def events(self) -> npt.NDArray | cpt.NDArray:
        return self.projector.events

    @events.setter
    def events(self, value: npt.NDArray | cpt.NDArray) -> None:
        self.projector.events = value
        self.projector.listmode_subsetter.num_elements = self.projector.events.shape[
            0]

    # abstract methods to be implemented
    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        return self.projector.get_subset_shape(subset)

    def forward_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.image_based_resolution_model is not None:
            x = self.image_based_resolution_model.forward(x)

        x_forward = self._multiplicative_corrections[
            subset_inds] * self.projector.forward_subset(x, subset_inds)

        return x_forward

    def adjoint_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        y_back = self.projector.adjoint_subset(
            self._multiplicative_corrections[subset_inds] * y_subset,
            subset_inds)

        if self.image_based_resolution_model is not None:
            y_back = self.image_based_resolution_model.adjoint(y_back)

        return y_back

    def forward_listmode_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self._multiplicative_correction_list is None:
            raise ValueError

        if self.image_based_resolution_model is not None:
            x = self.image_based_resolution_model.forward(x)

        x_forward = self._multiplicative_correction_list[
            subset_inds] * self.projector.forward_listmode_subset(
                x, subset_inds)

        return x_forward

    def adjoint_listmode_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self._multiplicative_correction_list is None:
            raise ValueError

        y_back = self.projector.adjoint_listmode_subset(
            self._multiplicative_correction_list[subset_inds] * y_subset,
            subset_inds)

        if self.image_based_resolution_model is not None:
            y_back = self.image_based_resolution_model.adjoint(y_back)

        return y_back
