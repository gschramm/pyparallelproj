import abc
import numpy as np
import numpy.typing as npt
import parallelproj

from . import coincidences
from . import subsets
from . import operators
from . import tof
from . import listmode

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class PETProjector(operators.LinearListmodeSubsetOperator):
    """
    Abstract base class for NonTOF and TOF PET projectors
    """

    def __init__(
        self,
        coincidence_descriptor: coincidences.PETCoincidenceDescriptor,
        image_shape: tuple[int, int, int],
        image_origin: tuple[float, float, float],
        voxel_size: tuple[float, float, float],
        subsetter: subsets.Subsetter | None = None,
    ) -> None:

        self._coincidence_descriptor = coincidence_descriptor
        self._image_shape = image_shape
        self._image_origin = image_origin
        self._voxel_size = voxel_size

        self._tof_parameters: tof.TOFParameters | None = None
        self._image_based_resolution_model: operators.LinearOperator | None = None
        self._multiplicative_corrections: npt.NDArray | cpt.NDArray | None = None
        self._multiplicative_correction_list: npt.NDArray | cpt.NDArray | None = None

        super().__init__(self.image_shape, self.output_shape,
                         self.coincidence_descriptor.scanner.xp, subsetter)

        self._events: listmode.PETListmodeEvents = listmode.GenericPETListmodeEvents(
            self.xp.zeros(1).astype(self.xp.int16),
            self.xp.zeros(1).astype(self.xp.int16),
            self.xp.zeros(1).astype(self.xp.int16),
            self.xp.zeros(1).astype(self.xp.int16),
            self.xp.zeros(1).astype(self.xp.int16),
            self._coincidence_descriptor.scanner)

    @property
    def events(self) -> listmode.PETListmodeEvents:
        return self._events

    @events.setter
    def events(self, value: listmode.PETListmodeEvents) -> None:
        self._events = value
        self._listmode_subsetter.num_elements = self._events.num_events

    @property
    def num_events(self) -> int:
        return self._events.num_events

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.tof:
            output_shape = (self.coincidence_descriptor.num_lors,
                            self.tof_parameters.num_tofbins)
        else:
            output_shape = (self.coincidence_descriptor.num_lors, )

        self._output_shape = output_shape

        return output_shape

    @property
    def coincidence_descriptor(self) -> coincidences.PETCoincidenceDescriptor:
        return self._coincidence_descriptor

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._image_shape

    @property
    def image_origin(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._image_origin, dtype=self.xp.float32)

    @property
    def voxel_size(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._voxel_size, dtype=self.xp.float32)

    @property
    def tof_parameters(self) -> tof.TOFParameters | None:
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: tof.TOFParameters | None) -> None:
        self._tof_parameters = value

    @property
    def tof(self) -> bool:
        return self.tof_parameters is not None

    @property
    def multiplicative_corrections(self) -> npt.NDArray | cpt.NDArray:
        return self._multiplicative_corrections

    @multiplicative_corrections.setter
    def multiplicative_corrections(
            self, value: npt.NDArray | cpt.NDArray | None) -> None:
        if value is None:
            self._multiplicative_corrections = value
        else:
            if self.tof and (value.ndim == 1):
                self._multiplicative_corrections = np.expand_dims(value, -1)
            else:
                self._multiplicative_corrections = value

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

    @image_based_resolution_model.setter
    def image_based_resolution_model(self,
                                     value: operators.LinearOperator) -> None:
        self._image_based_resolution_model = value

    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        if self.tof:
            subset_shape = (self.subsetter.get_subset_index_len(subset),
                            self.tof_parameters.num_tofbins)
        else:
            subset_shape = (self.subsetter.get_subset_index_len(subset), )

        return subset_shape

    @abc.abstractmethod
    def forward_geometrical_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint_geometrical_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def forward_geometrical_listmode_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def adjoint_geometrical_listmode_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:
        raise NotImplementedError

    def forward_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.image_based_resolution_model is not None:
            x = self.image_based_resolution_model.forward(x)

        x_forward = self.forward_geometrical_subset(x, subset_inds)

        if self.multiplicative_corrections is not None:
            x_forward *= self.multiplicative_corrections[subset_inds]

        return x_forward

    def adjoint_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.multiplicative_corrections is not None:
            y_subset = y_subset * self.multiplicative_corrections[subset_inds]

        y_back = self.adjoint_geometrical_subset(y_subset, subset_inds)

        if self.image_based_resolution_model is not None:
            y_back = self.image_based_resolution_model.adjoint(y_back)

        return y_back

    def forward_listmode_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.image_based_resolution_model is not None:
            x = self.image_based_resolution_model.forward(x)

        x_forward = self.forward_geometrical_listmode_subset(x, subset_inds)

        if self.multiplicative_correction_list is not None:
            x_forward *= self.multiplicative_correction_list[subset_inds]

        return x_forward

    def adjoint_listmode_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.multiplicative_correction_list is not None:
            y_subset = y_subset * self.multiplicative_correction_list[
                subset_inds]

        y_back = self.adjoint_geometrical_listmode_subset(
            y_subset, subset_inds)

        if self.image_based_resolution_model is not None:
            y_back = self.image_based_resolution_model.adjoint(y_back)

        return y_back


class PETJosephProjector(PETProjector):

    def forward_geometrical_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        if not self.tof:
            image_forward = self.xp.zeros(xstart.shape[0],
                                          dtype=self.xp.float32)
            parallelproj.joseph3d_fwd(xstart, xend, x.astype(self.xp.float32),
                                      self.image_origin, self.voxel_size,
                                      image_forward)
        else:
            image_forward = self.xp.zeros(
                (xstart.shape[0], self.tof_parameters.num_tofbins),
                dtype=self.xp.float32)
            parallelproj.joseph3d_fwd_tof_sino(
                xstart, xend, x.astype(self.xp.float32), self.image_origin,
                self.voxel_size, image_forward,
                self.tof_parameters.tofbin_width,
                self.xp.array([self.tof_parameters.sigma_tof],
                              dtype=self.xp.float32),
                self.xp.array([self.tof_parameters.tofcenter_offset],
                              dtype=self.xp.float32),
                self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return image_forward

    def adjoint_geometrical_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)

        if not self.tof:
            parallelproj.joseph3d_back(xstart, xend, back_image,
                                       self.image_origin, self.voxel_size,
                                       y_subset.astype(self.xp.float32))
        else:
            parallelproj.joseph3d_back_tof_sino(
                xstart, xend, back_image, self.image_origin, self.voxel_size,
                y_subset.astype(self.xp.float32),
                self.tof_parameters.tofbin_width,
                self.xp.array([self.tof_parameters.sigma_tof],
                              dtype=self.xp.float32),
                self.xp.array([self.tof_parameters.tofcenter_offset],
                              dtype=self.xp.float32),
                self.tof_parameters.num_sigmas,
                self.tof_parameters.num_tofbins)

        return back_image

    def forward_geometrical_listmode_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        xstart = self.events.get_event_lor_start_coordinates(subset_inds)
        xend = self.events.get_event_lor_end_coordinates(subset_inds)

        image_forward = self.xp.zeros(xstart.shape[0], dtype=self.xp.float32)

        if not self.tof:
            parallelproj.joseph3d_fwd(xstart, xend, x.astype(self.xp.float32),
                                      self.image_origin, self.voxel_size,
                                      image_forward)
        else:
            tofbin = self.xp.ascontiguousarray(
                self.events.get_event_tof_bins(subset_inds))

            parallelproj.joseph3d_fwd_tof_lm(
                xstart, xend, x.astype(self.xp.float32), self.image_origin,
                self.voxel_size, image_forward,
                self.tof_parameters.tofbin_width,
                self.xp.array([self.tof_parameters.sigma_tof],
                              dtype=self.xp.float32),
                self.xp.array([self.tof_parameters.tofcenter_offset],
                              dtype=self.xp.float32),
                self.tof_parameters.num_sigmas, tofbin)

        return image_forward

    def adjoint_geometrical_listmode_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        xstart = self.events.get_event_lor_start_coordinates(subset_inds)
        xend = self.events.get_event_lor_end_coordinates(subset_inds)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)

        if not self.tof:
            parallelproj.joseph3d_back(xstart, xend, back_image,
                                       self.image_origin, self.voxel_size,
                                       y_subset.astype(self.xp.float32))
        else:
            tofbin = self.xp.ascontiguousarray(
                self.events.get_event_tof_bins(subset_inds))

            parallelproj.joseph3d_back_tof_lm(
                xstart, xend, back_image, self.image_origin, self.voxel_size,
                y_subset.astype(self.xp.float32),
                self.tof_parameters.tofbin_width,
                self.xp.array([self.tof_parameters.sigma_tof],
                              dtype=self.xp.float32),
                self.xp.array([self.tof_parameters.tofcenter_offset],
                              dtype=self.xp.float32),
                self.tof_parameters.num_sigmas, tofbin)

        return back_image