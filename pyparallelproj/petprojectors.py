import abc
import numpy as np
import numpy.typing as npt

import pyparallelproj.coincidences as coincidences
import pyparallelproj.wrapper as wrapper
import pyparallelproj.subsets as subsets
import pyparallelproj.operators as operators
import pyparallelproj.tof as tof

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

    def __init__(self,
                 coincidence_descriptor: coincidences.PETCoincidenceDescriptor,
                 image_shape: tuple[int, int, int],
                 image_origin: tuple[float, float, float],
                 voxel_size: tuple[float, float, float],
                 subsetter: subsets.Subsetter | None = None,
                 tof_parameters: tof.TOFParameters | None = None) -> None:

        self._coincidence_descriptor = coincidence_descriptor
        self._image_shape = image_shape
        self._image_origin = image_origin
        self._voxel_size = voxel_size
        self._tof_parameters = tof_parameters

        if self.tof:
            output_shape = (self.coincidence_descriptor.num_lors,
                            self.tof_parameters.num_tofbins)
        else:
            output_shape = (self.coincidence_descriptor.num_lors, )

        super().__init__(self.image_shape, output_shape,
                         self.coincidence_descriptor.scanner.xp, subsetter)

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

    @property
    def tof(self) -> bool:
        return self.tof_parameters is not None

    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        if self.tof:
            subset_shape = (self.subsetter.get_subset_index_len(subset),
                            self.tof_parameters.num_tofbins)
        else:
            subset_shape = (self.subsetter.get_subset_index_len(subset), )

        return subset_shape


class NonTOFPETJosephProjector(PETProjector):

    def forward_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        image_forward = self.xp.zeros(xstart.shape[0], dtype=self.xp.float32)

        wrapper.joseph3d_fwd(xstart, xend, x.astype(self.xp.float32),
                             self.image_origin, self.voxel_size, image_forward)

        return image_forward

    def adjoint_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)
        wrapper.joseph3d_back(xstart, xend, back_image,
                              self.image_origin, self.voxel_size,
                              y_subset.astype(self.xp.float32))

        return back_image

    def forward_listmode_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 0],
            self.events[subset_inds, 1]).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 2],
            self.events[subset_inds, 3]).astype(self.xp.float32)

        image_forward = self.xp.zeros(xstart.shape[0], dtype=self.xp.float32)

        wrapper.joseph3d_fwd(xstart, xend, x.astype(self.xp.float32),
                             self.image_origin, self.voxel_size, image_forward)

        return image_forward

    def adjoint_listmode_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 0],
            self.events[subset_inds, 1]).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 2],
            self.events[subset_inds, 3]).astype(self.xp.float32)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)
        wrapper.joseph3d_back(xstart, xend, back_image,
                              self.image_origin, self.voxel_size,
                              y_subset.astype(self.xp.float32))

        return back_image


class TOFPETJosephProjector(PETProjector):

    def forward_subset(
            self, x: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        if self.tof_parameters is None:
            raise ValueError

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        image_forward = self.xp.zeros(
            (xstart.shape[0], self.tof_parameters.num_tofbins),
            dtype=self.xp.float32)

        wrapper.joseph3d_fwd_tof_sino(
            xstart, xend, x.astype(self.xp.float32), self.image_origin,
            self.voxel_size, image_forward, self.tof_parameters.tofbin_width,
            self.xp.array([self.tof_parameters.sigma_tof],
                          dtype=self.xp.float32),
            self.xp.array([self.tof_parameters.tofcenter_offset],
                          dtype=self.xp.float32),
            self.tof_parameters.num_sigmas, self.tof_parameters.num_tofbins)

        return image_forward

    def adjoint_subset(
            self, y_subset: npt.NDArray | cpt.NDArray,
            subset_inds: slice | npt.NDArray) -> npt.NDArray | cpt.NDArray:

        if self.tof_parameters is None:
            raise ValueError

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            subset_inds)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)
        wrapper.joseph3d_back_tof_sino(
            xstart, xend, back_image, self.image_origin, self.voxel_size,
            y_subset.astype(self.xp.float32), self.tof_parameters.tofbin_width,
            self.xp.array([self.tof_parameters.sigma_tof],
                          dtype=self.xp.float32),
            self.xp.array([self.tof_parameters.tofcenter_offset],
                          dtype=self.xp.float32),
            self.tof_parameters.num_sigmas, self.tof_parameters.num_tofbins)

        return back_image

    def forward_listmode_subset(
        self, x: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.tof_parameters is None:
            raise ValueError

        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self._events[subset_inds, 0],
            self._events[subset_inds, 1]).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self._events[subset_inds, 2],
            self._events[subset_inds, 3]).astype(self.xp.float32)
        tofbin = self._events[subset_inds, 4].astype(self._xp.int16)

        image_forward = self.xp.zeros(xstart.shape[0], dtype=self.xp.float32)

        wrapper.joseph3d_fwd_tof_lm(
            xstart, xend, x.astype(self.xp.float32), self.image_origin,
            self.voxel_size, image_forward, self.tof_parameters.tofbin_width,
            self.xp.array([self.tof_parameters.sigma_tof],
                          dtype=self.xp.float32),
            self.xp.array([self.tof_parameters.tofcenter_offset],
                          dtype=self.xp.float32),
            self.tof_parameters.num_sigmas, tofbin)

        return image_forward

    def adjoint_listmode_subset(
        self, y_subset: npt.NDArray | cpt.NDArray,
        subset_inds: slice | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:

        if self.tof_parameters is None:
            raise ValueError

        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 0],
            self.events[subset_inds, 1]).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            self.events[subset_inds, 2],
            self.events[subset_inds, 3]).astype(self.xp.float32)
        tofbin = self.events[subset_inds, 4].astype(self._xp.int16)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)

        wrapper.joseph3d_back_tof_lm(
            xstart, xend, back_image, self.image_origin, self.voxel_size,
            y_subset.astype(self.xp.float32), self.tof_parameters.tofbin_width,
            self.xp.array([self.tof_parameters.sigma_tof],
                          dtype=self.xp.float32),
            self.xp.array([self.tof_parameters.tofcenter_offset],
                          dtype=self.xp.float32),
            self.tof_parameters.num_sigmas, tofbin)

        return back_image
