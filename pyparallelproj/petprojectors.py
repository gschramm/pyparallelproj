import numpy as np
import numpy.typing as npt

import pyparallelproj.coincidences as coincidences
import pyparallelproj.wrapper as wrapper
import pyparallelproj.subsets as subsets
import pyparallelproj.operators as operators

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class NonTOFPETJosephProjector(operators.LinearOperator):

    def __init__(self,
                 coincidence_descriptor: coincidences.PETCoincidenceDescriptor,
                 image_shape: tuple[int, int, int],
                 image_origin: tuple[float, float,
                                     float], voxel_size: tuple[float, float,
                                                               float],
                 subsetter: subsets.LORSubsetter) -> None:

        self._coincidence_descriptor = coincidence_descriptor
        self._image_shape = image_shape
        self._image_origin = image_origin
        self._voxel_size = voxel_size
        self._subsetter = subsetter

        super().__init__(
            self._image_shape,
            (np.prod(self._coincidence_descriptor.sinogram_spatial_shape), ),
            self.coincidence_descriptor.scanner.xp)

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
    def subsetter(self) -> subsets.LORSubsetter:
        return self._subsetter

    def forward_subset(
            self,
            x: npt.NDArray | cpt.NDArray,
            subset: int = 0,
            lors: None | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:

        if lors is None:
            lors = self.subsetter.get_subset_indices(subset)

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            lors)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        image_forward = self.xp.zeros(xstart.shape[0], dtype=self.xp.float32)

        wrapper.joseph3d_fwd(xstart, xend, x, self.image_origin,
                             self.voxel_size, image_forward)

        return image_forward

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        image_forward = self.xp.zeros(self.coincidence_descriptor.num_lors,
                                      dtype=self.xp.float32)

        for subset in range(self.subsetter.num_subsets):
            lors = self.subsetter.get_subset_indices(subset)
            image_forward[lors] = self.forward_subset(x, lors=lors)

        return image_forward

    def adjoint_subset(
            self,
            y_subset: npt.NDArray | cpt.NDArray,
            subset: int = 0,
            lors: None | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:

        if lors is None:
            lors = self.subsetter.get_subset_indices(subset)

        start_mod, start_ind, end_mod, end_ind = self.coincidence_descriptor.get_lor_indices(
            lors)
        xstart = self.coincidence_descriptor.scanner.get_lor_endpoints(
            start_mod, start_ind).astype(self.xp.float32)
        xend = self.coincidence_descriptor.scanner.get_lor_endpoints(
            end_mod, end_ind).astype(self.xp.float32)

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)
        wrapper.joseph3d_back(xstart, xend, back_image, self.image_origin,
                              self.voxel_size, y_subset)

        return back_image

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        back_image = self.xp.zeros(self.image_shape, dtype=self.xp.float32)

        for subset in range(self.subsetter.num_subsets):
            lors = self.subsetter.get_subset_indices(subset)
            back_image += self.adjoint_subset(y[lors])

        return back_image
