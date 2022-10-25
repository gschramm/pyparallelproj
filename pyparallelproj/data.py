#TODO subset slicer subset = a[subset_slicer(i)]

import abc
import enum
import itertools

import math

import numpy as np
import numpy.typing as npt
try:
    import cupy.typing as cpt
except:
    import numpy.typing as cpt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pyparallelproj.scanner_modules as scanners


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""
    RVP = enum.auto()
    RPV = enum.auto()
    VRP = enum.auto()
    VPR = enum.auto()
    PRV = enum.auto()
    PVR = enum.auto()


class PETCoincidenceDescriptor(abc.ABC):

    def __init__(self,
                 scanner: scanners.ModularizedPETScannerGeometry) -> None:
        self._scanner = scanner

    @property
    def scanner(self) -> scanners.ModularizedPETScannerGeometry:
        return self._scanner

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods

    @abc.abstractmethod
    def get_modules_and_indices_in_coincidence(
            self, module: int, index_in_module: int) -> npt.NDArray:
        """ return (N,2) array of two integers showing which module/index_in_module combinations
            are in coincidence with the given input module / index_in_module
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """ mapping that maps the linear LOR index to the 4 1D arrays
            representing start_module, start_index_in_module, end_module,
            end_index_in_module
        """
        raise NotImplementedError

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def setup_lor_lookup_table(self) -> None:
        for mod, num_lor_endpoints in enumerate(
                self.scanner.num_lor_endpoints_per_module):
            for lor in range(num_lor_endpoints):
                tmp = self.get_modules_and_indices_in_coincidence(mod, lor)
                # make sure we do not LORs twice
                tmp = tmp[tmp[:, 0] >= mod]

                if mod == 0 and lor == 0:
                    self._lor_start_module_index = np.repeat(np.array(
                        [[mod, lor]], dtype=np.uint16),
                                                             tmp.shape[0],
                                                             axis=0)
                    self._lor_end_module_index = tmp.copy().astype(np.uint16)
                else:
                    self._lor_start_module_index = np.vstack(
                        (self._lor_start_module_index,
                         np.repeat(np.array([[mod, lor]]),
                                   tmp.shape[0],
                                   axis=0)))
                    self._lor_end_module_index = np.vstack(
                        (self._lor_end_module_index, tmp))

    def show_all_lors_for_endpoint(self,
                                   ax: plt.Axes,
                                   module: int,
                                   index_in_module: int,
                                   lw: float = 0.2,
                                   **kwargs) -> None:

        tmp = self.get_modules_and_indices_in_coincidence(
            module, index_in_module)

        end_mod = tmp[:, 0]
        end_ind = tmp[:, 1]

        start_mod = np.full(end_mod.shape[0], module)
        start_ind = np.full(end_mod.shape[0], index_in_module)

        p1s = self.scanner.get_lor_endpoints(start_mod, start_ind)
        p2s = self.scanner.get_lor_endpoints(end_mod, end_ind)

        # get_lor_endpoints can return a numpy or cupy array
        # it scanner uses cupy arrays, we have to convert them into numpy arrays
        if self.scanner.xp.__name__ == 'cupy':
            p1s = self.scanner.xp.asnumpy(p1s)
            p2s = self.scanner.xp.asnumpy(p2s)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)

    def show_lors(self,
                  ax: plt.Axes,
                  lors: None | npt.NDArray,
                  lw: float = 0.2,
                  **kwargs) -> None:
        start_mod, start_ind, end_mod, end_ind = self.get_lor_indices(lors)
        p1s = self.scanner.get_lor_endpoints(start_mod, start_ind)
        p2s = self.scanner.get_lor_endpoints(end_mod, end_ind)

        # get_lor_endpoints can return a numpy or cupy array
        # it scanner uses cupy arrays, we have to convert them into numpy arrays
        if self.scanner.xp.__name__ == 'cupy':
            p1s = self.scanner.xp.asnumpy(p1s)
            p2s = self.scanner.xp.asnumpy(p2s)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class GenericPETCoincidenceDescriptor(PETCoincidenceDescriptor):
    """ generic coincidence logic where a LOR endpoint in a module is connected to all
        all LOR endpoints of all other modules
    """

    def __init__(self, scanner: scanners.ModularizedPETScannerGeometry):

        super().__init__(scanner)
        self.setup_lor_lookup_table()

    @property
    def num_lors(self):
        return self._lor_start_module_index.shape[0]

    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if linear_lor_indices is None:
            linear_lor_indices = np.arange(self.num_lors)

        return self._lor_start_module_index[
            linear_lor_indices], self._lor_end_module_index[linear_lor_indices]

    def get_modules_and_indices_in_coincidence(
            self, module: int, index_in_module: int) -> npt.NDArray:

        modules = []
        indices = []

        for i, num_modules in enumerate(
                self.scanner.num_lor_endpoints_per_module):

            if i != module:
                modules += num_modules * [i]
                indices += range(num_modules)

        return np.array([modules, indices]).T


class RegularPolygonPETCoincidenceDescriptor(PETCoincidenceDescriptor):

    def __init__(
        self,
        scanner: scanners.RegularPolygonPETScannerGeometry,
        radial_trim: int = 3,
        max_ring_difference: int | None = None,
        sinogram_spatial_axis_order:
        SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.PVR
    ) -> None:

        super().__init__(scanner)

        self._radial_trim = radial_trim

        if max_ring_difference is None:
            self._max_ring_difference = self.scanner.num_rings - 1
        else:
            self._max_ring_difference = max_ring_difference

        self._num_rad = (self.scanner.num_lor_endpoints_per_ring +
                         1) - 2 * self._radial_trim
        self._num_views = self.scanner.num_lor_endpoints_per_ring // 2

        self._sinogram_spatial_axis_order = sinogram_spatial_axis_order

        self.setup_plane_indices()
        self.setup_view_indices()

        if self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RVP:
            self._sinogram_spatial_shape = (self.num_rad, self.num_views,
                                            self.num_planes)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RPV:
            self._sinogram_spatial_shape = (self.num_rad, self.num_planes,
                                            self.num_views)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VRP:
            self._sinogram_spatial_shape = (self.num_views, self.num_rad,
                                            self.num_planes)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VPR:
            self._sinogram_spatial_shape = (self.num_views, self.num_planes,
                                            self.num_rad)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PVR:
            self._sinogram_spatial_shape = (self.num_planes, self.num_views,
                                            self.num_rad)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PRV:
            self._sinogram_spatial_shape = (self.num_planes, self.num_rad,
                                            self.num_views)

    @property
    def radial_trim(self) -> int:
        return self._radial_trim

    @property
    def max_ring_difference(self) -> int:
        return self._max_ring_difference

    @property
    def num_planes(self) -> int:
        return self._num_planes

    @property
    def num_rad(self) -> int:
        return self._num_rad

    @property
    def num_views(self) -> int:
        return self._num_views

    @property
    def num_lors(self) -> int:
        return self.num_rad * self.num_views * self.num_planes

    @property
    def start_plane_index(self) -> npt.NDArray:
        return self._start_plane_index

    @property
    def end_plane_index(self) -> npt.NDArray:
        return self._end_plane_index

    @property
    def start_in_ring_index(self) -> npt.NDArray:
        return self._start_in_ring_index

    @property
    def end_in_ring_index(self) -> npt.NDArray:
        return self._end_in_ring_index

    @property
    def sinogram_spatial_axis_order(self) -> SinogramSpatialAxisOrder:
        return self._sinogram_spatial_axis_order

    @property
    def sinogram_spatial_shape(self) -> tuple[int, int, int]:
        return self._sinogram_spatial_shape

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods from the base class that we have to implement

    def get_modules_and_indices_in_coincidence(
            self, module: int, index_in_module: int) -> npt.NDArray:
        return np.array(
            list(
                itertools.product(
                    self.get_modules_in_coincidence(module),
                    self.get_indices_in_module_in_coincidence(
                        index_in_module))))

    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

        if linear_lor_indices is None:
            linear_lor_indices = np.arange(self.num_lors)

        if self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RVP:
            radial_elements, views, planes = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RPV:
            radial_elements, planes, views = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VRP:
            views, radial_elements, planes = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VPR:
            views, planes, radial_elements = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PVR:
            planes, views, radial_elements = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)
        elif self.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PRV:
            planes, radial_elements, views = np.unravel_index(
                linear_lor_indices, self.sinogram_spatial_shape)

        start_ring = self.start_plane_index[planes]
        end_ring = self.end_plane_index[planes]

        start_inds = self.start_in_ring_index[views, radial_elements]
        end_inds = self.end_in_ring_index[views, radial_elements]

        return start_ring, start_inds, end_ring, end_inds

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def get_modules_in_coincidence(self, module: int) -> npt.NDArray:

        ring_numbers = np.arange(self.scanner.num_rings)
        i1 = np.abs(ring_numbers - module) <= self.max_ring_difference

        return ring_numbers[i1]

    def get_indices_in_module_in_coincidence(
            self, index_in_module: int) -> npt.NDArray:

        tmp0 = self.end_in_ring_index[self.start_in_ring_index ==
                                      index_in_module]
        tmp1 = self.start_in_ring_index[self.end_in_ring_index ==
                                        index_in_module]

        indices_in_coinc = np.concatenate((tmp0, tmp1))
        indices_in_coinc.sort()

        return indices_in_coinc

    def setup_plane_indices(self) -> None:
        self._start_plane_index = np.arange(self.scanner.num_rings)
        self._end_plane_index = np.arange(self.scanner.num_rings)

        for i in range(1, self.max_ring_difference + 1):
            tmp1 = np.arange(self.scanner.num_rings - i)
            tmp2 = np.arange(self.scanner.num_rings - i) + i

            self._start_plane_index = np.concatenate(
                (self._start_plane_index, tmp1, tmp2))
            self._end_plane_index = np.concatenate(
                (self._end_plane_index, tmp2, tmp1))

        self._num_planes = self._start_plane_index.shape[0]

    def setup_view_indices(self) -> None:
        n = self.scanner.num_lor_endpoints_per_ring

        self._start_in_ring_index = np.zeros((self.num_views, self.num_rad),
                                             dtype=np.int16)
        self._end_in_ring_index = np.zeros((self.num_views, self.num_rad),
                                           dtype=np.int16)

        for view in np.arange(self.num_views):
            self._start_in_ring_index[view, :] = (
                np.concatenate((np.repeat(np.arange(n // 2), 2), [n // 2])) -
                view)[self.radial_trim:-self.radial_trim]
            self._end_in_ring_index[view, :] = (
                np.concatenate(([-1], np.repeat(-np.arange(n // 2) - 2, 2))) -
                view)[self.radial_trim:-self.radial_trim]

        # shift the negative indices
        neg_inds_start = np.where(self._start_in_ring_index < 0)
        neg_inds_end = np.where(self._end_in_ring_index < 0)

        self._start_in_ring_index[
            neg_inds_start] = n + self._start_in_ring_index[neg_inds_start]
        self._end_in_ring_index[
            neg_inds_end] = n + self._end_in_ring_index[neg_inds_end]

    def show_view(self,
                  ax: plt.Axes,
                  view: int,
                  plane: int,
                  lw: float = 0.2,
                  **kwargs) -> None:

        start_ring = self.start_plane_index[plane]
        end_ring = self.end_plane_index[plane]

        start_inds = self.start_in_ring_index[view, :]
        end_inds = self.end_in_ring_index[view, :]

        p1s = self.scanner.get_lor_endpoints(start_ring, start_inds)
        p2s = self.scanner.get_lor_endpoints(end_ring, end_inds)

        # get_lor_endpoints can return a numpy or cupy array
        # it scanner uses cupy arrays, we have to convert them into numpy arrays
        if self.scanner.xp.__name__ == 'cupy':
            p1s = self.scanner.xp.asnumpy(p1s)
            p2s = self.scanner.xp.asnumpy(p2s)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class LORSubsetter(abc.ABC):

    def __init__(self, num_lors: int) -> None:
        self._num_lors = num_lors

    @property
    @abc.abstractmethod
    def num_subsets(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_subset_indices(self, subset: int) -> npt.NDArray:
        raise NotImplementedError

    @property
    def num_lors(self) -> int:
        return self._num_lors


class RandomLORSubsetter(LORSubsetter):

    def __init__(self, num_lors: int, num_subsets: int) -> None:

        super().__init__(num_lors)

        self._num_subsets = num_subsets
        self._all_lor_indices = np.arange(self.num_lors)
        self.shuffle()
        self._all_lor_subset_indices = np.array_split(self._all_lor_indices,
                                                      self._num_subsets)

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    @property
    def all_lor_indices(self) -> npt.NDArray:
        return self._all_lor_indices

    @property
    def all_lor_subset_indices(self) -> list[npt.NDArray]:
        return self._all_lor_subset_indices

    def get_subset_indices(self, subset: int) -> npt.NDArray:
        if subset >= self.num_subsets:
            raise ValueError(f'subset must be < {self.num_subsets}')

        return self._all_lor_subset_indices[subset]

    def shuffle(self) -> None:
        np.random.shuffle(self._all_lor_indices)


class SingoramViewSubsetter(LORSubsetter):

    def __init__(
            self,
            coincidence_descriptor: RegularPolygonPETCoincidenceDescriptor,
            num_subsets: int) -> None:

        self._num_subsets = num_subsets
        self._coincidence_descriptor = coincidence_descriptor
        super().__init__(self._coincidence_descriptor.num_lors)

        if self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RVP:
            self._view_axis = 1
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.RPV:
            self._view_axis = 2
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VRP:
            self._view_axis = 0
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.VPR:
            self._view_axis = 0
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PVR:
            self._view_axis = 1
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is SinogramSpatialAxisOrder.PRV:
            self._view_axis = 2

        dtype = np.uint64

        if self._coincidence_descriptor.num_lors <= 2**16:
            dtype = np.uint16
        elif self._coincidence_descriptor.num_lors <= 2**32:
            dtype = np.uint32

        all_lor_indices = np.arange(self.num_lors, dtype=dtype).reshape(
            self._coincidence_descriptor.sinogram_spatial_shape)

        self._start_views = np.arange(self.num_subsets)
        self._start_views[0::2] = np.arange(
            self.num_subsets)[(self.num_subsets // 2):]
        self._start_views[1::2] = np.arange(
            self.num_subsets)[:(self.num_subsets // 2)]

        self._subset_views = []
        for i, start_view in enumerate(self._start_views):
            self._subset_views.append(
                np.arange(start_view, self._coincidence_descriptor.num_views,
                          self.num_subsets))

        self._all_lor_subset_indices = []

        for views in self._subset_views:
            if self._view_axis == 0:
                self._all_lor_subset_indices.append(
                    all_lor_indices[views, :, :].ravel())
            elif self._view_axis == 1:
                self._all_lor_subset_indices.append(
                    all_lor_indices[:, views, :].ravel())
            elif self._view_axis == 2:
                self._all_lor_subset_indices.append(
                    all_lor_indices[:, :, views].ravel())

        del all_lor_indices

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    def get_subset_indices(self, subset: int) -> npt.NDArray:
        return self._all_lor_subset_indices[subset]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    num_rings = 3

    sc = scanners.RegularPolygonPETScannerGeometry(350., 28, 16, 4., num_rings,
                                                   4 * np.arange(num_rings), 0)

    cd = RegularPolygonPETCoincidenceDescriptor(
        sc, sinogram_spatial_axis_order=SinogramSpatialAxisOrder.VRP)
    ss = SingoramViewSubsetter(cd, 3)

    if num_rings < 4:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        sc.show_lor_endpoints(ax1)
        sc.show_lor_endpoints(ax2)
        cd.show_lors(ax1, lors=None, color='b')
        cd.show_lors(ax2, lors=ss.get_subset_indices(0), color='r')
        cd.show_lors(ax2, lors=ss.get_subset_indices(1), color='g')
        cd.show_lors(ax2, lors=ss.get_subset_indices(2), color='k')
        fig.tight_layout()
        fig.show()