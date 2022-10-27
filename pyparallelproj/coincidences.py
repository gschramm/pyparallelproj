"""description of LOR endpoint combinations (LORs) in a scanners consisting of modules of LOR endpoints"""
import abc
import enum
import itertools

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pyparallelproj.scanners as scanners


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""

    RVP = enum.auto()
    """[radial,view,plane]"""
    RPV = enum.auto()
    """[radial,plane,view]"""
    VRP = enum.auto()
    """[view,radial,plane]"""
    VPR = enum.auto()
    """[view,plane,radial]"""
    PRV = enum.auto()
    """[plane,radial,view]"""
    PVR = enum.auto()
    """[plane,view,radial]"""


class PETCoincidenceDescriptor(abc.ABC):
    """abstract base class to describe which modules / indices in modules of a 
       modularized PET scanner are in coincidence; defining geometrical LORs"""

    def __init__(self,
                 scanner: scanners.ModularizedPETScannerGeometry) -> None:
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            a modularized PET scanner 
        """
        self._scanner = scanner

    @property
    def scanner(self) -> scanners.ModularizedPETScannerGeometry:
        """the scanner for which coincidences are described"""
        return self._scanner

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods

    @property
    @abc.abstractmethod
    def num_lors(self) -> int:
        """the total number of geometrical LORs 
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_modules_and_indices_in_coincidence(
            self, module: int, index_in_module: int) -> npt.NDArray:
        """ return (N,2) array of two integers showing which module/index_in_module combinations
            are in coincidence with the given input module / index_in_module

        Parameters
        ----------
        module : int
            the module number
        index_in_module : int
            the (LOR endpoint) index in the module

        Returns
        -------
        npt.NDArray
            (N,2) array of two integers showing which module/index_in_module
        """

    @abc.abstractmethod
    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """ mapping that maps the linear LOR index to the 4 1D arrays
            representing start_module, start_index_in_module, end_module,
            end_index_in_module

        Parameters
        ----------
        linear_lor_indices : None | npt.NDArray, optional
            containing the linear (flattened) indices of geometrical LORs, by default None

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
            start_module, start_index_in_module, end_module, end_index_in_module
        """

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def setup_lor_lookup_table(self) -> None:
        """setup a lookup table for the start and end modules / indecies in module for all 
           geometrical LORs
        """
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
        """show all geometrical LORs for a given LOR endpoint

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        module : int
            the module number
        index_in_module : int
            the index in the module
        lw : float, optional
            line width, by default 0.2
        **kwargs : 
            keyword arguments passed to Line3DCollection
        """

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
        """show a given set of LORs

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        lors : None | npt.NDArray
            the linear (flattened) index of the geometrical LORs to show
            None means all geometrical LORs are shown
        lw : float, optional
            linewidth, by default 0.2
        **kwargs : 
            keyword arguments passed to Line3DCollection
        """
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
    """ Generic coincidence logic where a LOR endpoint in a module is connected to all
        all LOR endpoints of all other modules.
        The endpoints module / index numbers are stored in a lookup table which can be slow
    """

    def __init__(self, scanner: scanners.ModularizedPETScannerGeometry):
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            modularized scanner
        """
        super().__init__(scanner)
        self.setup_lor_lookup_table()

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods to be implemented

    @property
    def num_lors(self):
        return self._lor_start_module_index.shape[0]

    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        if linear_lor_indices is None:
            linear_lor_indices = np.arange(self.num_lors)

        start_inds = self._lor_start_module_index[linear_lor_indices]
        end_inds = self._lor_end_module_index[linear_lor_indices]

        return start_inds[:, 0], start_inds[:, 1], end_inds[:, 0], end_inds[:,
                                                                            1]

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

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------


class RegularPolygonPETCoincidenceDescriptor(PETCoincidenceDescriptor):

    def __init__(
        self,
        scanner: scanners.RegularPolygonPETScannerGeometry,
        radial_trim: int = 3,
        max_ring_difference: int | None = None,
        sinogram_spatial_axis_order:
        SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.PVR
    ) -> None:
        """Coincidence descriptor for a regular polygon PET scanner where
           we have coincidences within and between "rings (polygons of modules)" 
           The geometrical LORs can be sorted into a sinogram having a
           "plane", "view" and "radial" axis.

        Parameters
        ----------
        scanner : RegularPolygonPETScannerGeometry
            a regular polygon PET scanner
        radial_trim : int, optional
            number of geometrial LORs to disregard in the radial direction, by default 3
        max_ring_difference : int | None, optional
            maximim ring difference to consider for coincidences, by default None means
            all ring differences are included
        sinogram_spatial_axis_order : SinogramSpatialAxisOrder, optional
            order of the spatial axis in the sinogram, by default SinogramSpatialAxisOrder.PVR
            which means "planes", "views", "radial"
        """

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
        """number of geometrial LORs to disregard in the radial direction"""
        return self._radial_trim

    @property
    def max_ring_difference(self) -> int:
        """the maximum ring difference"""
        return self._max_ring_difference

    @property
    def num_planes(self) -> int:
        """number of planes in the sinogram"""
        return self._num_planes

    @property
    def num_rad(self) -> int:
        """number of radial elements in the sinogram"""
        return self._num_rad

    @property
    def num_views(self) -> int:
        """number of views in the sinogram"""
        return self._num_views

    @property
    def start_plane_index(self) -> npt.NDArray:
        """start plane for all planes"""
        return self._start_plane_index

    @property
    def end_plane_index(self) -> npt.NDArray:
        """end plane for all planes"""
        return self._end_plane_index

    @property
    def start_in_ring_index(self) -> npt.NDArray:
        """start index within ring"""
        return self._start_in_ring_index

    @property
    def end_in_ring_index(self) -> npt.NDArray:
        """end index within ring"""
        return self._end_in_ring_index

    @property
    def sinogram_spatial_axis_order(self) -> SinogramSpatialAxisOrder:
        """spatial axis order of the sinogram"""
        return self._sinogram_spatial_axis_order

    @property
    def sinogram_spatial_shape(self) -> tuple[int, int, int]:
        """spatial shape of the sinogram"""
        return self._sinogram_spatial_shape

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods from the base class that we have to implement

    @property
    def num_lors(self) -> int:
        return self.num_rad * self.num_views * self.num_planes

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
        """return modules (rings) that are in coincidence with a given module (ring)

        Parameters
        ----------
        module : int
            the module (ring) number

        Returns
        -------
        npt.NDArray
            all modules (rings) that are in coincidence with a given module (ring)
        """

        ring_numbers = np.arange(self.scanner.num_rings)
        i1 = np.abs(ring_numbers - module) <= self.max_ring_difference

        return ring_numbers[i1]

    def get_indices_in_module_in_coincidence(
            self, index_in_module: int) -> npt.NDArray:
        """return indices (endpoints) within a ring that are in coincidence for a given endpoint

        Parameters
        ----------
        index_in_module : int
            the index of the endpoint in the module (ring)

        Returns
        -------
        npt.NDArray
            all indices (endpoint) within a ring that are in coincidence with a given endpoint
        """

        tmp0 = self.end_in_ring_index[self.start_in_ring_index ==
                                      index_in_module]
        tmp1 = self.start_in_ring_index[self.end_in_ring_index ==
                                        index_in_module]

        indices_in_coinc = np.concatenate((tmp0, tmp1))
        indices_in_coinc.sort()

        return indices_in_coinc

    def setup_plane_indices(self) -> None:
        """setup the start / end plane indices (similar to a Michelogram)
        """
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
        """setup the start / end view indices
        """
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
        """show all LORs of a single view in a given plane

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        view : int
            the view number
        plane : int
            the plane number
        lw : float, optional
            the line width, by default 0.2
        """

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


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    num_rings = 45

    sc = scanners.RegularPolygonPETScannerGeometry(350., 28, 16, 4., num_rings,
                                                   4 * np.arange(num_rings), 0)

    cd = RegularPolygonPETCoincidenceDescriptor(
        sc, sinogram_spatial_axis_order=SinogramSpatialAxisOrder.VRP)
    ss = SingoramViewSubsetter(cd, 3)

    if cd.num_lors < 7000:
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
    else:
        print(f'not plotting because of too many LORs {cd.num_lors}')