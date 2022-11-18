"""module for defining classes related to data subsetting"""
import abc

import numpy as np
import numpy.typing as npt

import pyparallelproj.coincidences as coincidences


class LORSubsetter(abc.ABC):

    def __init__(self, num_lors: int) -> None:
        """abstract base class for LORSubsetter

        Parameters
        ----------
        num_lors : int
            total number of geometrical LORs
        """
        self._num_lors = num_lors

    @property
    def num_lors(self) -> int:
        """the total number of geometrical LORs"""
        return self._num_lors

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods
    @property
    @abc.abstractmethod
    def num_subsets(self) -> int:
        """the number of defined subsets"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_subset_indices(self, subset: int) -> npt.NDArray:
        """get the flattened indices of all geometrical LORs in a given subset

        Parameters
        ----------
        subset : int
            the subset number

        Returns
        -------
        npt.NDArray
            the flattened indices of all geometrical LORs
        """
        raise NotImplementedError

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------


class RandomLORSubsetter(LORSubsetter):

    def __init__(self, num_lors: int, num_subsets: int) -> None:
        """split a set of geometrical LORs into random subsets

        Parameters
        ----------
        num_lors : int
            the total number of geometrical LORs
        num_subsets : int
            the total number of subsets
        """
        super().__init__(num_lors)

        self._num_subsets = num_subsets
        self._all_lor_indices = np.arange(self.num_lors)
        self.shuffle()
        self._all_lor_subset_indices = np.array_split(self._all_lor_indices,
                                                      self._num_subsets)

    @property
    def all_lor_indices(self) -> npt.NDArray:
        """return all geometrical LORs indices"""
        return self._all_lor_indices

    @property
    def all_lor_subset_indices(self) -> list[npt.NDArray]:
        """return a list of all geometrical LORs indices within the subsets"""
        return self._all_lor_subset_indices

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods we have to implement
    def get_subset_indices(self, subset: int) -> npt.NDArray:
        if subset >= self.num_subsets:
            raise ValueError(f'subset must be < {self.num_subsets}')

        return self._all_lor_subset_indices[subset]

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def shuffle(self) -> None:
        """shuffle the the LOR distribution across subsets"""
        np.random.shuffle(self._all_lor_indices)


class SingoramViewSubsetter(LORSubsetter):

    def __init__(self, coincidence_descriptor: coincidences.
                 RegularPolygonPETCoincidenceDescriptor,
                 num_subsets: int) -> None:
        """view-based subsetter for regular polygon PET scanner coincidences

        Parameters
        ----------
        coincidence_descriptor : RegularPolygonPETCoincidenceDescriptor
            coincidence descriptor of a regular polygon PET
        num_subsets : int
            the number of subsets
        """

        self._num_subsets = num_subsets
        self._coincidence_descriptor = coincidence_descriptor
        super().__init__(self._coincidence_descriptor.num_lors)

        if self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.RVP:
            self._view_axis = 1
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.RPV:
            self._view_axis = 2
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.VRP:
            self._view_axis = 0
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.VPR:
            self._view_axis = 0
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.PVR:
            self._view_axis = 1
        elif self._coincidence_descriptor.sinogram_spatial_axis_order is coincidences.SinogramSpatialAxisOrder.PRV:
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

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # abstract methods we have to implement
    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    def get_subset_indices(self, subset: int) -> npt.NDArray:
        return self._all_lor_subset_indices[subset]

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def get_subset_shape(self, subset: int) -> tuple[int]:
        return self._all_lor_subset_indices[subset].shape

    def get_sinogram_subset_shape(self, subset: int) -> tuple[int, int, int]:
        tmp = list(self._coincidence_descriptor.sinogram_spatial_shape)
        tmp[self._view_axis] = self._subset_views[subset].shape[0]

        return tuple(tmp)
