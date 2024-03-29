"""module for defining classes related to data subsetting"""
import abc
import math

import numpy as np
import numpy.typing as npt

try:
    import cupy.typing as cpt
except:
    import numpy.typing as cpt

from . import coincidences


class Subsetter(abc.ABC):

    # abstract methods
    @property
    @abc.abstractmethod
    def num_subsets(self) -> int:
        """the number of defined subsets"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_subset_indices(self, subset: int) -> npt.NDArray | slice:
        """get the indices along the subset axis (left most axis) belonging to a given subset

        Parameters
        ----------
        subset : int
            the subset number

        Returns
        -------
        npt.NDArray | slice
            of indices
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_subset_index_len(self, subset: int) -> int:
        """get the shape of the subset indices array

        Parameters
        ----------
        subset : int
            the subset number

        Returns
        -------
        int
            length of the subset index array
        """
        raise NotImplementedError


class Strided1DSubsetter(Subsetter):

    def __init__(self, num_elements: int, num_subsets: int) -> None:
        """split a 1D array into num_subsets into n strided subsets

        Parameters
        ----------
        num_elements : int
            total size of the 1D array
        num_subsets : int
            number of subsets
        """

        self._num_elements = num_elements
        self._num_subsets = num_subsets

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    @num_subsets.setter
    def num_subsets(self, value: int) -> None:
        self._num_subsets = value

    @property
    def num_elements(self) -> int:
        return self._num_elements

    @num_elements.setter
    def num_elements(self, value: int) -> None:
        self._num_elements = value

    def get_subset_indices(self, subset: int) -> npt.NDArray | slice:
        return slice(subset, None, self.num_subsets)

    def get_subset_index_len(self, subset: int) -> int:
        return math.ceil((self.num_elements - subset) / self.num_subsets)


class RandomLORSubsetter(Subsetter):

    def __init__(self, num_lors: int, num_subsets: int) -> None:
        """split a set of geometrical LORs into random subsets

        Parameters
        ----------
        num_lors : int
            the total number of geometrical LORs
        num_subsets : int
            the total number of subsets
        """

        self._num_lors = num_lors
        self._num_subsets = num_subsets
        self._all_lor_indices = np.arange(self.num_lors)
        self.shuffle()
        self._all_lor_subset_indices = np.array_split(self._all_lor_indices,
                                                      self._num_subsets)

    @property
    def num_lors(self) -> int:
        return self._num_lors

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

    def get_subset_index_len(self, subset: int) -> int:
        return self._all_lor_subset_indices[subset].shape[0]

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    def shuffle(self) -> None:
        """shuffle the the LOR distribution across subsets"""
        np.random.shuffle(self._all_lor_indices)


class SingoramViewSubsetter(Subsetter):

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
        self._num_lors = self._coincidence_descriptor.num_lors

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

    @property
    def num_lors(self) -> int:
        return self._num_lors

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

    def get_subset_index_len(self, subset: int) -> int:
        return self._all_lor_subset_indices[subset].shape[0]

    def get_sinogram_subset_shape(self, subset: int) -> tuple[int, int, int]:
        tmp = list(self._coincidence_descriptor.sinogram_spatial_shape)
        tmp[self._view_axis] = self._subset_views[subset].shape[0]

        return tuple(tmp)


def split_subset_data(data: npt.NDArray | cpt.NDArray,
                      subsetter: Subsetter) -> npt.NDArray | cpt.NDArray:
    """split a subset data array into a "chunked" array that facilitates access
       to the subsets

    Parameters
    ----------
    data : npt.NDArray | cpt.NDArray
        an array
    subsetter : Subsetter
        defining substes along the first axis of the array
        the size of all the subsets must be the same

    Returns
    -------
    npt.NDArray | cpt.NDArray
        reshaped "chunked" array
        if the input array has shape (50,27) and we have 5 subsets,
        the reshaped array has shape (5,10,27) nad [i,:,:] contains
        the data of the first subset
    """
    subset_ind_len = [
        subsetter.get_subset_index_len(i) for i in range(subsetter.num_subsets)
    ]

    if len(set(subset_ind_len)) > 1:
        raise ValueError("Subsets must have equal length.")

    subset_len = subset_ind_len[0]

    if data.ndim > 1:
        new_shape = (subsetter.num_subsets, subset_len) + data.shape[1:]
    else:
        new_shape = (subsetter.num_subsets, subset_len)

    reshaped_data = 0 * data.reshape(new_shape)

    for i in range(subsetter.num_subsets):
        subset_inds = subsetter.get_subset_indices(i)
        reshaped_data[i, ...] = data[subset_inds, ...]

    return reshaped_data


def merge_subset_data(chunked_data: npt.NDArray | cpt.NDArray,
                      subsetter: Subsetter) -> npt.NDArray | cpt.NDArray:
    """merged data of a chunked data array into a subset array

    Parameters
    ----------
    chunked_data : npt.NDArray | cpt.NDArray
        an array split into chunks
    subsetter : Subsetter
        defining substes along the first axis of the array
        the size of all the subsets must be the same

    Returns
    -------
    npt.NDArray | cpt.NDArray
        reshaped array
        if the chunked input array has shape (5,10,27) (5 subsets)
        the reshaped array has shape (50,27)
    """

    if chunked_data.ndim > 2:
        new_shape = (chunked_data.shape[0] *
                     chunked_data.shape[1], ) + chunked_data.shape[2:]
    else:
        new_shape = (chunked_data.shape[0] * chunked_data.shape[1], )

    reshaped_data = 0 * chunked_data.reshape(new_shape)

    for i in range(subsetter.num_subsets):
        subset_inds = subsetter.get_subset_indices(i)
        reshaped_data[subset_inds] = chunked_data[i, ...]

    return reshaped_data