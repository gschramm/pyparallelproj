import abc
import numpy.typing as npt

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt

from . import scanners


class ListmodeEvents(abc.ABC):

    @property
    @abc.abstractmethod
    def num_events(self) -> int:
        raise NotImplementedError


class PETListmodeEvents(ListmodeEvents):

    @abc.abstractmethod
    def get_event_lor_start_coordinates(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        """ get the start coordinates of the LORs for LM events of a subset"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_event_lor_end_coordinates(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        """ get the start coordinates of the LORs for LM events of a subset"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_event_tof_bins(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        """ get the TOF bins for LM events of a subset"""
        raise NotImplementedError


class GenericPETListmodeEvents(PETListmodeEvents):

    def __init__(self,
                 start_module_num: npt.NDArray | cpt.NDArray,
                 start_crystal_num: npt.NDArray | cpt.NDArray,
                 end_module_num: npt.NDArray | cpt.NDArray,
                 end_crystal_num: npt.NDArray | cpt.NDArray,
                 tof_bin: npt.NDArray | cpt.NDArray,
                 scanner: scanners.ModularizedPETScannerGeometry,
                 precalculate_coords: bool = True) -> None:

        self._start_module_num = start_module_num
        self._start_crystal_num = start_crystal_num
        self._end_module_num = end_module_num
        self._end_crystal_num = end_crystal_num

        self._scanner = scanner
        self._precalculate_coords = precalculate_coords

        self._event_tof_bins = tof_bin.astype(self.scanner.xp.int16)

        if self.precalculate_coords:
            self._event_lor_start_coordinates = self.scanner.get_lor_endpoints(
                start_module_num,
                start_crystal_num).astype(self.scanner.xp.float32)
        else:
            self._event_lor_start_coordinates = None

        if self.precalculate_coords:
            self._event_lor_end_coordinates = self.scanner.get_lor_endpoints(
                end_module_num,
                end_crystal_num).astype(self.scanner.xp.float32)
        else:
            self._event_lor_end_coordinates = None

    @property
    def num_events(self) -> int:
        return self._start_module_num.shape[0]

    @property
    def scanner(self) -> scanners.ModularizedPETScannerGeometry:
        return self._scanner

    @property
    def precalculate_coords(self) -> bool:
        return self._precalculate_coords

    def get_event_lor_start_coordinates(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        if self.precalculate_coords:
            return self._event_lor_start_coordinates[subset_inds]
        else:
            return self.scanner.get_lor_endpoints(
                self._start_module_num[subset_inds],
                self._start_crystal_num[subset_inds]).astype(self.scanner.xp.float32)

    def get_event_lor_end_coordinates(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        if self.precalculate_coords:
            return self._event_lor_end_coordinates[subset_inds]
        else:
            return self.scanner.get_lor_endpoints(
                self._end_module_num[subset_inds],
                self._end_crystal_num[subset_inds]).astype(
                    self.scanner.xp.float32)

    def get_event_tof_bins(
        self,
        subset_inds: None | slice | npt.NDArray = None
    ) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        return self._event_tof_bins[subset_inds]