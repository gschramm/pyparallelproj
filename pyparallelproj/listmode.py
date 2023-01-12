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
    @abc.abstractmethod
    def get_event_lor_start_coordinates(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        """ get the start coordinates of the LORs for LM events of a subset"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_event_lor_end_coordinates(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        """ get the start coordinates of the LORs for LM events of a subset"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_event_tof_bins(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        """ get the TOF bins for LM events of a subset"""
        raise NotImplementedError

class GenericListmodeEvents(ListmodeEvents):
    def __init__(self, 
                 events: npt.NDArray | cpt.NDArray, 
                 scanner: scanners.ModularizedPETScannerGeometry, 
                 precalculate_coords : bool = True) -> None:

        self._events = events
        self._scanner = scanner
        self._precalculate_coords = precalculate_coords

        if self.precalculate_coords:
            self._event_lor_start_coordinates = self.scanner.get_lor_endpoints(
                events[:, 0],
                events[:, 1]).astype(self.scanner.xp.float32)
        else:
            self._event_lor_start_coordinates = None


        if self.precalculate_coords:
            self._event_lor_end_coordinates = self.scanner.get_lor_endpoints(
                events[:, 2],
                events[:, 3]).astype(self.scanner.xp.float32)
        else:
            self._event_lor_end_coordinates = None

        self._event_tof_bins = events[:, 4].astype(self.scanner.xp.int16)

    @property
    def scanner(self) -> scanners.ModularizedPETScannerGeometry:
        return self._scanner

    @property
    def events(self) -> npt.NDArray | cpt.NDArray:
        return self._events

    @property
    def precalculate_coords(self) -> bool:
        return self._precalculate_coords

    def get_event_lor_start_coordinates(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        if self.precalculate_coords:
            return self._event_lor_start_coordinates[subset_inds]
        else:
            return self.scanner.get_lor_endpoints(
                self.events[subset_inds, 0],
                self.events[subset_inds, 1]).astype(self.scanner.xp.float32)

    def get_event_lor_end_coordinates(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        if self.precalculate_coords:
            return self._event_lor_end_coordinates[subset_inds]
        else:
            return self.scanner.get_lor_endpoints(
                self.events[subset_inds, 2],
                self.events[subset_inds, 3]).astype(self.scanner.xp.float32)

    def get_event_tof_bins(self, subset_inds: None | slice | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        if subset_inds is None:
            subset_inds = slice(None)

        return self._event_tof_bins[subset_inds]