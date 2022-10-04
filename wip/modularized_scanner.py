import abc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class CrystalModule(abc.ABC):

    def __init__(self,
                 num_crystals: int,
                 crystal_size=tuple[float, float, float]) -> None:
        self._num_crystals = num_crystals
        self._crystal_size = crystal_size

    @property
    def num_crystals(self) -> int:
        return self._num_crystals

    @property
    def crystal_size(self) -> tuple[float, float, float]:
        return self._crystal_size

    @abc.abstractmethod
    def get_crystal_positions(self,
                              crystal_numbers: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def show(self, **kwargs):
        crystal_coords = self.get_crystal_positions(
            np.arange(self.num_crystals))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(crystal_coords[:, 0], crystal_coords[:, 1],
                   crystal_coords[:, 2], **kwargs)
        eps = 1e-7
        ax.set_box_aspect(
            (max(np.ptp(crystal_coords[:, 0]),
                 eps), max(np.ptp(crystal_coords[:, 1]),
                           eps), max(np.ptp(crystal_coords[:, 2]), eps)))
        fig.show()

        return fig


class RectangularCrystalModule(CrystalModule):

    def __init__(self, n: tuple[int, int], crystal_size: tuple[float, float,
                                                               float]) -> None:

        self._n = n
        super().__init__(n[0] * n[1], crystal_size)

    @property
    def n(self) -> tuple[int, int]:
        return self._n

    def get_crystal_positions(self,
                              crystal_numbers: npt.NDArray) -> npt.NDArray:
        crystal_positions = np.zeros((crystal_numbers.shape[0], 3))

        crystal_positions[:, 0] = (crystal_numbers %
                                   self.n[0]) * self.crystal_size[0]
        crystal_positions[:, 1] = (crystal_numbers //
                                   self.n[0]) * self.crystal_size[1]

        return crystal_positions


class RegularPolygonCrystalModule(CrystalModule):

    def __init__(self,
                 R: float,
                 num_sides: int,
                 num_crystals_per_side: int,
                 crystal_size=tuple[float, float, float]) -> None:

        self._R = R
        self._num_sides = num_sides
        self._num_crystals_per_side = num_crystals_per_side
        super().__init__(num_sides * num_crystals_per_side, crystal_size)

    @property
    def R(self) -> float:
        return self._R

    @property
    def num_sides(self) -> int:
        return self._num_sides

    @property
    def num_crystals_per_side(self) -> int:
        return self._num_crystals_per_side

    def get_crystal_positions(self,
                              crystal_numbers: npt.NDArray) -> npt.NDArray:
        side = crystal_numbers // self.num_crystals_per_side
        tmp = crystal_numbers - side * self.num_crystals_per_side
        tmp = tmp - (self.num_crystals_per_side / 2 - 0.5)

        phi = 2 * np.pi * side / self.num_sides

        crystal_coordinates = np.zeros((crystal_numbers.shape[0], 3))
        crystal_coordinates[:, 0] = np.cos(phi) * self.R - np.sin(
            phi) * self.crystal_size[0] * tmp
        crystal_coordinates[:, 1] = np.sin(phi) * self.R + np.cos(
            phi) * self.crystal_size[0] * tmp

        return crystal_coordinates
