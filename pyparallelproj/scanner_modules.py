import abc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class PETScannerModule(abc.ABC):

    def __init__(
            self,
            num_lor_endpoints: int,
            lor_spacing: tuple[float, float, float],
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = np.arange(num_lor_endpoints)
        self._lor_spacing = lor_spacing

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = np.eye(4)
        else:
            self._affine_transformation_matrix = affine_transformation_matrix

        self._raw_lor_endpoints = self.get_raw_lor_endpoints()
        self._all_lor_endpoints = self.get_all_lor_endpoints()

    @property
    def num_lor_endpoints(self) -> int:
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> npt.NDArray:
        return self._lor_endpoint_numbers

    @property
    def lor_spacing(self) -> tuple[float, float, float]:
        return self._lor_spacing

    @property
    def affine_transformation_matrix(self) -> npt.NDArray:
        return self._affine_transformation_matrix

    @property
    def raw_lor_endpoints(self) -> npt.NDArray:
        return self._raw_lor_endpoints

    @property
    def all_lor_endpoints(self) -> npt.NDArray:
        return self._all_lor_endpoints

    @abc.abstractmethod
    def get_raw_lor_endpoints(self) -> npt.NDArray:
        raise NotImplementedError

    def get_all_lor_endpoints(self) -> npt.NDArray:
        return (np.append(self.raw_lor_endpoints,
                          np.ones((self.num_lor_endpoints, 1)), 1)
                @ self.affine_transformation_matrix.T)[:, :3]

    def get_lor_endpoints(self,
                          lor_endpoint_numbers: npt.NDArray) -> npt.NDArray:
        return self.all_lor_endpoints[lor_endpoint_numbers, :]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           annotation_fontsize: float = 0,
                           annotation_prefix: str = '',
                           **kwargs) -> None:

        ax.scatter(self.all_lor_endpoints[:, 0], self.all_lor_endpoints[:, 1],
                   self.all_lor_endpoints[:, 2], **kwargs)

        ax.set_box_aspect([
            ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')
        ])

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(self.all_lor_endpoints[i, 0],
                        self.all_lor_endpoints[i, 1],
                        self.all_lor_endpoints[i, 2],
                        f'{annotation_prefix}{i}',
                        fontsize=annotation_fontsize)


class RectangularPETScannerModule(PETScannerModule):

    def __init__(
            self,
            n: tuple[int, int],
            lor_spacing: tuple[float, float, float],
            ax0: int = 0,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:

        self._n = n
        self._ax0 = ax0
        self._ax1 = ax1
        super().__init__(n[0] * n[1], lor_spacing,
                         affine_transformation_matrix)

    @property
    def n(self) -> tuple[int, int]:
        return self._n

    @property
    def ax0(self) -> int:
        return self._ax0

    @property
    def ax1(self) -> int:
        return self._ax1

    def get_raw_lor_endpoints(self) -> npt.NDArray:
        lor_endpoints = np.zeros((self.num_lor_endpoints, 3))

        lor_endpoints[:,
                      self.ax0] = (self.lor_endpoint_numbers % self.n[0] -
                                   self.n[0] / 2 + 0.5) * self.lor_spacing[0]
        lor_endpoints[:,
                      self.ax1] = (self.lor_endpoint_numbers // self.n[0] -
                                   self.n[1] / 2 + 0.5) * self.lor_spacing[1]

        return lor_endpoints


class RegularPolygonPETScannerModule(PETScannerModule):

    def __init__(
            self,
            radius: float,
            num_sides: int,
            num_lor_endpoints_per_side: int,
            lor_spacing: tuple[float, float, float],
            ax0: int = 2,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._ax0 = ax0
        self._ax1 = ax1
        super().__init__(num_sides * num_lor_endpoints_per_side, lor_spacing,
                         affine_transformation_matrix)

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def num_sides(self) -> int:
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        return self._ax0

    @property
    def ax1(self) -> int:
        return self._ax1

    def get_raw_lor_endpoints(self) -> npt.NDArray:
        side = self.lor_endpoint_numbers // self.num_lor_endpoints_per_side
        tmp = self.lor_endpoint_numbers - side * self.num_lor_endpoints_per_side
        tmp = tmp - (self.num_lor_endpoints_per_side / 2 - 0.5)

        phi = 2 * np.pi * side / self.num_sides

        lor_endpoints = np.zeros((self.num_lor_endpoints, 3))
        lor_endpoints[:, self.ax0] = np.cos(phi) * self.radius - np.sin(
            phi) * self.lor_spacing[0] * tmp
        lor_endpoints[:, self.ax1] = np.sin(phi) * self.radius + np.cos(
            phi) * self.lor_spacing[0] * tmp

        return lor_endpoints
