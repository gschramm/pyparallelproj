import abc
import enum
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class PETScannerModule(abc.ABC):

    def __init__(
            self,
            num_lor_endpoints: int,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """abstract base class for PET scanner module

        Parameters
        ----------
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = np.arange(num_lor_endpoints)

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = np.eye(4)
        else:
            self._affine_transformation_matrix = affine_transformation_matrix

    @property
    def num_lor_endpoints(self) -> int:
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> npt.NDArray:
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> npt.NDArray:
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indicies within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints
        """
        if inds is None:
            inds = self.lor_endpoint_numbers
        raise NotImplementedError

    def get_lor_endpoints(self,
                          inds: npt.NDArray | None = None) -> npt.NDArray:
        """mapping from LOR endpoint indicies within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : npt.NDArray | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        npt.NDArray
            a 3 x len(inds) float array with the world coordinates of the LOR endpoints including an affine transformation
        """

        raw_lor_endpoints = self.get_raw_lor_endpoints(inds)

        return (
            np.append(raw_lor_endpoints, np.ones((self.num_lor_endpoints, 1)),
                      1) @ self.affine_transformation_matrix.T)[:, :3]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           annotation_fontsize: float = 0,
                           annotation_prefix: str = '',
                           annotation_offset: int = 0,
                           transformed: bool = True,
                           **kwargs) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : plt.Axes
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        ax.scatter(all_lor_endpoints[:, 0], all_lor_endpoints[:, 1],
                   all_lor_endpoints[:, 2], **kwargs)

        ax.set_box_aspect([
            ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')
        ])

        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('x2')

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(all_lor_endpoints[i, 0],
                        all_lor_endpoints[i, 1],
                        all_lor_endpoints[i, 2],
                        f'{annotation_prefix}{i+annotation_offset}',
                        fontsize=annotation_fontsize)


class RectangularPETScannerModule(PETScannerModule):

    def __init__(
            self,
            n: tuple[int, int],
            lor_spacing: tuple[float, float],
            ax0: int = 0,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """rectangular PET scanner module

        Parameters
        ----------
        n : tuple[int, int]
            number of LOR endpoints in the two direction of the module
        lor_spacing : tuple[float, float]
            spacing between the LOR endpoints in the two directions
        ax0 : int, optional
            axis number for the first direction
        ax1 : int, optional
            axis number for the second direction
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """
        self._n = n
        self._ax0 = ax0
        self._ax1 = ax1
        self._lor_spacing = lor_spacing
        super().__init__(n[0] * n[1], affine_transformation_matrix)

    @property
    def n(self) -> tuple[int, int]:
        return self._n

    @property
    def ax0(self) -> int:
        return self._ax0

    @property
    def ax1(self) -> int:
        return self._ax1

    @property
    def lor_spacing(self) -> tuple[float, float]:
        return self._lor_spacing

    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        if inds is None:
            inds = self.lor_endpoint_numbers

        lor_endpoints = np.zeros((inds.shape[0], 3))

        lor_endpoints[:, self.ax0] = (inds % self.n[0] - self.n[0] / 2 +
                                      0.5) * self.lor_spacing[0]
        lor_endpoints[:, self.ax1] = (inds // self.n[0] - self.n[1] / 2 +
                                      0.5) * self.lor_spacing[1]

        return lor_endpoints


class RegularPolygonPETScannerModule(PETScannerModule):

    def __init__(
            self,
            radius: float,
            num_sides: int,
            num_lor_endpoints_per_side: int,
            lor_spacing: float,
            ax0: int = 2,
            ax1: int = 1,
            affine_transformation_matrix: npt.NDArray | None = None) -> None:
        """regular Polygon PET scanner module

        Parameters
        ----------
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_sides: int
            number of LOR endpoints per side
        lor_spacing : float
            spacing between the LOR endpoints in the polygon direction
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : npt.NDArray | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._ax0 = ax0
        self._ax1 = ax1
        self._lor_spacing = lor_spacing
        super().__init__(num_sides * num_lor_endpoints_per_side,
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

    @property
    def lor_spacing(self) -> float:
        return self._lor_spacing

    def get_raw_lor_endpoints(self,
                              inds: npt.NDArray | None = None) -> npt.NDArray:
        if inds is None:
            inds = self.lor_endpoint_numbers

        side = inds // self.num_lor_endpoints_per_side
        tmp = inds - side * self.num_lor_endpoints_per_side
        tmp = tmp - (self.num_lor_endpoints_per_side / 2 - 0.5)

        phi = 2 * np.pi * side / self.num_sides

        lor_endpoints = np.zeros((self.num_lor_endpoints, 3))
        lor_endpoints[:, self.ax0] = np.cos(phi) * self.radius - np.sin(
            phi) * self.lor_spacing * tmp
        lor_endpoints[:, self.ax1] = np.sin(phi) * self.radius + np.cos(
            phi) * self.lor_spacing * tmp

        return lor_endpoints


class ModularizedPETScanner:

    def __init__(self, modules: tuple[PETScannerModule]) -> None:
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = np.array(
            [x.num_lor_endpoints for x in self._modules])
        self._num_lor_endpoints = self._num_lor_endpoints_per_module.sum()

        self._lor_endpoint_index_offset = np.cumsum(
            np.pad(self._num_lor_endpoints_per_module,
                   (1, 0)))[:self._num_modules]

        self._all_lor_endpoints = np.vstack(
            [x.get_lor_endpoints() for x in self._modules])

        self._lor_endpoint_module_number = np.repeat(
            np.arange(self._num_modules), self._num_lor_endpoints_per_module)

        # setup the module coicidence matrix that indicates which module are in coincidence which each other
        self._module_coincidence_matrix = np.ones(
            (self._num_modules, self._num_modules), dtype=bool)
        np.fill_diagonal(self._module_coincidence_matrix, False)

    @property
    def modules(self) -> tuple[PETScannerModule]:
        return self._modules

    @property
    def num_modules(self) -> int:
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> npt.NDArray:
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        return self._num_lor_endpoints

    @property
    def lor_endpoint_index_offset(self) -> npt.NDArray:
        return self._lor_endpoint_index_offset

    @property
    def all_lor_endpoints(self) -> npt.NDArray:
        return self._all_lor_endpoints

    @property
    def lor_endpoint_module_number(self) -> npt.NDArray:
        return self._lor_endpoint_module_number

    @property
    def module_coincidence_matrix(self) -> npt.NDArray:
        return self._module_coincidence_matrix

    def set_module_coincidence(self, module1: int, module2: int,
                               value: bool) -> None:
        self._module_coincidence_matrix[module1, module2] = value
        self._module_coincidence_matrix[module2, module1] = value

    def linear_lor_endpoint_index(self, module: npt.NDArray,
                                  index_in_module: npt.NDArray) -> npt.NDArray:
        return self.lor_endpoint_index_offset[module] + index_in_module

    def get_lor_endpoints(self, module: npt.NDArray,
                          index_in_module: npt.NDArray):
        return self.all_lor_endpoints[
            self.linear_lor_endpoint_index(module, index_in_module), :]

    def get_lor_endpoints_indices_in_coicidence(
            self, module: int, index_in_module: int) -> npt.NDArray:
        return np.in1d(
            self.lor_endpoint_module_number,
            np.arange(
                self.num_modules)[self.module_coincidence_matrix[module, :]])

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           show_linear_index: bool = True,
                           **kwargs) -> None:
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = self.lor_endpoint_index_offset[i]
                prefix = f''
            else:
                offset = 0
                prefix = f'{i},'

            module.show_lor_endpoints(ax,
                                      annotation_offset=offset,
                                      annotation_prefix=prefix,
                                      **kwargs)

    def show_all_lors_for_endpoint(self,
                                   ax: plt.Axes,
                                   module: int,
                                   index_in_module: int,
                                   lw: float = 0.2,
                                   **kwargs) -> None:

        # generate all endpoints for which the given start point is in coincidence with
        coinc_inds = self.get_lor_endpoints_indices_in_coicidence(
            module, index_in_module)
        p2s = self.all_lor_endpoints[coinc_inds, :]

        start = self.all_lor_endpoints[self.linear_lor_endpoint_index(
            np.array([module]), np.array([index_in_module])), :]
        p1s = np.repeat(start, p2s.shape[0], 0)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class RegularPolygonPETScanner(ModularizedPETScanner):

    def __init__(self,
                 radius: float,
                 num_sides: int,
                 num_lor_endpoints_per_side: int,
                 lor_spacing: float,
                 num_rings: int,
                 ring_positions: npt.NDArray,
                 symmetry_axis: int = 0) -> None:

        self._radius = radius
        self._num_sides = num_sides
        self._num_lor_endpoints_per_side = num_lor_endpoints_per_side
        self._num_rings = num_rings
        self._lor_spacing = lor_spacing
        self._symmetry_axis = symmetry_axis

        if symmetry_axis == 0:
            self._ax0 = 2
            self._ax1 = 1
        elif symmetry_axis == 1:
            self._ax0 = 0
            self._ax1 = 2
        elif symmetry_axis == 2:
            self._ax0 = 1
            self._ax1 = 0

        modules = []

        for ring in range(num_rings):
            aff_mat = np.eye(4)
            aff_mat[symmetry_axis, -1] = ring_positions[ring]

            modules.append(
                RegularPolygonPETScannerModule(
                    radius,
                    num_sides,
                    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                    lor_spacing=lor_spacing,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1))

        modules = tuple(modules)
        super().__init__(modules)

        self._all_lor_endpoints_ring_number = np.arange(
            self.num_lor_endpoints) // self.num_lor_endpoints_per_module[0]

        self._all_lor_endpoints_index_in_ring = np.arange(
            self.num_lor_endpoints) % self.num_lor_endpoints_per_module[0]

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
    def num_rings(self) -> int:
        return self._num_rings

    @property
    def lor_spacing(self) -> float:
        return self._lor_spacing

    @property
    def symmetry_axis(self) -> int:
        return self._symmetry_axis