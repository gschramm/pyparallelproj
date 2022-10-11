#TODO: - split return of get_lor_indices() in 4 1D arrays
#      - enum for sinogram axis order
import abc
import itertools
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
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

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
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

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


class RandomizedRectangularPETScannerModule(PETScannerModule):

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
            "average" spacing between the LOR endpoints in the two directions
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

        self._raw_lor_endpoints = np.zeros((n[0] * n[1], 3))
        self._raw_lor_endpoints[:, self.ax0] = (np.random.rand(
            n[0] * n[1]) - 0.5) * self.n[0] * self.lor_spacing[0]
        self._raw_lor_endpoints[:, self.ax1] = (np.random.rand(
            n[0] * n[1]) - 0.5) * self.n[1] * self.lor_spacing[1]

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

        return self._raw_lor_endpoints[inds, :]


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


class ModularizedPETScannerGeometry:

    def __init__(self, modules: tuple[PETScannerModule]) -> None:
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = np.array(
            [x.num_lor_endpoints for x in self._modules])
        self._num_lor_endpoints = self._num_lor_endpoints_per_module.sum()

        self._all_lor_endpoints_index_offset = np.cumsum(
            np.pad(self._num_lor_endpoints_per_module,
                   (1, 0)))[:self._num_modules]

        self._all_lor_endpoints = np.vstack(
            [x.get_lor_endpoints() for x in self._modules])

        self._all_lor_endpoints_module_number = np.repeat(
            np.arange(self._num_modules), self._num_lor_endpoints_per_module)

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
    def all_lor_endpoints_index_offset(self) -> npt.NDArray:
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> npt.NDArray:
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> npt.NDArray:
        return self._all_lor_endpoints

    def linear_lor_endpoint_index(self, module: npt.NDArray,
                                  index_in_module: npt.NDArray) -> npt.NDArray:
        return self.all_lor_endpoints_index_offset[module] + index_in_module

    def get_lor_endpoints(self, module: npt.NDArray,
                          index_in_module: npt.NDArray) -> npt.NDArray:
        return self.all_lor_endpoints[
            self.linear_lor_endpoint_index(module, index_in_module), :]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           show_linear_index: bool = True,
                           **kwargs) -> None:
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = self.all_lor_endpoints_index_offset[i]
                prefix = f''
            else:
                offset = 0
                prefix = f'{i},'

            module.show_lor_endpoints(ax,
                                      annotation_offset=offset,
                                      annotation_prefix=prefix,
                                      **kwargs)


class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):

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

        self._all_lor_endpoints_index_in_ring = np.arange(
            self.num_lor_endpoints
        ) - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[
            0]

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

    @property
    def all_lor_endpoints_ring_number(self) -> npt.NDArray:
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> npt.NDArray:
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        return self._num_lor_endpoints_per_module[0]


class PETCoincidenceDescriptor(abc.ABC):

    def __init__(self, scanner: ModularizedPETScannerGeometry) -> None:
        self._scanner = scanner

    @property
    def scanner(self) -> ModularizedPETScannerGeometry:
        return self._scanner

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
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """ mapping that maps the linear LOR index to the index pair (module, lor endpoint in module)
            for start and endpoint of LORs
        """
        raise NotImplementedError

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
        coinc_inds = self.scanner.linear_lor_endpoint_index(
            tmp[:, 0], tmp[:, 1])
        p2s = self.scanner.all_lor_endpoints[coinc_inds, :]

        start = self.scanner.all_lor_endpoints[
            self.scanner.linear_lor_endpoint_index(np.array(
                [module]), np.array([index_in_module])), :]
        p1s = np.repeat(start, p2s.shape[0], 0)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)

    def show_all_lors(self, ax: plt.Axes, lw: float = 0.2, **kwargs) -> None:

        start_inds, end_inds = self.get_lor_indices()

        p1s = self.scanner.get_lor_endpoints(start_inds[:, 0], start_inds[:,
                                                                          1])
        p2s = self.scanner.get_lor_endpoints(end_inds[:, 0], end_inds[:, 1])

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class GenericPETCoincidenceDescriptor(PETCoincidenceDescriptor):
    """ generic coincidence logic where a LOR endpoint in a module is connected to all
        all LOR endpoints of all other modules
    """

    def __init__(self, scanner: ModularizedPETScannerGeometry):

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

    def __init__(self,
                 scanner: RegularPolygonPETScannerGeometry,
                 radial_trim: int = 3,
                 max_ring_difference: int | None = None) -> None:

        super().__init__(scanner)

        self._radial_trim = radial_trim

        if max_ring_difference is None:
            self._max_ring_difference = self.scanner.num_rings - 1
        else:
            self._max_ring_difference = max_ring_difference

        self._num_rad = (self.scanner.num_lor_endpoints_per_ring +
                         1) - 2 * self._radial_trim
        self._num_views = self.scanner.num_lor_endpoints_per_ring // 2

        self.setup_plane_indices()
        self.setup_view_indices()

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

    def get_lor_indices(
        self,
        linear_lor_indices: None | npt.NDArray = None
    ) -> tuple[npt.NDArray, npt.NDArray]:

        if linear_lor_indices is None:
            linear_lor_indices = np.arange(self.num_lors)

        radial_elements, views, planes = np.unravel_index(
            linear_lor_indices,
            (self.num_rad, self.num_views, self.num_planes))

        start_ring = self.start_plane_index[planes]
        end_ring = self.end_plane_index[planes]

        start_inds = self.start_in_ring_index[views, radial_elements]
        end_inds = self.end_in_ring_index[views, radial_elements]

        return np.vstack((start_ring, start_inds)).T, np.vstack(
            (end_ring, end_inds)).T

    def get_modules_in_coincidence(self, module: int) -> npt.NDArray:

        ring_numbers = np.arange(self.scanner.num_rings)
        i1 = np.abs(ring_numbers - module) <= self.max_ring_difference

        return ring_numbers[i1]

    def get_indices_in_module_in_coincidence(
            self, index_in_module: int) -> npt.NDArray:

        module_indices = np.arange(self.scanner.num_lor_endpoints_per_ring)

        tmp0 = (index_in_module -
                module_indices) % self.scanner.num_lor_endpoints_per_ring
        tmp1 = (module_indices -
                index_in_module) % self.scanner.num_lor_endpoints_per_ring
        i2 = np.minimum(tmp0, tmp1) >= self.min_in_ring_difference

        return module_indices[i2]

    def get_modules_and_indices_in_coincidence(
            self, module: int, index_in_module: int) -> npt.NDArray:
        return np.array(
            list(
                itertools.product(
                    self.get_modules_in_coincidence(module),
                    self.get_indices_in_module_in_coincidence(
                        index_in_module))))

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

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)
