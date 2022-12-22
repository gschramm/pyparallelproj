"""description of (PET) scanner geometries consisting of LOR endpoint modules"""
import types

import numpy as np
import numpy.typing as npt
try:
    import cupy.typing as cpt
except:
    import numpy.typing as cpt

import matplotlib.pyplot as plt

from . import scannermodules as mods


class ModularizedPETScannerGeometry:
    """description of a PET scanner geometry consisting of LOR endpoint modules"""

    def __init__(self,
                 modules: tuple[mods.PETScannerModule],
                 xp: types.ModuleType | None = None) -> None:
        """
        Parameters
        ----------
        modules : tuple[PETScannerModule]
            a tuple of scanner modules
        xp : types.ModuleType | None, optional default None
            module indicating whether to store all LOR endpoints as numpy as cupy array
            default None means that numpy is used
        """

        # member variable that determines whether we want to use
        # a numpy or cupy array to store the array of all lor endpoints
        if xp is None:
            self._xp = np
        else:
            self._xp = xp

        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = self._xp.array(
            [x.num_lor_endpoints for x in self._modules])
        self._num_lor_endpoints = self._num_lor_endpoints_per_module.sum()
        if self._xp.__name__ not in ['numpy', 'cupy']:
            raise ValueError('xp must be numpy or cupy module')

        self.setup_all_lor_endpoints()

    def setup_all_lor_endpoints(self) -> None:
        """calculate the position of all lor endpoints by iterating over
           the modules and calculating the transformed coordinates of all
           module endpoints
        """
        self._all_lor_endpoints_index_offset = self._xp.cumsum(
            self._xp.pad(self._num_lor_endpoints_per_module,
                         (1, 0)))[:self._num_modules]

        self._all_lor_endpoints = self._xp.vstack(
            [x.get_lor_endpoints() for x in self._modules])

        self._all_lor_endpoints_module_number = self._xp.repeat(
            self._xp.arange(self._num_modules),
            self._num_lor_endpoints_per_module.tolist())

    @property
    def modules(self) -> tuple[mods.PETScannerModule]:
        """tuple of modules defining the scanner"""
        return self._modules

    @property
    def num_modules(self) -> int:
        """the number of modules defining the scanner"""
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> npt.NDArray | cpt.NDArray:
        """numpy array showing how many LOR endpoints are in every module"""
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        """the total number of LOR endpoints in the scanner"""
        return self._num_lor_endpoints

    @property
    def all_lor_endpoints_index_offset(self) -> npt.NDArray | cpt.NDArray:
        """the offset in the linear (flattend) index for all LOR endpoints"""
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> npt.NDArray | cpt.NDArray:
        """the module number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> npt.NDArray | cpt.NDArray:
        """the world coordinates of all LOR endpoints"""
        return self._all_lor_endpoints

    @property
    def xp(self) -> types.ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._xp

    @xp.setter
    def xp(self, value: types.ModuleType):
        """set the module to use for storing all LOR endpoints"""
        self._xp = value
        self.setup_all_lor_endpoints()

    def linear_lor_endpoint_index(
        self, module: npt.NDArray | cpt.NDArray,
        index_in_module: npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index

        Parameters
        ----------
        module : npt.NDArray
            containing module numbers
        index_in_module : npt.NDArray
            containing index in modules

        Returns
        -------
        npt.NDArray
            the flattened LOR endpoint index
        """
        if (self._xp.__name__ == 'cupy') and isinstance(
                index_in_module, np.ndarray):
            index_in_module = self._xp.asarray(index_in_module)

        return self.all_lor_endpoints_index_offset[module] + index_in_module

    def get_lor_endpoints(
            self, module: npt.NDArray,
            index_in_module: npt.NDArray) -> npt.NDArray | cpt.NDArray:
        """get the coordinates for LOR endpoints defined by module and index in module

        Parameters
        ----------
        module : npt.NDArray
            the module number of the LOR endpoints
        index_in_module : npt.NDArray
            the index in module number of the LOR endpoints

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the 3 world coordinates of the LOR endpoints
        """
        return self.all_lor_endpoints[
            self.linear_lor_endpoint_index(module, index_in_module), :]

    def show_lor_endpoints(self,
                           ax: plt.Axes,
                           show_linear_index: bool = True,
                           **kwargs) -> None:
        """show all LOR endpoints in a 3D plot

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        show_linear_index : bool, optional
            annotate the LOR endpoints with the linear LOR endpoint index
        **kwargs : keyword arguments
            passed to show_lor_endpoints() of the scanner module
        """
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
    """description of a PET scanner geometry consisting stacked regular polygons"""

    def __init__(self,
                 radius: float,
                 num_sides: int,
                 num_lor_endpoints_per_side: int,
                 lor_spacing: float,
                 num_rings: int,
                 ring_positions: npt.NDArray,
                 symmetry_axis: int,
                 xp: types.ModuleType | None = None) -> None:
        """
        Parameters
        ----------
        radius : float
            radius of the scanner
        num_sides : int
            number of sides (faces) of each regular polygon
        num_lor_endpoints_per_side : int
            number of LOR endpoints in each side (face) of each polygon
        lor_spacing : float
            spacing between the LOR endpoints in each side
        num_rings : int
            the number of rings (regular polygons)
        ring_positions : npt.NDArray
            1D array with the coordinate of the rings along the ring axis
        symmetry_axis : int
            the ring axis (0,1,2)
        xp : types.ModuleType | None, optional default None
            numpy or cupy module used to store the coordinates of all LOR endpoints, by default np
            None means that numpy is used
        """

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
                mods.RegularPolygonPETScannerModule(
                    radius,
                    num_sides,
                    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                    lor_spacing=lor_spacing,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1))

        modules = tuple(modules)
        super().__init__(modules, xp)

        self._all_lor_endpoints_index_in_ring = self._xp.arange(
            self.num_lor_endpoints
        ) - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[
            0]

    @property
    def radius(self) -> float:
        """radius of the scanner"""
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides (faces) of each polygon"""
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side (face) in each polygon"""
        return self._num_lor_endpoints_per_side

    @property
    def num_rings(self) -> int:
        """number of rings (regular polygons)"""
        return self._num_rings

    @property
    def lor_spacing(self) -> float:
        """the spacing between the LOR endpoints in every side (face) of each polygon"""
        return self._lor_spacing

    @property
    def symmetry_axis(self) -> int:
        """The symmetry axis. Also called axial (or ring) direction."""
        return self._symmetry_axis

    @property
    def all_lor_endpoints_ring_number(self) -> npt.NDArray:
        """the ring (regular polygon) number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> npt.NDArray:
        """the index withing the ring (regular polygon) number of all LOR endpoints"""
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        """the number of LOR endpoints per ring (regular polygon)"""
        return int(self._num_lor_endpoints_per_module[0])


class GEDiscoveryMI(RegularPolygonPETScannerGeometry):

    def __init__(self,
                 num_rings: int = 36,
                 symmetry_axis: int = 2,
                 xp: types.ModuleType = np):

        ring_positions = 5.31556 * np.arange(num_rings) + (
            np.arange(num_rings) // 9) * 2.8
        ring_positions -= 0.5 * ring_positions.max()
        super().__init__(radius=0.5 * (744.1 + 2 * 8.51),
                         num_sides=34,
                         num_lor_endpoints_per_side=16,
                         lor_spacing=4.03125,
                         num_rings=num_rings,
                         ring_positions=ring_positions,
                         symmetry_axis=symmetry_axis,
                         xp=xp)
