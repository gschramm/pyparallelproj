"""description of (PET) scanner modules containing LOR endpoints"""
import abc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


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
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> npt.NDArray:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        npt.NDArray
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> npt.NDArray:
        """4x4 affine transformation matrix

        Returns
        -------
        npt.NDArray
        """
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
        """number of LOR endpoints in the two direction of the module

        Returns
        -------
        tuple[int, int]
        """
        return self._n

    @property
    def ax0(self) -> int:
        """axis number for the first direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> tuple[float, float]:
        """spacing between the LOR endpoints in the two directions

        Returns
        -------
        tuple[float, float]
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
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
        """number of LOR endpoints in the two direction of the module

        Returns
        -------
        tuple[int, int]
        """
        return self._n

    @property
    def ax0(self) -> int:
        """axis number for the first direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> tuple[float, float]:
        """spacing between the LOR endpoints in the two directions

        Returns
        -------
        tuple[float, float]
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
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
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float:
        """spacing between the LOR endpoints in a module along the polygon

        Returns
        -------
        float
        """
        return self._lor_spacing

    # abstract method from base class to be implemented
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
