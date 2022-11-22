import abc
import types

import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as np
    import numpy.typing as cpt

import pyparallelproj.subsets as subsets


class LinearOperator(abc.ABC):

    def __init__(self, input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...], xp: types.ModuleType) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        input_shape : tuple
            shape of x array
        output_shape : tuple
            shape of y array
        xp : types.ModuleType
            module indicating whether to store all LOR endpoints as numpy as cupy array
        """

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._xp = xp

    @property
    def input_shape(self) -> tuple[int, ...]:
        """shape of x array

        Returns
        -------
        tuple
            shape of x array
        """
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """shape of y array

        Returns
        -------
        tuple
            shape of y array
        """
        return self._output_shape

    @property
    def xp(self) -> types.ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._xp

    @abc.abstractmethod
    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """forward step

        Parameters
        ----------
        x : npt.NDArray
            x array

        Returns
        -------
        npt.NDArray
            the linear operator applied to x
        """
        pass

    @abc.abstractmethod
    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """adjoint of forward step

        Parameters
        ----------
        y : npt.NDArray
            y array

        Returns
        -------
        npt.NDArray
            the adjoint of the linear operator applied to y
        """
        raise NotImplementedError()

    def adjointness_test(self, verbose=False, padwidth=0) -> None:
        """test if adjoint is really the adjoint of forward

        Parameters
        ----------
        verbose : bool, optional
            prnt verbose output
        """
        if padwidth > 0:
            x = self.xp.pad(
                self.xp.random.rand(*tuple(x - 2 * padwidth
                                           for x in self._input_shape)),
                padwidth).astype(self.xp.float32)
        else:
            x = self.xp.random.rand(*self._input_shape).astype(self.xp.float32)

        y = self.xp.random.rand(*self._output_shape).astype(self.xp.float32)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        a = (x_fwd * y).sum()
        b = (x * y_back).sum()

        if verbose:
            print(f'<y, A x>   {a}')
            print(f'<A^T y, x> {b}')

        assert (self.xp.isclose(a, b))

    def norm(self, num_iter=20) -> float:
        """estimate norm of operator via power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of iterations, by default 20

        Returns
        -------
        float
            the estimated norm
        """

        x = self.xp.random.rand(*self._input_shape).astype(self.xp.float32)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = self.xp.linalg.norm(x.ravel())
            x /= n

        return self.xp.sqrt(n)


class MatrixOperator(LinearOperator):

    def __init__(self, A: npt.NDArray | cpt.NDArray,
                 xp: types.ModuleType) -> None:
        super().__init__((A.shape[1], ), (A.shape[0], ), xp)
        self._A = A

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self._A @ x

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        return self._A.T @ y


class LinearSubsetOperator(LinearOperator):

    def __init__(self, input_shape: tuple[int, ...], output_shape: tuple[int,
                                                                         ...],
                 xp: types.ModuleType, subsetter: subsets.Subsetter) -> None:
        """Linear operator with subsets abstract base class that maps real array x to real array y

        Parameters
        ----------
        input_shape : tuple
            shape of x array
        output_shape : tuple
            shape of y array
        xp : types.ModuleType
            module indicating whether to store all LOR endpoints as numpy as cupy array
        subsetter: subsets.Subsetter
            subsetter defining how to split operator
        """

        super().__init__(input_shape, output_shape, xp)

        self._subsetter = subsetter

    @property
    def subsetter(self) -> subsets.Subsetter:
        return self._subsetter

    @abc.abstractmethod
    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        """get the shape of a given subset

        Parameters
        ----------
        subset : int
            the subset number

        Returns
        -------
        tuple[int, ...]
            the shape of the output subset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_subset(
            self,
            x: npt.NDArray | cpt.NDArray,
            subset: int = 0,
            inds: None | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        """evaluate the operator for a given subset

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            the input array
        subset : int, optional
            the subset number
        inds : None | npt.NDArray, optional
            instead of specifying the subset directly, the subset can
            also be specified by the subset indices

        Returns
        -------
        npt.NDArray | cpt.NDArray
            output of the linear operator for a given subset
        """

        ## you have to implement the following 2 lines
        #if inds is None:
        #    inds = self.subsetter.get_subset_indices(subset)

        raise NotImplementedError

    @abc.abstractmethod
    def adjoint_subset(
            self,
            y_subset: npt.NDArray | cpt.NDArray,
            subset: int = 0,
            inds: None | npt.NDArray = None) -> npt.NDArray | cpt.NDArray:
        """adjoint of the operator for a given subset

        Parameters
        ----------
        y_subset : npt.NDArray | cpt.NDArray
            subset of y for the evaluation
        subset : int, optional
            the subset number
        inds : None | npt.NDArray, optional
            instead of specifying the subset directly, the subset can
            also be specified by the subset indices

        Returns
        -------
        npt.NDArray | cpt.NDArray
            output of the subset adjoint
        """

        ## you have to implement the following 2 lines
        #if inds is None:
        #    inds = self.subsetter.get_subset_indices(subset)

        raise NotImplementedError

    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        x_forward = self.xp.zeros(self.output_shape, dtype=self.xp.float32)

        for subset in range(self.subsetter.num_subsets):
            inds = self.subsetter.get_subset_indices(subset)
            x_forward[inds] = self.forward_subset(x, inds=inds)

        return x_forward

    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:

        y_back = self.xp.zeros(self.input_shape, dtype=self.xp.float32)

        for subset in range(self.subsetter.num_subsets):
            inds = self.subsetter.get_subset_indices(subset)
            y_back += self.adjoint_subset(y[inds], inds=inds)

        return y_back


class GradientOperator(LinearOperator):
    """finite difference gradient operator"""

    def __init__(self, input_shape: tuple[int, ...],
                 xp: types.ModuleType) -> None:

        output_shape = (len(input_shape), ) + input_shape
        super().__init__(input_shape, output_shape, xp)

    def forward(self, x):
        g = self.xp.zeros(self.output_shape, dtype=x.dtype)
        for i in range(x.ndim):
            g[i, ...] = self.xp.diff(x,
                                     axis=i,
                                     append=self._xp.take(x, [-1], i))

        return g

    def adjoint(self, y):
        d = self.xp.zeros(self.input_shape, dtype=y.dtype)

        for i in range(y.shape[0]):
            d -= self.xp.diff(y[i, ...],
                              axis=i,
                              prepend=self._xp.take(y[i, ...], [0], i))

        return d


class ProjectedGradientOperator(GradientOperator):
    """projected finite difference gradient operator"""

    def __init__(self, input_shape: tuple[int, ...],
                 joint_gradient_field: npt.NDArray | cpt.NDArray,
                 xp: types.ModuleType) -> None:

        super().__init__(input_shape, xp)

        if joint_gradient_field is not None:
            norm = self.xp.linalg.norm(joint_gradient_field, axis=0)
            inds = self.xp.where(norm > 0)
            self.normalized_joint_gradient_field = joint_gradient_field.copy()

            for i in range(self.output_shape[0]):
                self.normalized_joint_gradient_field[
                    i,
                    ...][inds] = joint_gradient_field[i,
                                                      ...][inds] / norm[inds]

    def _project(self, g):
        return g - (g * self.normalized_joint_gradient_field
                    ).sum(0) * self.normalized_joint_gradient_field

    def forward(self, x):
        return self._project(super().forward(x))

    def adjoint(self, y):
        return super().adjoint(self._project(y))
