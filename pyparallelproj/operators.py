import abc

import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy as np
    import numpy.typing as cpt


class LinearOperator(abc.ABC):

    def __init__(self, x_shape: tuple, y_shape: tuple) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        x_shape : tuple
            shape of x array
        y_shape : tuple
            shape of y array
        """
        super().__init__()

        self._x_shape = x_shape
        self._y_shape = y_shape

    @property
    def x_shape(self) -> tuple:
        """shape of x array

        Returns
        -------
        tuple
            shape of x array
        """
        return self._x_shape

    @property
    def y_shape(self):
        """shape of y array

        Returns
        -------
        tuple
            shape of y array
        """
        return self._y_shape

    @abc.abstractmethod
    def forward(self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
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
    def adjoint(self, y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
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

    def adjointness_test(self) -> None:
        """test if adjoint is really the adjoint of forward
        """
        x = np.random.rand(*self._x_shape)
        y = np.random.rand(*self._y_shape)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        assert (np.isclose((x_fwd * y).sum(), (x * y_back).sum()))

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

        x = np.random.rand(*self._x_shape)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = np.linalg.norm(x.ravel())
            x /= n

        return np.sqrt(n)

