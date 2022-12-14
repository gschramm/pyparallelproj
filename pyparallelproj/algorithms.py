import types
from time import time

import numpy as np
import numpy.typing as npt

from . import operators
from . import functionals

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class OSEM:

    def __init__(self,
                 data: npt.NDArray | cpt.NDArray,
                 contamination: npt.NDArray | cpt.NDArray,
                 data_operator: operators.LinearSubsetOperator,
                 verbose: bool = True) -> None:

        self._data = data
        self._contamination = contamination
        self._data_operator = data_operator
        self._verbose = verbose

        self._epoch_counter = 0
        self._cost = np.array([], dtype=self.xp.float32)
        self._walltime = []
        self._x = None

        # allocated array for the sensitivity images
        self._adjoint_ones = self.xp.zeros(
            (self.data_operator.subsetter.num_subsets, ) +
            self.data_operator.input_shape,
            dtype=self.xp.float32)

        self.setup()

    @property
    def data(self) -> npt.NDArray | cpt.NDArray:
        return self._data

    @property
    def data_operator(self) -> operators.LinearSubsetOperator:
        return self._data_operator

    @property
    def contamination(self) -> npt.NDArray | cpt.NDArray:
        return self._contamination

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def xp(self) -> types.ModuleType:
        return self.data_operator.xp

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._x

    @property
    def adjoint_ones(self) -> npt.NDArray | cpt.NDArray:
        return self._adjoint_ones

    @property
    def epoch_counter(self) -> int:
        return self._epoch_counter

    @property
    def cost(self) -> npt.NDArray:
        return self._cost

    @property
    def walltime(self) -> list[float, ...]:
        return self._walltime

    def setup(self, x: None | npt.NDArray | cpt.NDArray = None):
        self._epoch_counter = 0
        self._cost = np.array([], dtype=self.xp.float32)

        if x is None:
            self._x = self.xp.full(self.data_operator.input_shape,
                                   1.0,
                                   dtype=self.xp.float32)
        else:
            self._x = x.copy()

        # calculate the sensitivity images
        for subset in range(self.data_operator.subsetter.num_subsets):
            ones = self.xp.ones(self.data_operator.get_subset_shape(subset),
                                dtype=self.xp.float32)

            self._adjoint_ones[
                subset, ...] = self.data_operator.adjoint_subset(
                    ones,
                    self.data_operator.subsetter.get_subset_indices(subset))

    def subset_update(self, subset: int) -> None:
        # get the LOR indices belonging to the current subset
        subset_inds = self.data_operator.subsetter.get_subset_indices(subset)
        # calculate the expected data given the current reconstruction
        expected_data = self.data_operator.forward_subset(
            self._x, subset_inds) + self._contamination[subset_inds]
        # ratio of measured data and expected data
        ratio = (self._data[subset_inds] / expected_data).astype(
            self.xp.float32)
        # OSEM update
        self._x *= (self.data_operator.adjoint_subset(ratio, subset_inds) /
                    self._adjoint_ones[subset, ...])

    def run(self, niter: int, evaluate_cost: bool = False):
        cost = np.zeros(niter, dtype=np.float32)
        self._walltime.append(time())

        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.data_operator.subsetter.num_subsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.subset_update(isub)

            if evaluate_cost:
                cost[it] = self.evaluate_cost()

            self._epoch_counter += 1
            self._walltime.append(time())

        self._cost = np.concatenate((self.cost, cost))

    def evaluate_cost(self,
                      x: None | npt.NDArray | cpt.NDArray = None) -> float:

        if x is None:
            x = self.x

        expected_data = self.data_operator.forward(x) + self._contamination

        cost = float(
            (expected_data - self.data * self.xp.log(expected_data)).sum())

        return cost


class PDHG:
    """generic primal-dual hybrid gradient algorithm (Chambolle-Pock) for optimizing
       data_distance(data_operator x) + beta*(prior_functional(prior_operator x)) + g_functional(x)"""

    def __init__(
            self,
            data_operator: operators.LinearOperator,
            data_distance: functionals.FunctionalWithDualProx,
            prior_operator: operators.LinearOperator,
            prior_functional: functionals.FunctionalWithDualProx,
            beta: float,
            sigma: float,
            tau: float,
            theta: float = 0.999,
            contamination: None | npt.NDArray | cpt.NDArray = None,
            g_functional: functionals.FunctionalWithProx | None = None
    ) -> None:
        """
        Parameters
        ----------
        data_operator : operators.LinearOperator
            operator mapping current image to expected data
        data_distance : functionals.FunctionalWithDualProx
            norm applied to (expected data - data)
        prior_operator : operators.LinearOperator
            prior operator
        prior_functional : functionals.FunctionalWithDualProx
            prior norm
        beta : float
            weight of prior
        sigma : float
            primal step size 
        tau : float
            dual step size 
        theta : float, optional
            theta parameter, by default 0.999
        contamination : None | npt.NDArray | cpt.NDArray, optional
            vector of additive contaminations in forward model, optional
        g_functional : None | functionals.FunctionalWithProx
            the G functional
        """

        self._data_operator = data_operator
        self._data_distance = data_distance

        self._prior_operator = prior_operator
        self._prior_functional = prior_functional

        self._beta = beta

        self._sigma = sigma
        self._tau = tau
        self._theta = theta

        self._contamination = contamination
        self._g_functional = g_functional

        self._x = self.xp.zeros(self._data_operator.input_shape,
                                dtype=self.xp.float32)
        self._xbar = self.xp.zeros(self._data_operator.input_shape,
                                   dtype=self.xp.float32)
        self._y_data = self.xp.zeros(self._data_operator.output_shape,
                                     dtype=self.xp.float32)
        self._y_prior = self.xp.zeros(self._prior_operator.output_shape,
                                      dtype=self.xp.float32)

        self.setup()

    @property
    def data_operator(self) -> operators.LinearOperator:
        return self._data_operator

    @property
    def xp(self) -> types.ModuleType:
        return self._data_operator.xp

    @property
    def x(self) -> npt.NDArray | cpt.NDArray:
        return self._x

    @property
    def y_data(self) -> npt.NDArray | cpt.NDArray:
        return self._y_data

    @property
    def y_prior(self) -> npt.NDArray | cpt.NDArray:
        return self._y_prior

    @property
    def cost_data(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._cost_data)

    @property
    def cost_prior(self) -> npt.NDArray | cpt.NDArray:
        return self.xp.array(self._cost_prior)

    @property
    def cost(self) -> npt.NDArray | cpt.NDArray:
        return self.cost_data + self.cost_prior

    def setup(self) -> None:
        self._x = self.xp.zeros(self._data_operator.input_shape,
                                dtype=self.xp.float32)
        self._xbar = self.xp.zeros(self._data_operator.input_shape,
                                   dtype=self.xp.float32)
        self._y_data = self.xp.zeros(self._data_operator.output_shape,
                                     dtype=self.xp.float32)
        self._y_prior = self.xp.zeros(self._prior_operator.output_shape,
                                      dtype=self.xp.float32)

        self._epoch_counter = 0
        self._cost_data = []
        self._cost_prior = []

    def update(self) -> None:
        # data forward step
        xbar_fwd = self._data_operator.forward(self._xbar)

        if self._contamination is not None:
            xbar_fwd += self._contamination

        self._y_data = self._y_data + self._sigma * xbar_fwd

        # prox of data fidelity
        self._y_data = self._data_distance.prox_convex_dual(self._y_data,
                                                            sigma=self._sigma)

        # prior operator forward step
        if self._beta > 0:
            self._y_prior = self._y_prior + self._sigma * self._prior_operator.forward(
                self._xbar)
            # prox of prior norm
            self._y_prior = self._beta * self._prior_functional.prox_convex_dual(
                self._y_prior / self._beta, sigma=self._sigma / self._beta)

        x_plus = self._x - self._tau * self._data_operator.adjoint(
            self._y_data)

        if self._beta > 0:
            x_plus -= self._tau * self._prior_operator.adjoint(self._y_prior)

        if self._g_functional is not None:
            x_plus = self._g_functional.prox(x_plus, self._tau)

        self._xbar = x_plus + self._theta * (x_plus - self._x)

        self._x = x_plus.copy()

        self._epoch_counter += 1

    def run(self,
            num_iterations: int,
            calculate_cost: bool = False,
            verbose: bool = True) -> None:

        for i in range(num_iterations):
            self.update()
            if verbose:
                print(f'iteration {self._epoch_counter}')
            if calculate_cost:
                x_fwd = self._data_operator.forward(self._x)
                if self._contamination is not None:
                    x_fwd += self._contamination
                self._cost_data.append(self._data_distance(x_fwd))
                self._cost_prior.append(self._beta * self._prior_functional(
                    self._prior_operator.forward(self._x)))