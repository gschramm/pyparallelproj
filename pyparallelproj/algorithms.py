import types

import numpy as np
import numpy.typing as npt

import pyparallelproj.operators as operators

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
                 acquisition_model: operators.LinearSubsetOperator,
                 verbose: bool = True):

        self._data = data
        self._contamination = contamination
        self._acquisition_model = acquisition_model
        self._verbose = verbose

        self._epoch_counter = 0
        self._cost = np.array([], dtype=self.xp.float32)
        self._x = None

        # allocated array for the sensitivity images
        self._adjoint_ones = self.xp.zeros(
            (self.acquisition_model.subsetter.num_subsets, ) +
            self.acquisition_model.input_shape,
            dtype=self.xp.float32)

        self.setup()

    @property
    def data(self) -> npt.NDArray | cpt.NDArray:
        return self._data

    @property
    def acquisition_model(self) -> operators.LinearSubsetOperator:
        return self._acquisition_model

    @property
    def contamination(self) -> npt.NDArray | cpt.NDArray:
        return self._contamination

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def xp(self) -> types.ModuleType:
        return self.acquisition_model.xp

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

    def setup(self, x: None | npt.NDArray | cpt.NDArray = None):
        self._epoch_counter = 0
        self._cost = np.array([], dtype=self.xp.float32)

        if x is None:
            self._x = self.xp.full(self.acquisition_model.input_shape,
                                   1.0,
                                   dtype=self.xp.float32)
        else:
            self._x = x.copy()

        # calculate the sensitivity images
        for subset in range(self.acquisition_model.subsetter.num_subsets):
            ones = self.xp.ones(
                self.acquisition_model.get_subset_shape(subset),
                dtype=self.xp.float32)

            self._adjoint_ones[subset,
                               ...] = self.acquisition_model.adjoint_subset(
                                   ones, subset)

    def subset_update(self, subset: int) -> None:
        # get the LOR indices belonging to the current subset
        inds = self.acquisition_model.subsetter.get_subset_indices(subset)
        # calculate the expected data given the current reconstruction
        expected_data = self.acquisition_model.forward_subset(
            self._x, inds=inds) + self._contamination[inds]
        # ratio of measured data and expected data
        ratio = (self._data[inds] / expected_data).astype(self.xp.float32)
        # OSEM update
        self._x *= (self.acquisition_model.adjoint_subset(ratio, inds=inds) /
                    self._adjoint_ones[subset, ...])

    def run(self, niter: int, evaluate_cost: bool = False):

        cost = np.zeros(niter, dtype=np.float32)

        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.acquisition_model.subsetter.num_subsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.subset_update(isub)

            if evaluate_cost:
                cost[it] = self.evaluate_cost()

            self._epoch_counter += 1

        self._cost = np.concatenate((self.cost, cost))

    def evaluate_cost(self):
        cost = 0
        for subset in range(self.acquisition_model.subsetter.num_subsets):
            inds = self.acquisition_model.subsetter.get_subset_indices(subset)
            expected_data = self.acquisition_model.forward_subset(
                self._x, inds=inds) + self._contamination[inds]

            cost += float((expected_data -
                           self.data[inds] * self.xp.log(expected_data)).sum())

        return cost
