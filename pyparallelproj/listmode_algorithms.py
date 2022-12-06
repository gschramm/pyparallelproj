import types
from time import time

import numpy as np
import numpy.typing as npt

import pyparallelproj.operators as operators
import pyparallelproj.functionals as functionals

try:
    import cupy.typing as cpt
except:
    import warnings
    warnings.warn('cupy module not available')
    import numpy.typing as cpt


class LM_OSEM:

    def __init__(self,
                 contamination_list: npt.NDArray | cpt.NDArray,
                 data_operator: operators.LinearListmodeSubsetOperator,
                 adjoint_ones: npt.NDArray | cpt.NDArray,
                 verbose: bool = True):

        self._contamination_list = contamination_list
        self._data_operator = data_operator
        self._verbose = verbose

        self._epoch_counter = 0
        self._cost = np.array([], dtype=self.xp.float32)
        self._walltime = []
        self._x = None

        self._adjoint_ones = adjoint_ones

        self.setup()

    @property
    def data_operator(self) -> operators.LinearListmodeSubsetOperator:
        return self._data_operator

    @property
    def contamination_list(self) -> npt.NDArray | cpt.NDArray:
        return self._contamination_list

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
    def num_subsets(self) -> int:
        return self.data_operator.listmode_subsetter.num_subsets

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

    def subset_update(self, subset: int) -> None:
        # get the LOR indices belonging to the current subset
        subset_inds = self.data_operator.listmode_subsetter.get_subset_indices(
            subset)

        exp_list = self.data_operator.forward_listmode_subset(
            self.x, subset_inds) + self.contamination_list[subset_inds]

        self._x *= (self.data_operator.adjoint_listmode_subset(
            self.num_subsets / exp_list, subset_inds) / self.adjoint_ones)

    def run(self, niter: int):
        self._walltime.append(time())
        for it in range(niter):
            if self.verbose:
                print(f"iteration {self.epoch_counter+1}")
            for isub in range(self.num_subsets):
                if self.verbose:
                    print(f"subset {isub+1}", end="\r")
                self.subset_update(isub)

            self._epoch_counter += 1
            self._walltime.append(time())
