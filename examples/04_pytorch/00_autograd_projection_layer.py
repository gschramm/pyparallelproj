"""minimal example that shows how to use an autograd function involving a linear subset operator implemented in cupy
   https://docs.cupy.dev/en/stable/user_guide/interoperability.html#using-custom-kernels-in-pytorch
   https://pytorch.org/docs/stable/notes/extending.html
"""

import torch
import math
import cupy as cp
import cupy.typing as cpt

from pyparallelproj.operators import LinearSubsetOperator
from pyparallelproj.subsets import Strided1DSubsetter

from layers import LinearSubsetForwardLayer, LinearSubsetAdjointLayer


class DummySubsetOperator(LinearSubsetOperator):
    """dummy PET projector that maps a 3D array into a 4D array using
       a random dense matrix

       implemented to test pytorch autograd functions
    """

    def __init__(self, input_shape=(3, 4, 5), output_shape=(15, )) -> None:
        super().__init__(input_shape, output_shape, cp,
                         Strided1DSubsetter(output_shape[0], 4))

        self._A = cp.random.rand(math.prod(self.output_shape),
                                 math.prod(self.input_shape)).astype(
                                     cp.float64)

    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        return (self.subsetter.get_subset_index_len(subset), )

    def forward_subset(self, x: cpt.NDArray, subset_inds) -> cpt.NDArray:
        return (self._A[subset_inds] @ x.ravel())

    def adjoint_subset(self, y: cpt.NDArray, subset_inds) -> cpt.NDArray:
        return (self._A[subset_inds].T @ y.ravel()).reshape(self.input_shape)


if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device("cuda:0")

    # here we only use a dummy subset operator based on matrix multiplication
    # but the operator could be replaced by a real world subset operator, e.g.
    # a projector that can handle subsets
    subset_op = DummySubsetOperator()

    fwd_layer = LinearSubsetForwardLayer.apply
    adjoint_layer = LinearSubsetAdjointLayer.apply

    for i in range(subset_op.subsetter.num_subsets):
        print(f'{i+1}/{subset_op.subsetter.num_subsets}')
        x = torch.rand(subset_op.input_shape,
                       device=device,
                       dtype=dtype,
                       requires_grad=True)

        y = torch.rand(subset_op.get_subset_shape(i),
                       device=device,
                       dtype=dtype,
                       requires_grad=True)

        # check the gradients using torch's gradcheck
        grad_test_fwd = torch.autograd.gradcheck(fwd_layer, (x, subset_op, i),
                                                 eps=1e-6)
        print(f'grad_test_fwd: {grad_test_fwd}')
        grad_test_adjoint = torch.autograd.gradcheck(adjoint_layer,
                                                     (y, subset_op, i),
                                                     eps=1e-6)
        print(f'grad_test_adjoint: {grad_test_adjoint}')
