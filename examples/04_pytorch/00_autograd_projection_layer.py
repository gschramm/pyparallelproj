"""minimal example that shows how to create projection and back projection layers that can
   be used with pytorch autograd 
   https://docs.cupy.dev/en/stable/user_guide/interoperability.html#using-custom-kernels-in-pytorch
   https://pytorch.org/docs/stable/notes/extending.html
"""

import torch
import math
import cupy as cp
import cupy.typing as cpt


class PETProjector:
    """dummy PET projector that maps a 3D array into a 4D array using
       a random dense matrix
    """

    def __init__(
        self, input_shape=(3, 4, 5), output_shape=(2, 3, 2, 4)) -> None:
        self._input_shape = input_shape
        self._output_shape = output_shape

        self._A = cp.random.rand(math.prod(self._output_shape),
                                 math.prod(self._input_shape)).astype(
                                     cp.float32)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        return self._output_shape

    def forward(self, x: cpt.NDArray) -> cpt.NDArray:
        return (self._A @ x.ravel()).reshape(self._output_shape)

    def adjoint(self, y: cpt.NDArray) -> cpt.NDArray:
        return (self._A.T @ y.ravel()).reshape(self._input_shape)


class PETFwdProjectionLayer(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, pet_projector: PETProjector):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.pet_projector = pet_projector

        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function written in cp
        cp_y = pet_projector.forward(cp_x)

        torch_y = torch.from_dlpack(cp_y)

        return torch_y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None
        else:
            pet_projector = ctx.pet_projector

            cp_grad_output = cp.from_dlpack(grad_output.detach())

            torch_grad = torch.from_dlpack(
                pet_projector.adjoint(cp_grad_output))

            return torch_grad, None


if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device("cuda:0")  # Uncomment this to run on GPU

    proj = PETProjector()

    x = torch.rand(proj.input_shape,
                   device=device,
                   dtype=dtype,
                   requires_grad=True)

    # To apply our Function, we use Function.apply method. We alias this as 'P3'.
    pet_fwd = PETFwdProjectionLayer.apply

    # define a scalar dummy function involving a forward projection layer
    Q = pet_fwd(x, proj).sum()

    # check the gradients using gradcheck
    test_result = torch.autograd.gradcheck(pet_fwd, (x, proj), eps=1e-6)

    # call the backpropagation step
    Q.backward()