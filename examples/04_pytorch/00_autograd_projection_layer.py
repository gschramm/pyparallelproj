"""minimal example that shows how to create projection and back projection layers that can
   be used with pytorch autograd 
   https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
   https://docs.cupy.dev/en/stable/user_guide/interoperability.html#using-custom-kernels-in-pytorch
"""

import torch
import cupy
import cupy.typing as cpt


class PETProjector:
    """dummy PET projector that maps a 2 element cupy array 
       to a 3 element cupy array using a fixed dense matrix
    """

    def __init__(self) -> None:
        self._A = cupy.array([[1., 1.], [-1., 1.], [0, 1.]],
                             dtype=cupy.float32)

    def forward(self, x: cpt.NDArray) -> cpt.NDArray:
        return self._A @ x

    def adjoint(self, y: cpt.NDArray) -> cpt.NDArray:
        return self._A.T @ y


class PETProjectionLayer(torch.autograd.Function):
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
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        cupy_x = cupy.ascontiguousarray(cupy.from_dlpack(x.detach()))

        ctx.pet_projector = pet_projector

        # a custom function written in cupy
        cupy_y = pet_projector.forward(cupy_x)

        torch_y = torch.from_dlpack(cupy_y)

        return torch_y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        pet_projector = ctx.pet_projector

        cupy_grad_output = cupy.from_dlpack(grad_output.detach()).ravel()

        torch_grad = torch.from_dlpack(pet_projector.adjoint(cupy_grad_output))

        return torch_grad, None


if __name__ == '__main__':
    dtype = torch.float64
    device = torch.device("cuda:0")  # Uncomment this to run on GPU

    x = torch.tensor([1., 1.], device=device, dtype=dtype, requires_grad=True)

    # To apply our Function, we use Function.apply method. We alias this as 'P3'.
    PPL = PETProjectionLayer.apply

    proj = PETProjector()

    Q = PPL(x, proj).sum()

    # check the gradients using gradcheck
    test_result = torch.autograd.gradcheck(PPL, (x, proj), eps=1e-6)