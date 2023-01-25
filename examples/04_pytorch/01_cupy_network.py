"""minimal example that shows how to setup a cascaded model with layers combining
   (cupy) projections and 3D convolutions

   in this mini demo, we use a dummy cupy projector for demonstration purposes
"""

import torch
import math
import cupy as cp
import cupy.typing as cpt


class PETProjector:
    """dummy PET projector that maps a 3D array into a 4D array using
       a random dense matrix

       implemented to test pytorch autograd functions
    """

    def __init__(
        self, input_shape=(5, 6, 7), output_shape=(3, 5, 4, 3)) -> None:
        self._input_shape = input_shape
        self._output_shape = output_shape

        self._A = cp.random.rand(math.prod(self._output_shape),
                                 math.prod(self._input_shape)).astype(
                                     cp.float32) / 100

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
    """PET forward projection layer mapping a 3D image to a 4D sinogram

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

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = pet_projector.forward(cp_x)

        return torch.from_dlpack(cp_y)

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

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes two input arguments (x and projector)
            # we have to return two arguments (the latter is None)
            return torch.from_dlpack(
                pet_projector.adjoint(cp_grad_output)), None


class PETAdjointProjectionLayer(torch.autograd.Function):
    """ adjoint of PET forward projection layer mapping a 4D sinogram to a 3D image
    
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

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = pet_projector.adjoint(cp_x)

        return torch.from_dlpack(cp_y)

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

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes two input arguments (x and projector)
            # we have to return two arguments (the latter is None)
            return torch.from_dlpack(
                pet_projector.forward(cp_grad_output)), None


class MyModel(torch.nn.Module):
    """dummy cascaded model that includes layers combining projections and convolutions"""

    def __init__(self, proj: PETProjector, data: torch.Tensor,
                 contaminations: torch.Tensor,
                 device: torch.DeviceObjType) -> None:
        super(MyModel, self).__init__()
        self._proj = proj
        self._data = data
        self._contaminations = contaminations
        self._device = device

        self._pet_fwd = PETFwdProjectionLayer.apply
        self._pet_adjoint = PETAdjointProjectionLayer.apply

        self._conv = torch.nn.Conv3d(1,
                                     1, (3, 3, 3),
                                     padding='same',
                                     device=self._device,
                                     dtype=torch.float32)

    def _proj_conv(self, x: torch.Tensor) -> torch.Tensor:
        """layer combining a Poisson logL update with a 3DConv
           input is a 5D torch tensor with dimension (batch_size, 1, n0, n1, n2)
           ouput tensor has same shape as the input
        """

        num_batch = x.shape[0]

        x_fwd_pet = torch.zeros_like(x)

        for i in range(num_batch):
            x_fwd_pet[i, 0, ...] = self._pet_adjoint(
                1 - self._data[i, 0, ...] /
                (self._pet_fwd(x[i, 0, ...], self._proj) +
                 self._contaminations[i, 0, ...]), self._proj)

        x_fwd_conv = self._conv(x)

        return x_fwd_pet + x_fwd_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """cascaded proj+conv layers
           input is a 5D torch tensor with dimension (batch_size, 1, n0, n1, n2)
           ouput tensor has same shape as the input
        """
        x1 = self._proj_conv(x)
        x2 = self._proj_conv(x1)
        x3 = self._proj_conv(x2)

        return x3


if __name__ == '__main__':
    dtype = torch.float32
    device = torch.device("cuda:0")  # Uncomment this to run on GPU

    proj = PETProjector()

    batch_size = 3

    x = torch.rand((batch_size, 1) + proj.input_shape,
                   device=device,
                   dtype=dtype,
                   requires_grad=False)

    d = torch.rand((batch_size, 1) + proj.output_shape,
                   device=device,
                   dtype=dtype,
                   requires_grad=False)

    c = torch.rand((batch_size, 1) + proj.output_shape,
                   device=device,
                   dtype=dtype,
                   requires_grad=False)

    model = MyModel(proj, d, c, device)

    x_fwd = model.forward(x)

    y = x_fwd.sum()
    y.backward()