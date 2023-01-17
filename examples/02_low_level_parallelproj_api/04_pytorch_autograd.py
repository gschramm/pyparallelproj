""" basic example on how to use cupy and pytorch autograd """
import cupy
import torch


class CuPyLog(torch.autograd.Function):
    """ simple autograd function that calculates element wise log(x) using cupy """

    @staticmethod
    def forward(ctx, x):
        ctx.input = x
        # Enforce contiguous arrays to simplify RawKernel indexing.
        cupy_x = cupy.ascontiguousarray(cupy.from_dlpack(x.detach()))

        #-------------------------------------------------------------
        #--- the actual function -------------------------------------
        #-------------------------------------------------------------
        cupy_y = cupy.log(cupy_x)
        #-------------------------------------------------------------
        #-------------------------------------------------------------
        #-------------------------------------------------------------

        # the ownership of the device memory backing cupy_y is implicitly
        # transferred to torch_y, so this operation is safe even after
        # going out of scope of this function.
        torch_y = torch.from_dlpack(cupy_y)
        return torch_y

    @staticmethod
    def backward(ctx, grad_y):
        # Enforce contiguous arrays to simplify RawKernel indexing.
        cupy_input = cupy.from_dlpack(ctx.input.detach()).ravel()
        cupy_grad_y = cupy.from_dlpack(grad_y.detach()).ravel()

        #-------------------------------------------------------------
        #---- gradient of the function -------------------------------
        #-------------------------------------------------------------
        cupy_grad_x = cupy_grad_y / cupy_input
        #-------------------------------------------------------------
        #-------------------------------------------------------------
        #-------------------------------------------------------------

        # the ownership of the device memory backing cupy_grad_x is implicitly
        # transferred to torch_y, so this operation is safe even after
        # going out of scope of this function.
        torch_grad_x = torch.from_dlpack(cupy_grad_x).reshape(ctx.input.shape)

        return torch_grad_x


if __name__ == '__main__':
    x = torch.rand(3, 4, 5, requires_grad=True, device='cuda')
    torch.autograd.gradcheck(CuPyLog.apply, x, eps=1e-4)
