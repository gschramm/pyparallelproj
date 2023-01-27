import torch
import cupy as cp
import pyparallelproj.operators as operators


class LinearSubsetForwardLayer(torch.autograd.Function):
    """PET forward projection layer mapping a 3D image to a 4D sinogram

    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, subset_operator: operators.LinearSubsetOperator,
                subset: int):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.subset_operator = subset_operator
        ctx.subset = subset

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = subset_operator.forward_subset(
            cp_x, subset_operator.subsetter.get_subset_indices(subset))

        return torch.from_dlpack(cp_y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        For details on how to implement the backward pass, see
        https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        """

        if grad_output is None:
            return None, None, None
        else:
            subset_operator = ctx.subset_operator
            subset = ctx.subset

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes three input arguments (x, projector, subset)
            # we have to return three arguments (the latter is None)
            return torch.from_dlpack(
                subset_operator.adjoint_subset(
                    cp_grad_output,
                    subset_operator.subsetter.get_subset_indices(
                        subset))), None, None


class LinearSubsetAdjointLayer(torch.autograd.Function):
    """ adjoint of PET forward projection layer mapping a 4D sinogram to a 3D image
    
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, subset_operator: operators.LinearSubsetOperator,
                subset: int):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation.
        """

        ctx.set_materialize_grads(False)
        ctx.subset_operator = subset_operator
        ctx.subset = subset

        # convert pytorch input tensor into cupy array
        cp_x = cp.ascontiguousarray(cp.from_dlpack(x.detach()))

        # a custom function that maps from cupy array to cupy array
        cp_y = subset_operator.adjoint_subset(
            cp_x, subset_operator.subsetter.get_subset_indices(subset))

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
            return None, None, None
        else:
            subset_operator = ctx.subset_operator
            subset = ctx.subset

            # convert torch array to cupy array
            cp_grad_output = cp.from_dlpack(grad_output.detach())

            # since forward takes three input arguments (x, projector, subset)
            # we have to return three arguments (the latter is None)
            return torch.from_dlpack(
                subset_operator.forward_subset(
                    cp_grad_output,
                    subset_operator.subsetter.get_subset_indices(
                        subset))), None, None
