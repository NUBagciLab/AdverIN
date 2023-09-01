import torch
import torch.nn as nn
from torch.autograd import Function

eps = 1e-4

class _ReverseGrad(Function):

    @staticmethod
    def forward(ctx, input, grad_scaling=1.):
        ctx.grad_scaling = grad_scaling
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_scaling = ctx.grad_scaling
        return -grad_scaling * grad_output, None


reverse_grad = _ReverseGrad.apply


class ReverseGrad(nn.Module):
    """Gradient reversal layer after norm.

    It acts as an identity layer in the forward,
    but reverses the sign of the gradient in
    the backward.
    """

    def forward(self, x, grad_scaling=1.):
        return reverse_grad(x, grad_scaling)


class _ReverseNormGrad(Function):

    @staticmethod
    def forward(ctx, input, p, grad_norm, grad_axis):
        ctx.grad_norm = grad_norm
        ctx.p = p
        ctx.grad_axis = grad_axis
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_norm, p, axis = ctx.grad_norm, ctx.p, ctx.grad_axis
        grad_output = grad_norm*grad_output / (torch.norm(grad_output, p=p, dim=axis, keepdim=True)+eps)
        return -grad_output, None, None, None


reverse_normed_grad = _ReverseNormGrad.apply

class ReverseNormGrad(nn.Module):
    """Gradient reversal layer.

    It acts as an identity layer in the forward,
    but reverses the sign of the gradient in
    the backward.
    """

    def forward(self, x:torch.Tensor, p:int=2, grad_norm=1., axis:tuple=None):
        if axis is None:
            axis = tuple(range(x.dim()))
        return reverse_normed_grad(x, p, grad_norm, axis)
