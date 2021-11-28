import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.overrides import get_default_nowrap_functions
from maskedtensor import MaskedTensor


def masked_all_all(data, mask=None):
    if mask is None:
        return data.all()
    return data.masked_fill(~mask, True).all()


def masked_all_dim(data, dim, keepdim=False, mask=None):
    if mask is None:
        return torch.all(data, dim=dim, keepdim=keepdim)
    return torch.all(data.masked_fill(~mask, True), dim=dim, keepdim=keepdim)


# TODO: Add masked_all to torch._masked?
def masked_all(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 1:
        return masked_all_all(args[0], mask=kwargs["mask"])
    return masked_all_dim(*args, **kwargs)


def get_masked_fn(fn):
    if fn == "all":
        return masked_all
    return getattr(torch._masked, fn)


def torch_reduce_all(fn):
    def reduce_all(self):
        data = self.masked_data
        mask = self.masked_mask
        masked_fn = get_masked_fn(fn)
        return MaskedTensor(masked_fn(data, mask=mask), torch.any(mask))

    return reduce_all


# If hope this signature won't change to frequently
def torch_reduce_dim(fn):
    def reduce_dim(self, dim, keepdim=False, dtype=None):
        data = self.masked_data
        mask = self.masked_mask
        masked_fn = get_masked_fn(fn)
        if fn == "all":
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            result_data = masked_fn(
                data, dim=dim, keepdim=keepdim, dtype=dtype, mask=mask
            )
        return MaskedTensor(result_data, torch.any(mask, dim=dim, keepdim=keepdim),)

    return reduce_dim


def torch_reduce(fn):
    def torch_sum(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return torch_reduce_all(fn)(args[0])
        return torch_reduce_dim(fn)(*args, **kwargs)

    return torch_sum


def torch_grad_reduce_all(fn):
    class MaskedReduceAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            mask = input.masked_mask
            ctx.mark_non_differentiable(mask)
            ctx.save_for_backward(mask)
            return torch_reduce_all(fn)(input)

        @staticmethod
        def backward(ctx, grad_output):
            (mask,) = ctx.saved_tensors
            grad_data = grad_output.masked_data.expand_as(mask)
            return MaskedTensor(grad_data, mask)

    return MaskedReduceAll.apply


def torch_grad_reduce_dim(fn):
    class MaskedReduceDim(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, dim, keepdim, dtype):
            mask = input.masked_mask
            ctx.mark_non_differentiable(mask)
            ctx.save_for_backward(mask)
            return torch_reduce_dim(fn)(input, dim, keepdim, dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (mask,) = ctx.saved_tensors
            grad_data = grad_output.masked_data.expand_as(mask)
            return MaskedTensor(grad_data, mask)

    return MaskedReduceDim.apply


def reduce_dim_args(input, dim, keepdim=False, dtype=None):
    return input, dim, keepdim, dtype


def torch_grad_reduce(fn):
    def grad_reduce(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            return torch_grad_reduce_all(fn)(args[0])
        # TODO: autograd.Function doesn't support kwarg
        input, dim, keepdim, dtype = reduce_dim_args(*args, **kwargs)
        return torch_grad_reduce_dim(fn)(input, dim, keepdim, dtype)

    return grad_reduce


REDUCE_NAMES = ["sum", "mean", "amin", "amax", "prod", "all"]

NATIVE_REDUCE_MAP = {
    getattr(torch.ops.aten, name): torch_reduce(name) for name in REDUCE_NAMES
}

TORCH_REDUCE_MAP = {
    getattr(torch, name): torch_grad_reduce(name) for name in REDUCE_NAMES
}

TENSOR_REDUCE_MAP = {
    getattr(torch.Tensor, name): torch_grad_reduce(name) for name in REDUCE_NAMES
}


def is_reduction(fn):
    return fn in NATIVE_REDUCE_MAP or fn in TORCH_REDUCE_MAP or fn in TENSOR_REDUCE_MAP


def apply_reduction(fn, *args, **kwargs):
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    return NotImplemented
