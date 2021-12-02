import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.overrides import get_default_nowrap_functions

UNARY_FNS = [
    torch.ops.aten.acos,
    torch.ops.aten.cos,
    torch.ops.aten.sqrt,
    torch.ops.aten.log,
    torch.ops.aten.exp,
    torch.ops.aten.pow,
    torch.ops.aten.sin,
    torch.ops.aten.clamp,
    torch.ops.aten.isnan,
    torch.ops.aten.abs,
]
BINARY_FNS = [
    torch.ops.aten.add,
    torch.ops.aten.sub,
    torch.ops.aten.div,
    torch.ops.aten.mul,
    torch.ops.aten.add_,
    torch.ops.aten.le,
    torch.ops.aten.ne,
    torch.ops.aten.eq,
    torch.ops.aten.bitwise_and_,
    torch.ops.aten.bitwise_or_,
]
VERBOSE = False


def is_masked_tensor(a):
    return isinstance(a, MaskedTensor)


def get_data(a):
    if is_masked_tensor(a):
        return a.masked_data
    return a


def get_mask(a):
    if is_masked_tensor(a):
        return a.masked_mask
    return None


def get_at_least_one_mask(a, b):
    assert is_masked_tensor(a) or is_masked_tensor(b)
    assert masks_match(a, b)
    if is_masked_tensor(a):
        return get_mask(a)
    return get_mask(b)


def masks_match(a, b):
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = get_mask(a)
        mask_b = get_mask(b)
        if VERBOSE:
            print(
                " mask_a.size(): ",
                mask_a.size(),
                " mask_b.size(): ",
                mask_b.size(),
                " mask_a.stride(): ",
                mask_a.stride(),
                " mask_b.stride(): ",
                mask_b.stride(),
            )
        return (mask_a.dim() == mask_b.dim()) and torch.eq(mask_a, mask_b).all().item()
    return True


def masked_tensor_str(data, mask, formatter):
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(
            map(lambda x: 8 if x[1] else len(x[0]), zip(formatted_elements, ~mask))
        )
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    sub_strings = [masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


class MaskedContigous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert is_masked_tensor(input)
        mask = get_mask(input)
        data = get_data(input)
        return MaskedTensor(data.contiguous(), mask.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Needed until https://github.com/pytorch/pytorch/issues/65243 is fixed
# since derivative includes usage of zeros_like
# https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml#L1516-L1519
class MaskedWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond, self, other):
        if VERBOSE:
            print("Calling MaskedWhere.forward")
        ctx.mark_non_differentiable(cond)
        ctx.save_for_backward(cond)
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    def backward(ctx, grad_output):
        if VERBOSE:
            print("Calling MaskedWhere.backward")
        (cond,) = ctx.saved_tensors

        def masked_out_like(mt):
            return MaskedTensor(get_data(mt), torch.zeros_like(get_mask(mt)).bool())

        return (
            None,
            torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)),
            torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output),
        )


class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {}
        kwargs["device"] = data.device
        kwargs["dtype"] = data.dtype
        kwargs["layout"] = data.layout
        kwargs["requires_grad"] = requires_grad
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

    def __init__(self, data, mask, requires_grad=False):
        if VERBOSE:
            print("----in\ntype(data): ", type(data), " type(mask): ", type(mask))
        assert type(data) == type(mask)
        assert torch.is_tensor(data)
        assert mask.dtype == torch.bool
        assert (
            data.dtype == torch.float16
            or data.dtype == torch.float32
            or data.dtype == torch.float64
            or data.dtype == torch.bool
            or data.dtype == torch.int8
            or data.dtype == torch.int16
            or data.dtype == torch.int32
            or data.dtype == torch.int64
        )
        # .contiguous cannot be overwritten so it's always contiguous
        data = data.contiguous()
        mask = mask.contiguous()
        if VERBOSE:
            print("data.dim(): ", data.dim(), " mask.dim(): ", mask.dim())
            print("data.size(): ", data.size(), " mask.size(): ", mask.size())
            print("data.stride(): ", data.stride(), " mask.stride(): ", mask.stride())
            print("data:\n", data)
            print("mask:\n", mask)
        assert data.dim() == mask.dim()
        assert data.size() == mask.size()
        assert not mask.requires_grad
        # Have to pick awkward names to not conflict with existing fields such as data
        self.masked_data = data
        self.masked_mask = mask

    def __repr__(self):
        formatter = "{0:8.4f}"
        if self.dim() == 0:
            scalar_data = get_data(self).item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not get_mask(self).item():
                data_formatted = "--"
            return (
                "masked_tensor("
                + data_formatted
                + ", "
                + str(get_mask(self).item())
                + ")"
            )
        s = masked_tensor_str(get_data(self), get_mask(self), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "masked_tensor(\n" + s + "\n)"

    # Seems like this needs to be defined before torch_dispatch to work
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if VERBOSE:
            data = get_data(args[0])
            mask = get_mask(args[0])
            print(f"----tf\n__torch_function__ calling into {func}")
            print("len(args): ", args if args is None else len(args))
            print("len(kwargs): ", len(kwargs))
            print(
                " data.dtype: ",
                data.dtype,
                " data.device: ",
                data.device,
                " data.size(): ",
                data.size(),
                " data.stride(): ",
                data.stride(),
            )
            if mask is not None:
                print(
                    " mask.dtype: ",
                    mask.dtype,
                    " mask.device: ",
                    mask.device,
                    " mask.size(): ",
                    mask.size(),
                    " mask.stride(): ",
                    mask.stride(),
                )
        if func is torch.nn.functional.multi_head_attention_forward:
            from .functions import multi_head_attention_forward as mha_mt

            return mha_mt(*args, **kwargs)
        from maskedtensor import is_reduction
        from maskedtensor import apply_reduction

        if is_reduction(func):
            return apply_reduction(func, *args, **kwargs)
        if func in [torch.Tensor.where, torch.where]:
            assert len(args) == 3
            assert len(kwargs) == 0
            return MaskedWhere.apply(*args)
        if func is torch.Tensor.contiguous:
            return MaskedContigous.apply(args[0])
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        if VERBOSE:
            print("tf redispatching to td")
        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    @classmethod
    def binary(cls, fn, args0, args1):
        if not masks_match(args0, args1):
            raise ValueError("If both inputs are MaskedTensors their masks must match.")
        result_mask = get_at_least_one_mask(args0, args1)
        return MaskedTensor(fn(get_data(args0), get_data(args1)), result_mask)

    @classmethod
    def matmul(cls, input0, input1, func):
        if VERBOSE:
            print("Calling matmul with type(input0): ", type(input0), type(input1))
        if is_masked_tensor(input0) and is_masked_tensor(input1):
            data0 = get_data(input0)
            data1 = get_data(input1)
            input_mask0 = get_mask(input0)
            input_mask1 = get_mask(input1)
            input_data0 = data0.masked_fill(~input_mask0, 0)
            input_data1 = data1.masked_fill(~input_mask1, 0)
            result_data = func(input_data0, input_data1)
            result_mask = func(input_mask0.float(), input_mask1.float())
            if VERBOSE:
                print("bmm input_data1: ", input_data1)
                print("bmm input_mask1: ", input_mask1)
                print("bmm input_data0: ", input_data0)
                print("bmm input_mask0: ", input_mask0)
                print("bmm result_data: ", result_data)
                print("bmm result_mask0: ", result_mask)
            result_mask = result_mask > 0
            if VERBOSE:
                print("bmm result_mask1: ", result_mask)
            if func is torch.ops.aten.mm:
                if VERBOSE:
                    print("input_mask1.transpose(0, 1): ", input_mask1.transpose(0, 1))
                assert torch.equal(input_mask0, input_mask1.transpose(0, 1))
            if func is torch.ops.aten.bmm:
                if VERBOSE:
                    print("input_mask1.transpose(1, 2): ", input_mask1.transpose(1, 2))
                assert torch.equal(input_mask0, input_mask1.transpose(1, 2))
            return MaskedTensor(result_data, result_mask)
        if is_masked_tensor(input0) and not is_masked_tensor(input1):
            data0 = get_data(input0)
            input_mask0 = get_mask(input0)
            input_data0 = data0.masked_fill(~input_mask0, 0)
            result_data = func(input_data0, input1)
            result_mask = func(input_mask0.float(), torch.ones_like(input1).float())
            result_mask = result_mask > 0
            return MaskedTensor(result_data, result_mask)
        if not is_masked_tensor(input0) and is_masked_tensor(input1):
            data1 = get_data(input1)
            input_mask1 = get_mask(input1)
            input_data1 = data1.masked_fill(~input_mask1, 0)
            result_data = func(input0, input_data1)
            result_mask = func(torch.ones_like(input0).float(), input_mask1.float())
            result_mask = result_mask > 0
            return MaskedTensor(result_data, result_mask)

        assert False, "can't do it"
        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        from maskedtensor import is_reduction
        from maskedtensor import apply_reduction

        if is_reduction(func):
            return apply_reduction(func, *args, **kwargs)

        from maskedtensor import is_pass_through_fn
        from maskedtensor import apply_pass_through_fn

        if is_pass_through_fn(func):
            return apply_pass_through_fn(func, *args, **kwargs)

        assert len(args) > 0
        if VERBOSE:
            print("----tp\nfunc: ", func, " args: ", args, " kwargs: ", kwargs)
        if func in [torch.ops.aten.mm, torch.ops.aten.bmm]:
            len(args) == 2
            len(kwargs) == 0
            return cls.matmul(args[0], args[1], func)
        # Doesn't work for addmm where the first argument is a Tensor
        data = get_data(args[0])
        mask = get_mask(args[0])
        if VERBOSE:
            print(
                " data.dtype: ",
                data.dtype,
                " data.device: ",
                data.device,
                " data.size(): ",
                data.size(),
                " data.stride(): ",
                data.stride(),
            )
        if mask is not None and VERBOSE:
            print(
                " mask.dtype: ",
                mask.dtype,
                " mask.device: ",
                mask.device,
                " mask.size(): ",
                mask.size(),
                " mask.stride(): ",
                mask.stride(),
            )
        if func is torch.ops.aten.new_empty_strided:
            assert len(args) == 3
            assert tuple(args[1]) == tuple(data.size())
            assert tuple(args[2]) == tuple(data.stride())
            return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)
        if func is torch.ops.aten._local_scalar_dense:
            assert mask
            return func(data)
        if func is torch.ops.aten._to_copy:
            return MaskedTensor(func(data, *args[1:]), mask)
        if func is torch.ops.aten.new_empty_strided:
            assert len(args) == 3
            assert tuple(args[1]) == tuple(data.size())
            assert tuple(args[2]) == tuple(data.stride())
            return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)
        if func in UNARY_FNS:
            assert is_masked_tensor(args[0])
            if len(kwargs) == 0 and len(args) == 1:
                return cls.unary(func, get_data(args[0]), get_mask(args[0]))
            # e.g. pow
            if len(kwargs) == 0 and len(args) == 2:
                return MaskedTensor(func(get_data(args[0]), args[1]), get_mask(args[0]))
            # e.g. clamp
            if len(kwargs) == 0 and len(args) == 3:
                return MaskedTensor(
                    func(get_data(args[0]), args[1], args[2]), get_mask(args[0])
                )
        if func in BINARY_FNS:
            assert len(kwargs) == 0
            assert len(args) == 2
            return cls.binary(func, args[0], args[1])
        if func in [torch.ops.aten.detach]:
            assert len(args) == 1
            assert len(kwargs) == 0
            return MaskedTensor(func(data), mask)
        if func is torch.ops.aten._softmax:
            assert len(args) == 3
            assert len(kwargs) == 0
            input_data = get_data(args[0]).masked_fill(
                ~get_mask(args[0]), float("-inf")
            )
            if VERBOSE:
                print("softmax data: ", data)
            result_data = func(input_data, args[1], args[2])
            if VERBOSE:
                print("softmax result_data: ", result_data)
            return MaskedTensor(result_data, get_mask(args[0]))
            assert len(args) == 1
            assert len(kwargs) == 0
            result_data = func(data)
            result_mask = func(mask)
            return MaskedTensor(result_data, result_mask)
        if func in [torch.ops.aten.ones_like]:
            len(args) == 1
            res_data = func(get_data(args[0]), **kwargs)
            if VERBOSE:
                print("ones_like - get_mask(args[0]): ", get_mask(args[0]))
                print(
                    "res_data.dtype: ",
                    res_data.dtype,
                    " res_data.device: ",
                    res_data.device,
                )
            return MaskedTensor(res_data, get_mask(args[0]))
        if func is torch.ops.aten._softmax_backward_data:
            assert len(args) == 4
            grad = args[0]
            output = args[1]
            dim = args[2]
            input_dtype = args[3]
            if is_masked_tensor(grad) and is_masked_tensor(output):
                assert masks_match(grad, output)
                grad_data = get_data(grad).masked_fill(~get_mask(grad), 1)
                output_data = get_data(output).masked_fill(~get_mask(output), 0)
                new_grad_data = torch.ops.aten._softmax_backward_data(
                    grad_data, output_data, dim, input_dtype
                )
                res = MaskedTensor(new_grad_data, get_mask(grad))
                return res
        if func is torch.ops.aten.copy_:
            assert len(args) == 2
            assert masks_match(get_mask(args[0]), get_mask(args[1]))
            func(data, get_data(args[1]))
            return args[0]
        if func in [torch.ops.aten._s_where]:
            assert len(kwargs) == 0
            assert len(args) == 3
            assert torch.is_tensor(args[0])
            mx = args[1]
            my = args[2]
            if not is_masked_tensor(mx):
                mx = MaskedTensor(mx, torch.ones_like(mx).bool())
            if not is_masked_tensor(my):
                my = MaskedTensor(my, torch.ones_like(my).bool())
            assert is_masked_tensor(mx)
            assert is_masked_tensor(my)
            new_data = func(args[0], get_data(mx), get_data(my))
            new_mask = func(args[0], get_mask(mx), get_mask(my))
            return MaskedTensor(new_data, new_mask)
        return NotImplemented

    def __lt__(self, other):
        return MaskedTensor(get_data(self) < other, get_mask(self))

    def to_tensor(self, value):
        return get_data(self).masked_fill(~get_mask(self), value)

    def mask(self):
        return self.masked_mask
