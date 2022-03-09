# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch.overrides import get_default_nowrap_functions
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map

BINARY_NAMES = [
    "add",
    #    "addcdiv",
    #    "addcmul",
    "atan2",
    "arctan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "div",
    "divide",
    "floor_divide",
    "fmod",
    "logaddexp",
    "logaddexp2",
    #    "logical_and",
    #    "logical_or",
    #    "logical_xor",
    "mul",
    "multiply",
    "nextafter",
    "remainder",
    "sub",
    "subtract",
    "true_divide",
    "eq",
    "ne",
    "le",
    "ge",
    "greater",
    "greater_equal",
    # "equal",
    "gt",
    # "isclose",
    "less_equal",
    "lt",
    "less",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "not_equal",
]

INPLACE_BINARY_NAMES = [
    n + "_"
    for n in (
        list(
            set(BINARY_NAMES)
            - set(
                [
                    "logaddexp",
                    "logaddexp2",
                    "equal",
                    "fmin",
                    "minimum",
                    "maximum",
                    "fmax",
                ]
            )
        )
    )
]


def get_mask(a):
    from maskedtensor import is_masked_tensor

    if is_masked_tensor(a):
        return a.masked_mask
    return None


def masks_match(a, b):
    from maskedtensor import is_masked_tensor

    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = get_mask(a)
        mask_b = get_mask(b)
        return (mask_a.dim() == mask_b.dim()) and torch.eq(mask_a, mask_b).all().item()
    return True


def get_at_least_one_mask(a, b):
    from maskedtensor import is_masked_tensor

    assert is_masked_tensor(a) or is_masked_tensor(b)
    assert masks_match(a, b)
    if is_masked_tensor(a):
        return get_mask(a)
    return get_mask(b)


def torch_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs, _wrap_result

    def binary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 2:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        if not masks_match(*args[:2]):
            raise ValueError(
                "Input masks must match. If you need support for this, please open an issue on Github."
            )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        result_data = fn(*data_args)
        result_mask = get_at_least_one_mask(*args[:2])
        result_mask = result_mask.expand_as(result_data)
        return _wrap_result(result_data, result_mask)

    return binary_fn


def torch_inplace_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs, _wrap_result

    def binary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 2:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        if not masks_match(*args[:2]):
            raise ValueError(
                "Input masks must match. If you need support for this, please open an issue on Github."
            )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        result_data = fn(*data_args)

        from maskedtensor import is_masked_tensor

        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]

    return binary_fn


NATIVE_BINARY_MAP = {
    getattr(torch.ops.aten, name): torch_binary(name) for name in BINARY_NAMES
}
NATIVE_INPLACE_BINARY_MAP = {
    getattr(torch.ops.aten, name): torch_inplace_binary(name)
    for name in INPLACE_BINARY_NAMES
}

NATIVE_BINARY_FNS = list(NATIVE_BINARY_MAP.keys())
NATIVE_INPLACE_BINARY_FNS = list(NATIVE_INPLACE_BINARY_MAP.keys())


def is_native_binary(fn):
    return fn in NATIVE_BINARY_FNS or fn in NATIVE_INPLACE_BINARY_FNS


def apply_native_binary(fn, *args, **kwargs):
    if fn in NATIVE_BINARY_FNS:
        return NATIVE_BINARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_BINARY_FNS:
        return NATIVE_INPLACE_BINARY_MAP[fn](*args, **kwargs)
    return NotImplemented
