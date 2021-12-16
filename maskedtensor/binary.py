import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.overrides import get_default_nowrap_functions

BINARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

INPLACE_BINARY_NAMES = [
    n + "_" for n in (list(set(BINARY_NAMES) - set(["angle", "positive", "signbit", "isnan"])))
]

# Explicitly tracking functions we know are currently not supported
# This might be due to missing code gen or because of complex semantics
BINARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]


def torch_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs, _wrap_result

    def binary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 1:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        result_data = fn(*data_args)
        return _wrap_result(result_data, mask_args[0])

    return binary_fn


def torch_inplace_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)
    from .passthrough import _map_mt_args_kwargs, _wrap_result

    def binary_fn(*args, **kwargs):
        assert len(kwargs) == 0
        if len(args) > 1:
            for a in args[1:]:
                assert not torch.is_tensor(a)
        mask_args, mask_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_mask
        )
        data_args, data_kwargs = _map_mt_args_kwargs(
            args, kwargs, lambda x: x.masked_data
        )
        result_data = fn(*data_args)
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
