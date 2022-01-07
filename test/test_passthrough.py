# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import pytest
import torch
import maskedtensor
from maskedtensor import masked_tensor
from maskedtensor.passthrough import PASSTHROUGH_FNS, apply_pass_through_fn
from maskedtensor.unary import NATIVE_UNARY_FNS, NATIVE_INPLACE_UNARY_FNS


def _get_test_data(fn_name):
    data = torch.randn(10, 10)
    mask = torch.rand(10, 10) > 0.5
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    if fn_name in ["log", "log10", "log1p", "log2", "sqrt"]:
        data = data.mul(0.5).abs()
    if fn_name in ["rsqrt"]:
        data = data.abs() + 1  # Void division by zero
    if fn_name in ["acos", "arccos", "asin", "arcsin", "logit"]:
        data = data.abs().mul(0.5).clamp(0, 1)
    if fn_name in ["atanh", "arctanh", "erfinv"]:
        data = data.mul(0.5).clamp(-1, 1)
    if fn_name in ["acosh", "arccosh"]:
        data = data.abs() + 1
    if fn_name in ["bitwise_not"]:
        data = data.mul(128).to(torch.int8)
    return data, mask


def _get_sample_kwargs(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    kwargs = {}
    if fn_name in ["clamp", "clip"]:
        kwargs["min"] = -0.5
        kwargs["max"] = 0.5
    return kwargs


def _get_sample_args(fn_name, data, mask):
    def _gen_dim_and_index(data):
        dim = random.randrange(data.dim())
        index = random.randrange(data.size(dim))
        return dim, index

    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    mt = masked_tensor(data, mask)
    t_args = [data]
    mt_args = [mt]
    args = []
    if fn_name in ["expand"]:
        args = [[-1, 4]]
    if fn_name in ["index"]:
        dim, index = _gen_dim_and_index(data)
        args = [torch.tensor([dim, index])]
    if fn_name in ["moveaxis", "movedim"]:
        args = [1, 0]
    if fn_name in ["narrow"]:
        dim, index = _gen_dim_and_index(data)
        args = [dim, 0, index]
    if fn_name in ["permute"]:
        args = [[1, 0]]
    if fn_name in ["reshape"]:
        args = [[2, int(data.numel()/2)]]
    if fn_name in ["select"]:
        dim, index = _gen_dim_and_index(data)
        args = [dim, index]
    if fn_name in ["slice"]:
        dim, index = _gen_dim_and_index(data)
        args = [dim, 0, index]
    # would be good to also test the sections parameter as well
    if fn_name in ["split"]:
        args = [[1, data.size(0)-1]]
    if fn_name in ["swapaxes", "swapdims", "transpose"]:
        args = [0, 1]
    if fn_name in ["take_along_dim"]:
        args = [torch.argmax(data)]
    if fn_name in ["take"]:
        args = [torch.tensor([0, 2, 5])]
    if fn_name in ["tile"]:
        args = [[2, 2]]
    if fn_name in ["unsqueeze"]:
        args = [1]
    if fn_name in ["view"]:
        args = [[int(data.numel())]]
    t_args += args
    mt_args += args
    return t_args, mt_args


def _compare_mt_t(mt_result, t_result):
    if isinstance(mt_result, list) and isinstance(t_result, list):
        for mt_ele, t_ele in zip(mt_result, t_result):
            _compare_mt_t(mt_ele, t_ele)
    
    else:
        mask = mt_result.masked_mask
        mt_result_data = mt_result.masked_data
        a = t_result.masked_fill_(~mask, 0)
        b = mt_result_data.masked_fill_(~mask, 0)
        assert torch.allclose(a, b)


@pytest.mark.parametrize("fn", PASSTHROUGH_FNS)
def test_passthrough(fn):
    torch.random.manual_seed(0)
    fn_name = fn.__name__
    data, mask = _get_test_data(fn_name)
    kwargs = _get_sample_kwargs(fn_name)

    t_args, mt_args = _get_sample_args(fn_name, data, mask)

    mt_result = apply_pass_through_fn(fn, *mt_args, **kwargs)
    t_result = fn(*t_args, **kwargs)
    _compare_mt_t(mt_result, t_result)

