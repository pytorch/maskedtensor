# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pytest
import torch
import maskedtensor
from maskedtensor import masked_tensor
from maskedtensor.unary import NATIVE_UNARY_FNS


@pytest.mark.parametrize("fn", NATIVE_UNARY_FNS)
def test_unary(fn):
    torch.random.manual_seed(0)
    data = torch.randn(10, 10)
    mask = torch.rand(10, 10) > 0.5

    kwargs = {}
    if fn.__name__ in ['log', 'log10', 'log1p', 'log2', 'sqrt']:
        data = data.mul(0.5).abs()
    if fn.__name__ in ['rsqrt']:
        data = data.abs() + 1 # Void division by zero
    if fn.__name__ in ['acos', 'arccos', 'asin', 'arcsin', 'logit']:
        data = data.abs().mul(0.5).clamp(0, 1)
    if fn.__name__ in ['atanh', 'arctanh', 'erfinv']:
        data = data.mul(0.5).clamp(-1, 1)
    if fn.__name__ in ['acosh', 'arccosh']:
        data = data.abs() + 1
    if fn.__name__ in ['bitwise_not']:
        data = data.to(torch.int8)
        # TODO: Print this

    if fn.__name__ in ['clamp', 'clip']:
        kwargs['min'] = -0.5
        kwargs['max'] = 0.5

    mt = masked_tensor(data, mask)
    t_args = [data]
    mt_args = [mt]
    if fn.__name__ in ['pow']:
        t_args += [2.]
        mt_args += [2.]
    # if len(kwargs) == 0:
    #     mt_result = fn(mt)
    #     t_result = fn(data)
    # else:
    mt_result = fn(*mt_args, **kwargs)
    t_result = fn(*t_args, **kwargs)

    mt_result_data = mt_result.masked_data
    a = t_result.masked_fill_(~mask, 0)
    b = mt_result_data.masked_fill_(~mask, 0)
    assert torch.allclose(a, b)
