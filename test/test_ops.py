from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import torch.nn.functional as F
import functools
import unittest
import itertools
from contextlib import contextmanager

from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import unary_ufuncs, binary_ufuncs

from maskedtensor import masked_tensor
from maskedtensor.unary import UNARY_NAMES
from maskedtensor.binary import BINARY_NAMES
from maskedtensor_additional_op_db import additional_op_db, create_mask

def is_unary(op):
    return op.name in UNARY_NAMES

def is_binary(op):
    return op.name in BINARY_NAMES

mt_unary_ufuncs = [op for op in unary_ufuncs if is_unary(op)]
mt_binary_ufuncs = [op for op in binary_ufuncs if is_binary(op)]

def _compare_mt_t(mt_result, t_result):
    mask = mt_result.masked_mask
    mt_result_data = mt_result.masked_data
    a = t_result.detach().masked_fill_(~mask, 0)
    b = mt_result_data.masked_fill_(~mask, 0)
    assert torch.allclose(a, b)

class TestOperators(TestCase):
    def test_native_masked_result_equality(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        for sample in samples:
            input = sample.input
            sample_args, sample_kwargs = sample.args, sample.kwargs
            if 'mask' not in sample_kwargs:
                mask = create_mask(input.shape, device)
            else:
                mask = sample_kwargs.pop('mask')

            # Binary operations currently only support same size masks
            if is_binary(op):
                if input.shape != sample_args[0].shape:
                    continue
                # Binary operations also don't support kwargs right now
                else:
                    sample_kwargs = {}

            mt = masked_tensor(input, mask)
            mt_args = [
                masked_tensor(arg, mask) if torch.is_tensor(arg)
                else arg
                for arg in sample_args
            ]

            mt_result = op(mt, *mt_args, **sample_kwargs)
            t_result = op(sample.input, *sample_args, **sample_kwargs)

            _compare_mt_t(mt_result, t_result)

            # If the operation is binary, check that lhs = masked, rhs = regular tensor also works
            if is_binary(op):
                mt_result2 = op(mt, *sample_args, **sample_kwargs)
                _compare_mt_t(mt_result2, t_result)

    @ops(mt_unary_ufuncs)
    def test_unary_core(self, device, dtype, op):
        self.test_native_masked_result_equality(device, dtype, op)

    @ops(mt_binary_ufuncs)
    def test_binary_core(self, device, dtype, op):
        self.test_native_masked_result_equality(device, dtype, op)

    @ops(additional_op_db, allowed_dtypes=(torch.float,))
    def test_maskedtensor_result(self, device, dtype, op):
        self.test_native_masked_result_equality(device, dtype, op)



only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
