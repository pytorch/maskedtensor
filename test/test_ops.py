import torch
from common_utils import _compare_mt_t
from maskedtensor import masked_tensor
from maskedtensor.binary import BINARY_NAMES
from maskedtensor.unary import UNARY_NAMES
from maskedtensor_additional_op_db import additional_op_db, create_mask
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs,
    unary_ufuncs,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def is_unary(op):
    return op.name in UNARY_NAMES


def is_binary(op):
    return op.name in BINARY_NAMES


mt_unary_ufuncs = [op for op in unary_ufuncs if is_unary(op)]
mt_binary_ufuncs = [op for op in binary_ufuncs if is_binary(op)]

MASKEDTENSOR_FLOAT_TYPES = {
    torch.float16,
    torch.float32,
    torch.float64,
}


def _test_native_masked_result_equality(device, dtype, op, is_sparse=False):
    samples = op.sample_inputs(device, dtype, requires_grad=True)

    for sample in samples:
        input = sample.input
        sample_args, sample_kwargs = sample.args, sample.kwargs
        if "mask" not in sample_kwargs:
            mask = create_mask(input.shape, device)
        else:
            mask = sample_kwargs.pop("mask")

        if is_sparse:
            input = input.to_sparse_coo()
            mask = mask.to_sparse_coo()

        # Binary operations currently only support same size masks
        if is_binary(op):
            if input.shape != sample_args[0].shape:
                continue
            # Binary operations also don't support kwargs right now
            else:
                sample_kwargs = {}

        mt = masked_tensor(input, mask)
        mt_args = [
            masked_tensor(arg.to_sparse_coo() if is_sparse else arg, mask)
            if torch.is_tensor(arg)
            else arg
            for arg in sample_args
        ]

        mt_result = op(mt, *mt_args, **sample_kwargs)
        t_result = op(sample.input, *sample_args, **sample_kwargs)

        _compare_mt_t(mt_result, t_result)

        # If the operation is binary, check that lhs = masked, rhs = regular tensor also works
        if is_binary(op):
            mt_result2 = op(mt, *mt_args, **sample_kwargs)
            _compare_mt_t(mt_result2, t_result)


class TestOperators(TestCase):
    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)
    def test_unary_core(self, device, dtype, op):
        # Skip tests that don't have len(kwargs) == 0
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        if op.name == "round" and op.variant_test_name in skip_variants:
            return
        _test_native_masked_result_equality(device, dtype, op)

    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)
    def test_binary_core(self, device, dtype, op):
        _test_native_masked_result_equality(device, dtype, op)

    @ops(additional_op_db, allowed_dtypes=(torch.float,))
    def test_maskedtensor_result(self, device, dtype, op):
        _test_native_masked_result_equality(device, dtype, op)

    @ops(mt_unary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)
    def test_unary_core_sparse(self, device, dtype, op):
        # Skip tests that don't have len(kwargs) == 0
        skip_variants = {
            "decimals_0",
            "decimals_3",
            "decimals_neg_3",
        }
        if op.name == "round" and op.variant_test_name in skip_variants:
            return
        _test_native_masked_result_equality(device, dtype, op, True)

    @ops(mt_binary_ufuncs, allowed_dtypes=MASKEDTENSOR_FLOAT_TYPES)
    def test_binary_core_sparse(self, device, dtype, op):
        _test_native_masked_result_equality(device, dtype, op, True)

    @ops(additional_op_db, allowed_dtypes=(torch.float,))
    def test_maskedtensor_results_sparse(self, device, dtype, op):
        _test_native_masked_result_equality(device, dtype, op, True)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
