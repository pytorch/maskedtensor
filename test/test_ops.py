from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import torch.nn.functional as F
import functools
import unittest
import itertools
from contextlib import contextmanager
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_dtype import integral_types
from torch.testing._internal.common_device_type import \
     toleranceOverride, tol

from maskedtensor_additional_op_db import additional_op_db
from test_unary import _get_test_data, _get_sample_args, _get_sample_kwargs, _compare_mt_t

from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
import torch.autograd.forward_ad as fwAD


class TestOperators(TestCase):
    @ops(additional_op_db, allowed_dtypes=(torch.float,))
    def test_maskedtensor_result(self, device, dtype, op):
        fn_name = op.name
        data, mask = _get_test_data(fn_name)
        kwargs = _get_sample_kwargs(fn_name)

        t_args, mt_args = _get_sample_args(fn_name, data, mask)

        mt_result = op(*mt_args, **kwargs)
        t_result = op(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
