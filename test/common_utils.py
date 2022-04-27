# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from maskedtensor.core import _masks_match, _tensors_match
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.testing._internal.common_utils import make_tensor


def _compare_mt_t(mt_result, t_result):
    mask = mt_result.masked_mask
    mt_result_data = mt_result.masked_data
    if mask.layout == torch.sparse_coo:
        mask = mask.to_dense()
    if mt_result_data.layout == torch.sparse_coo:
        mt_result_data = mt_result_data.to_dense()
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    assert _tensors_match(a, b)


def _compare_mts(mt1, mt2):
    assert mt1.masked_data.layout == mt2.masked_data.layout
    assert mt1.masked_mask.layout == mt2.masked_mask.layout
    assert _masks_match(mt1, mt2)
    mask = mt1.masked_mask
    mt_data1 = mt1.masked_data
    mt_data2 = mt2.masked_data
    if mask.layout == torch.sparse_coo:
        mask = mask.to_dense()
    if mt_data1.layout == torch.sparse_coo:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)
    assert _tensors_match(a, b)


def _create_random_mask(shape, device):
    return make_tensor(
        shape, device=device, dtype=torch.bool, low=0, high=1, requires_grad=False
    )


def _generate_sample_data(
    device="cpu", dtype=torch.float, requires_grad=True, sparse=False
):
    shapes = [
        [],
        [2],
        [3, 5],
        [3, 2, 1, 2],
    ]
    inputs = []
    for s in shapes:
        data = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        mask = _create_random_mask(s, device)
        if sparse:
            mask = mask.to_sparse_coo()
            data = data.sparse_mask(mask)
        inputs.append(SampleInput(data, kwargs={"mask": mask}))
    return inputs
