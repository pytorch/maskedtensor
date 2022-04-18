# Copyright (c) Meta Platforms, Inc. and affiliates

import torch


def _compare_mt_t(mt_result, t_result):
    mask = mt_result.masked_mask
    mt_result_data = mt_result.masked_data
    if mask.layout == torch.sparse_coo:
        mask = mask.to_dense()
    if mt_result_data.layout == torch.sparse_coo:
        mt_result_data = mt_result_data.to_dense()
    a = t_result.detach().masked_fill_(~mask, 0)
    b = mt_result_data.masked_fill_(~mask, 0)
    assert torch.allclose(a, b)


def _compare_mts(mt1, mt2):
    assert mt1.layout == mt2.layout
    assert masks_match(mt1, mt2)
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
    assert torch.allclose(a, b)


def tensors_match(a, b):
    if a.layout == b.layout == torch.sparse_coo:
        a = a.to_dense()
        b = b.to_dense()
    return (a.dim() == b.dim()) and torch.eq(a, b).all().item()


def masks_match(a, b):
    from maskedtensor import is_masked_tensor

    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.masked_mask
        mask_b = b.masked_mask
        return tensors_match(mask_a, mask_b)
    return True
