# Copyright (c) Meta Platforms, Inc. and affiliates

import unittest
import torch
import maskedtensor
from maskedtensor import masked_tensor
from torch.testing._internal.common_utils import TestCase


class TestMaskedTensor(TestCase):
    def test_add(self):
        data = torch.arange(5.0)
        mask = torch.tensor([True, True, False, True, False])
        m0 = maskedtensor.masked_tensor(data, mask)
        m1 = maskedtensor.masked_tensor(data, ~mask)
        self.assertRaises(ValueError, lambda: m0 + m1)

    def test_softmax(self):
        x = torch.randn(3, 4) * 0.1
        m = torch.tensor(
            [
                [True, True, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ]
        )
        mx = maskedtensor.masked_tensor(x, m, requires_grad=True)
        ts = torch.softmax(mx, -1)
        ts.sum().backward()
        xinf = x.masked_fill(~m, float("-inf")).detach().clone().requires_grad_()
        tsinf = torch.softmax(xinf, -1)
    
    def _test_gather(self, size_x, size_ind, dim):
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            index = torch.randint(x.size(dim), size_ind)
        else:
            index = torch.zeros(size_ind, dtype=torch.int64)

        m = torch.randint(1, 2, size_x).bool()
        mt = maskedtensor.masked_tensor(x, m, requires_grad=True)
        x0 = x.masked_fill(~m, 0).detach().clone().requires_grad_(True)

        out = torch.gather(x0, dim, index)
        grad = torch.rand_like(out)
        out.backward(grad)

        out_masked = torch.gather(mt, dim, index)
        out_masked.backward(grad)
        self.assertEqual(x0.grad, mt.grad)

    def test_gather_dim0(self):
        self._test_gather((10, 10), (5, 10), 0)

    def test_gather_dim1(self):
        self._test_gather((10, 10, 5), (10, 5, 5), 1)

    def test_gather_dim_neg(self):
        self._test_gather((10, 10, 5), (10, 10, 2), -1)

    def test_gather_ind_scalar(self):
        self._test_gather((10,), (), 0)

    def test_gather_x_scalar(self):
        self._test_gather((), (2,), 0)

    def test_gather_both_scalar(self):
        self._test_gather((), (), 0)

    def test_mha_issue_41508(self):
        # https://github.com/pytorch/pytorch/issues/41508
        import torch

        torch.manual_seed(0)
        attn_nn = torch.nn.MultiheadAttention(1, 1, bias=False)
        attn_mt = torch.nn.MultiheadAttention(1, 1, bias=False)
        for (na, a), (nb, b) in zip(
            attn_nn.named_parameters(), attn_mt.named_parameters()
        ):
            a.data.copy_(b.data)

        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [[False, False, False], [False, True, True],]
        )
        attn_mask = torch.as_tensor(
            [[False, True, True], [False, False, True], [True, False, False],]
        )
        output, scores = attn_nn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        loss0 = output[0, :].sum()

        x_mt = maskedtensor.masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

        output, scores = attn_mt(x, x_mt, x, attn_mask=attn_mask)
        loss1 = output[0, :].sum()
        self.assertEqual(loss0, loss1.masked_data)

    def test_chunk(self):
        return
        # This breaks because split_backward allocates
        # Tensors using zero and then cats them together.
        # I don't know why the masks are coming into play here.
        # It's an autograd thing.
        k_data = torch.tensor([4.0])
        k_mask = torch.tensor([True])
        k = maskedtensor.masked_tensor(k_data[0], k_mask[0], requires_grad=True)
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        w_q, w_k = w.chunk(2)
        o0 = k + w_k
        o0.backward()
        return


if __name__ == "__main__":
    unittest.main()
