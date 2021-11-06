import unittest
import torch
import maskedtensor
from torch.testing._internal.common_utils import TestCase


class TestMaskedTensor(TestCase):
    def test_add(self):
        data = torch.arange(5.0)
        mask = torch.tensor([True, True, False, True, False])
        m0 = maskedtensor.masked_tensor(data, mask)
        m1 = maskedtensor.masked_tensor(data, ~mask)
        self.assertRaises(ValueError, lambda: m0 + m1)

    def test_mha(self):
        mha_nn = torch.nn.MultiheadAttention(64, 4, bias=False)
        mha_mt = maskedtensor.MultiheadAttention(64, 4, bias=False)
        for (na, a), (nb, b) in zip(
            mha_nn.named_parameters(), mha_mt.named_parameters()
        ):
            a.data.copy_(b.data)
        N = 4
        S = 8
        E = 64
        q = torch.randn(N, S, E)
        k, v = q.clone(), q.clone()
        output_nn, _ = mha_nn(q, k, v)
        output_mt, _ = mha_mt(q, k, v)
        self.assertEqual(output_nn, output_mt)

    def test_mha_issue_41508(self):
        # https://github.com/pytorch/pytorch/issues/41508
        # TODO: 
        # 0. attn_mask in bool
        # 1. Restore matmul mask assert
        # 2. masked_matmul + grad
        # 3. ...
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
            [[False, False, False], [False, True, True]], dtype=torch.bool
        )
        attn_mask = torch.as_tensor(
            [
                [0.0, float("-inf"), float("-inf")],
                [0.0, 0.0, float("-inf")],
                [float("-inf"), 0.0, 0.0],
            ]
        )
        attn_mask = None
        output, scores = attn_nn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        print("")
        print("0 scores")
        print(scores)
        loss = output[0, :].sum()
        print("0 loss")
        print(loss)
        loss.backward()
        print("0 grads")
        for n, p in attn_nn.named_parameters():
            print(0, n, p.grad)

        print("")
        # print("x.shape: ", x.shape)
        # print("key_padding_mask.shape: ", key_padding_mask.shape)
        x_mt = maskedtensor.masked_tensor(x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x)))

        output, scores = attn_mt(
            x, x_mt, x, attn_mask=attn_mask
        )
        print("1 scores")
        print(scores)
        loss = output[0, :].sum()
        print("1 loss")
        print(loss)
        loss.backward()
        print("1 grads")
        for n, p in attn_nn.named_parameters():
            print(1, n, p.grad)


if __name__ == "__main__":
    unittest.main()
