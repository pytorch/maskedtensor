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

    def test_sum(self):
        d = torch.randn(3, 4, 2)
        m = d > 0
        mt = maskedtensor.masked_tensor(d, m)
        print(mt)
        print(mt.sum())
        print(mt.sum(dim=1))
        pass

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
        xinf = x.masked_fill(~m, float('-inf')).detach().clone().requires_grad_()
        tsinf = torch.softmax(xinf, -1)

    def test_bmm(self):
        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = maskedtensor.masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        x = x.masked_fill(~x_mt.mask(), 0)
        attn_2 = torch.bmm(x, x.transpose(-2, -1))
        attn_3 = torch.bmm(x_mt, x_mt.transpose(-2, -1))
        self.assertEqual(
                attn_3.masked_data.masked_fill(~attn_3.mask(), 0),
                attn_2)

    def test_bmm_2(self):
        x = torch.arange(3 * 2 * 2).reshape(3, 2, 2).float()
        x_t = x.transpose(-2, -1) + x.sum()
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = maskedtensor.masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        y = torch.bmm(x, x_t)
        y = torch.bmm(x, x_mt.transpose(-2, -1) + x.sum())

    def test_masked_bmm(self):
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x_mt = maskedtensor.masked_tensor(
            x,
            ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x)),
            requires_grad=True
        )
        print("x_mt: ", x_mt)
        attn_mask_bool = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        attn_mask = attn_mask_bool.float().masked_fill_(attn_mask_bool, float('-inf'))
        v = maskedtensor.masked_bmm(x, x_mt.transpose(1, 2), attn_mask)
        print("maskd v", v)
        v.sum().backward()
        print("maskd x.grad", x.grad)
        print("maskd x_mt.grad", x_mt.grad)
        print("-=-=-=-=-=-=-")
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x0 = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        y = torch.bmm(x, x0.transpose(-2, -1))
        y = y * (~attn_mask_bool).float()

        y.sum().backward()
        print("dense ", y.int())
        print("dense x.grad", x.grad)
        print("dense x0.grad", x0.grad)


    def test_linear(self):
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
        w_x = torch.arange(10).reshape(5, 2) + x.amax()
        linear = torch.nn.functional.linear
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x_mt = maskedtensor.masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

    def test_mha_issue_41508(self):
        # https://github.com/pytorch/pytorch/issues/41508
        # TODO:
        # 1. Restore matmul mask assert
        # 2. masked_bmm + grad
        # 3. Rename to masked_bmm
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
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        attn_mask = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        output, scores = attn_nn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        print("")
        print("0 scores")
        print(scores)
        loss0 = output[0, :].sum()
        print("0 loss")
        print(loss0)
        loss0.backward()
        print("0 grads")
        for n, p in attn_nn.named_parameters():
            print(0, n, p.grad)

        print("")
        x_mt = maskedtensor.masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

        output, scores = attn_mt(x, x_mt, x, attn_mask=attn_mask)
        print("1 scores")
        print(scores)
        loss1 = output[0, :].sum()
        print("1 loss")
        print(loss1)
        loss1.backward()
        print("1 grads")
        for n, p in attn_nn.named_parameters():
            print(1, n, p.grad)

        self.assertEqual(loss0, loss1.masked_data)


if __name__ == "__main__":
    unittest.main()
