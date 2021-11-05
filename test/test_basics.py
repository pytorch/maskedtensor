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


if __name__ == "__main__":
    unittest.main()
