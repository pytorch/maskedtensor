import unittest
import torch
from maskedtensor import masked_tensor
from torch.testing._internal.common_utils import TestCase


class TestMaskedTensorReductions(TestCase):
    def test_not_implemented(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m)
        self.assertRaises(TypeError, lambda: mt.max())

    def test_sum(self):
        d = torch.tensor([[0, 1, 2, 6], [3, 4, 5., 7]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(17.),
                                       torch.tensor(True)),
                         mt.sum())
        self.assertEqual(masked_tensor(torch.tensor([0., 4., 1., 13]),
                                       torch.tensor([True, True, False, True])),
                         mt.sum(dim=0))

    def test_sum_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.sum().backward()
        self.assertEqual(mt.grad, masked_tensor(torch.tensor(1.).expand_as(m), m))

    def test_mean(self):
        d = torch.tensor([[0, 1, 3, 2], [3, 4, 1., 4]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(2.5),
                                       torch.tensor(True)),
                         mt.mean())
        self.assertEqual(masked_tensor(torch.tensor([0., 4., 1., 3]),
                                       torch.tensor([True, True, False, True])),
                         mt.mean(dim=0))

    def test_mean_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.mean().backward()
        self.assertEqual(mt.grad, masked_tensor(torch.tensor(1.).expand_as(m), m))

    def test_amax(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1., 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(3.),
                                       torch.tensor(True)),
                         mt.amax())
        self.assertEqual(masked_tensor(torch.tensor([0., -4., 1., 3]),
                                       torch.tensor([True, True, False, True])),
                         mt.amax(dim=0))

    def test_amax_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amax().backward()
        self.assertEqual(mt.grad, masked_tensor(torch.tensor(1.).expand_as(m), m))

    def test_amin(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1., 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(-4.),
                                       torch.tensor(True)),
                         mt.amin())
        self.assertEqual(masked_tensor(torch.tensor([0., -4., 1., -3]),
                                       torch.tensor([True, True, False, True])),
                         mt.amin(dim=0))

    def test_amin_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amin().backward()
        self.assertEqual(mt.grad, masked_tensor(torch.tensor(1.).expand_as(m), m))

    def test_prod(self):
        d = torch.tensor([[0, 1, 3, 0.], [float('nan'), 4, 1., 5.]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(0.),
                                       torch.tensor(True)),
                         mt.prod())
        self.assertEqual(masked_tensor(torch.tensor([0., 4., 1., 0.]),
                                       torch.tensor([True, True, False, True])),
                         mt.prod(dim=0))

    def test_prod_grad(self):
        d = torch.tensor([[0, float('nan'), 2], [3, 4, 5.]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.prod().backward()
        self.assertEqual(mt.grad, masked_tensor(torch.tensor(1.).expand_as(m), m))

    def test_all(self):
        d = torch.tensor([[True, True, False, False], [False, True, True, True]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor(False),
                                       torch.tensor(True)),
                         mt.all())
        self.assertEqual(masked_tensor(torch.tensor([True, True, True, False]),
                                       torch.tensor([True, True, False, True])),
                         mt.all(dim=0))

        m = torch.tensor([[True, False, True, False], [False, True, False, False]])
        mt = masked_tensor(d, m)
        self.assertEqual(masked_tensor(torch.tensor([True, True, False, True]),
                                       torch.tensor([True, True, True, False])),
                         mt.all(dim=0))

    def test_all_grad(self):
        d = torch.tensor([[True, True, False], [False, True, True]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        self.assertRaises(RuntimeError, lambda: masked_tensor(d, m, requires_grad=True))


if __name__ == "__main__":
    unittest.main()
