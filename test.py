import torch
print(torch.__version__)
from torch.testing._internal.common_utils import gradcheck
from maskedtensor import masked_tensor

# t = torch.tensor([True, True, True])
# mask = torch.tensor([True, True, True])
# print(t.sum())
# print(t.sum().item())
# print(masked_tensor(t, mask).sum())
# print(masked_tensor(t, t).sum().item())
# print(masked_tensor(t, mask).to(torch.float))

# t = torch.tensor([5,5,5])
# m = torch.tensor([True, False, True])
# mt = masked_tensor(t, m)
# print(mt)
# print(mt.to_dense())

# device = 'cpu'
# i = torch.tensor([
#     [0, 1, 2, 2],
#     [0, 0, 0, 3],
#     [0, 0, 1, 4],
# ], device=device)
# v = torch.tensor([2, 1, 3, 4], dtype=torch.float64, device=device)
# x = torch.sparse_coo_tensor(i, v, dtype=torch.double, device=device)
# mask = torch.randint(low=0, high=2, size=(3,4,5)).bool()
# mt = masked_tensor(x.clone().detach().to_dense().requires_grad_(True), mask, requires_grad=True)
# x.requires_grad_(True)
# print ("mt", mt)
# print ("sparse tensor", x, x.dtype)

# def fn(x):
#     return x.to_dense()
# def fn2(x):
#     return x.to_dense().masked_data

# # gradcheck(fn, (x,), check_sparse_nnz=True, check_batched_grad=False)
# gradcheck(fn2, (mt,), check_batched_grad=False)

# device='cpu'
# tensor = torch.randn(3, dtype=torch.float64, requires_grad=True)
# tensor2 = tensor.clone().detach().requires_grad_(True)
# tensor3 = tensor.clone().detach().requires_grad_(True)

# mask = torch.ones(3, device=device)
# mask[1] = 0
# ms = mask.to_sparse()

# mt = masked_tensor(tensor.clone().detach().requires_grad_(True), mask.bool(), requires_grad=True)

# i = torch.tensor([
#     [0, 1, 2, 2],
#     [0, 0, 0, 3],
#     [0, 0, 1, 4],
# ], device=device)
# # we don't have to_dense for half types on CPU because it is implemented
# # with a slower add_ operation
# v = torch.tensor([2, 1, 3, 4], dtype=torch.float64, device=device)
# x = torch.sparse_coo_tensor(i, v, dtype=torch.double, device=device)
# x.requires_grad_(True)
# print ("sparse tensor", x, x.dtype)

# def fn(x):
#     return x.to_dense()

# # print ("grad check 1", gradcheck(fn, (tensor.to_sparse(),), check_sparse_nnz=True, fast_mode=False))
# print ("grad check 2", gradcheck(fn, (x,), check_sparse_nnz=True))

# device = 'cpu'

# i = torch.tensor([
#     [0, 1, 2, 2],
#     [0, 0, 0, 3],
#     [0, 0, 1, 4],
# ], device=device)
# v = torch.tensor([2, 1, 3, 4], dtype=torch.float64, device=device)
# x = torch.sparse_coo_tensor(i, v, torch.Size([3, 4, 5]), dtype=torch.double, device=device)

# mask = torch.randint(0, 2, (3,4,5)).bool()
# ms = mask.to_sparse_coo()

# t1 = x.clone().detach().to_dense().requires_grad_(True)
# t1s = x.clone().detach().requires_grad_(True)
# mt = masked_tensor(t1, mask, requires_grad=True)
# mts = masked_tensor(t1s, ms, requires_grad=True)

# print ("mt", mt)
# print ("mts", mts)

# # converted2 = mt.to_sparse().to_dense()
# # converted2.sum().backward()
# # print("converted4", mt.layout, mt.grad.layout, mt.grad, mt.grad.layout, mt.grad.masked_data)
# # print("done")

# converted3 = mts.to_dense()
# s = converted3.sum()
# print ("converted3 sum", s)
# s.backward()
# print ("before converted")
# print ("converted5", mts.grad, mts.grad.layout, mts.grad.masked_data)

# # mts = mt.to_sparse().to_dense()
# # print ("doing backward now..")
# # mts.sum().backward()
# # print ("tensor grad", tensor.grad)
# # print ("mt grad", mt.grad, mt.grad.masked_data)
# # print ("mts grad", mts.grad)
# # print ("mask", mask.to_dense())

# ** torch dispatch aten._indices
# ** torch dispatch aten._values
# ** torch dispatch aten._indices
# ** torch dispatch aten._values
# ** torch dispatch aten._values
# ** torch dispatch aten._indices
# before converted
# ** torch function <method-wrapper '__get__' of property object at 0x7fb37d941950>
# ** torch function <method-wrapper '__get__' of property object at 0x7fb37d941950>
# ** torch function <method-wrapper '__get__' of property object at 0x7fb37d941950>


device = 'cpu'
i = torch.tensor([
    [0, 1, 2, 2],
    [0, 0, 0, 3],
    [0, 0, 1, 4],
], device=device)
v = torch.tensor([2, 1, 3, 4], dtype=torch.float64, device=device)
x = torch.sparse_coo_tensor(i, v, torch.Size([3, 4, 5]), dtype=torch.double, device=device)

mask = torch.randint(0, 2, (3, 4, 5)).bool()
ms = mask.to_sparse_coo()

mts = masked_tensor(x, ms, requires_grad=True)
mts.to_dense().sum().backward()
