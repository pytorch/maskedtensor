import torch
from maskedtensor import masked_tensor

torch.manual_seed(1)

shape = (2,3)
t1 = torch.randn(shape) * -1
t2 = torch.arange(6).reshape(shape) + 1
mask = torch.randint(0, 2, shape).bool()

m1 = masked_tensor(t1, mask)
m2 = masked_tensor(t2, mask)
print (m1)
print (m2)

s1 = t1.to_sparse_coo()
s2 = t2.to_sparse_coo()
sparse_mask = mask.to_sparse_coo()
print ("s1", s1)
print ("s2", s2)
print ("mask", mask)
print ("sparse mask", sparse_mask)

m3 = masked_tensor(s1, sparse_mask)
m4 = masked_tensor(s2, sparse_mask)

print ("m3!!", m3)
print ("m4!!", m4)

print("abs 1", torch.abs(m1))
x = torch.cos(m3)
print ("cos of m3", x.layout, x)

x2 = torch.add(m3, m4)
print ("add of m3+m4", x2.layout, x2)
