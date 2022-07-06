import torch
from maskedtensor import masked_tensor

crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])

t_values = torch.tensor([1, 2, 3, 4])
m_values = torch.tensor([True, False, False, True])

t_csr = torch.sparse_csr_tensor(crow_indices, col_indices, t_values, dtype=torch.double)
m_csr = torch.sparse_csr_tensor(crow_indices, col_indices, m_values, dtype=torch.bool)

# i = [[0, 1, 1],
#      [2, 0, 2]]
# v =  [3, 4, 5]
# m =  [False, True, False]

# t_csr = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.double)
# m_csr = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.bool)

print ("t", t_csr.to_dense())
print ("m", m_csr.to_dense())

mt = masked_tensor(t_csr, m_csr, requires_grad=True)
print ("mt", mt)
print ("mt data", mt.masked_data)
print ("to dense", mt.to_dense())
# y = masked_tensor(t_csr.to_dense(), m_csr.to_dense())
# y = mt.layout()
# print ("y", y)
# print ("y data", y.masked_data)
