import torch
from maskedtensor import masked_tensor

t = torch.randn(5, 3)
mask = torch.randint(0, 2, (5, 3)).bool()

m_csr = mask.to_sparse_csr()
t_csr = t.sparse_mask(m_csr)

m_coo = mask.to_sparse_coo()
t_coo = t.sparse_mask(m_coo)

mt2 = masked_tensor(t_csr, m_csr, requires_grad=True)

mt3 = mt2.to_dense()  # fails
print ("Mt3", mt3)
