---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "JF8WDyTRq0Hn"}

# Overview of MaskedTensors

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/pytorch/maskedtensor/tree/main/docs/notebooks/overview.ipynb)

```{code-cell} ipython3
import torch
import numpy as np
if "1.11.0" not in torch.__version__:
    !pip uninstall --y torch
    !pip install torch -f https://download.pytorch.org/whl/test/cu102/torch_test.html --pre
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 2o4aL_UWd8xR
outputId: 2d81e664-a8ab-4f9a-e823-0fa5aef2b8ab
---
!pip install -i https://test.pypi.org/simple/ maskedtensor
```

```{code-cell} ipython3
:id: XHPLFh2Qf4ZL

# Import factory function
from maskedtensor import masked_tensor
from maskedtensor import as_masked_tensor
```

+++ {"id": "aSD_zzXcWLvK"}

## Basic masking semantics

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FrNr_-yfjcsr
outputId: b1b03759-1e8c-4cf3-bdae-0be4900ee8ea
---
# First example of addition
data = torch.arange(5.)
mask = torch.tensor([True, True, False, True, False])
m0 = masked_tensor(data, mask)
m1 = masked_tensor(data, ~mask)
print(m0)
print(m1)
print(torch.cos(m0))
print(m0 + m0)
try:
  # For now the masks must match. We treat them like shapes.
  # We can relax this later on, but should have a good reason for it.
  # We'll revisit this once we have reductions.
  print(m0 + m1)
except ValueError as e:
  print(e)
```

+++ {"id": "RMHT1RebL8PR"}

NumPy's MaskedArray implements intersection semantics here. If one of two elements are masked out the resulting element will be masked out as well. Note that MaskedArray's factory function inverts the mask (similar to torch.nn.MHA). For MaskedTensor we'd apply the logical_and operator to both masks during a binary operation to get the semantics NumPy has. Since NumPy stores the inverted mask they [apply the logical_or operator](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1016-L1024). But to repeat this point we suggest to not support addition between MaskedTensors with masks that don't match. See the section on reductions for why we should have good reasons for this.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -tmWxZ9NKEgE
outputId: 5997cae5-f4c4-4422-b6d1-c80fd59c4001
---
npm0 = np.ma.masked_array(data.numpy(), (~mask).numpy())
npm1 = np.ma.masked_array(data.numpy(), (mask).numpy())
print(npm0)
print(npm1)
print(npm0 + npm1)
```

+++ {"id": "SljV_QCfMEu7"}

MaskedTensor also supports these semantics by giving access to the masks and conveniently converting a MaskedTensor to a Tensor with masked values filled in with a particular value.

NumPy of course has the opportunity to avoid addition altogether in this case by check whether any results are not masked, but [chooses not to](https://github.com/numpy/numpy/blob/68299575d8595d904aff6f28e12d21bf6428a4ba/numpy/ma/core.py#L1013). Presumably it's more expensive to allreduce the mask every time to avoid the binary addition of the data in this case.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: D-TCtDEJJzeV
outputId: de7bd486-4a0f-4645-8c38-63b5b16eb55f
---
m0t = m0.to_tensor(0)
m1t = m1.to_tensor(0)

m2t = masked_tensor(m0t + m1t, m0.mask() & m1.mask())
print(m0t)
print(m1t)
print(m2t)
```

+++ {"id": "zAbRZK3QZgge"}

Example of printing a 2d MaskedTensor and setup for reductions below

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: y9tv_SP8oI7Z
outputId: 006fbfc2-a59c-4205-919e-d866203aa840
---
data = torch.randn(8, 3).mul(10).int().float()
mask = torch.randint(2, (8, 3), dtype=torch.bool)
print(data)
print(mask)
m = masked_tensor(data, mask)
print(m)
```

+++ {"id": "8fUbL3yAZqZF"}

Reduction semantics based on https://github.com/pytorch/rfcs/pull/27

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: M2wGD5hRpVDV
outputId: f5867cef-70b6-427b-c4bc-8fc066a98469
---
print(torch.sum(m, 1))
print(torch.mean(m, 1))
print(torch.prod(m, 1))
print(torch.amin(m, 1))
print(torch.amax(m, 1))
```

+++ {"id": "TyLv4Nf4dVtS"}

Now that we have reductions, let's revisit as to why we'll probably want to have a good reason to allow addition of MaskedTensors with different masks.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xMPyMU87fICJ
outputId: 910e01fd-0da9-4e56-f91e-4a2087b4377b
---
data0 = torch.arange(10.).reshape(2, 5)
data1 = torch.arange(10.).reshape(2, 5) + 10
mask0 = torch.tensor([[True, True, False, False, False], [False, False, False, True, True]])
mask1 = torch.tensor([[False, False, False, True, True], [True, True, False, False, False]])

npm0 = np.ma.masked_array(data0.numpy(), (mask0).numpy())
npm1 = np.ma.masked_array(data1.numpy(), (mask1).numpy())
print("\nnpm0:\n", npm0)
print("\nnpm1:\n", npm1)
print("\n(npm0 + npm1).sum(0):\n", (npm0 + npm1).sum(0))
print("\nnpm0.sum(0) + npm1.sum(0):\n", (npm0.sum(0) + npm1.sum(0)))
print("\n(data0 + data1).sum(0):\n", (data0 + data1).sum(0))
print("\n(data0 + data1).sum(0):\n", (data0.sum(0) + data1.sum(0)))
```

+++ {"id": "FjSyaqqKgvYh"}

Sum and addition should be associative. However with NumPy's semantics we allow them not to be. Instead of allowing these semantics, at least in the case of addition and sum, we could ask the user to fill the MaskedTensor's undefined elements with 0 values or as in the MaskedTensor addition examples above be very specific about the semantics used. 

While it's obviously possible to support this, I think we should cover other operators first and really make sure we can't avoid this behavior via other means.

+++ {"id": "ADp2guJ6ZlMo"}

Now let's print some higher dimensional MaskedTensors.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c3ESKLl0pYEj
outputId: 41f3d277-177a-484a-9ae1-f89ef552a8d9
---
data = torch.randn(4, 5, 3).mul(5).float()
mask = torch.randint(2, (4, 5, 3), dtype=torch.bool)
m = masked_tensor(data, mask)
print(m)
```

+++ {"id": "LgHrnMB7ZtOo"}

Example of indexing and advanced indexing

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: RWGG2AA1r_R-
outputId: 20ad0002-bb61-4a3a-c700-541ce229718d
---
print(m[0])
print(m[torch.tensor([0, 2])])
print(m[m.mask()])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qEejo-1-sBMw
outputId: edd0d800-2c7c-4ea3-f67f-bbce070fc044
---
torch.manual_seed(22)
# Sum needs custom autograd, since the mask of the input should be maintained
data = torch.randn(2, 2, 3).mul(5).float()
mask = torch.randint(2, (2, 2, 3), dtype=torch.bool)
m = masked_tensor(data, mask, requires_grad=True)
print(m)
s = torch.sum(m)
print("s: ", s)
s.backward()
print("m.grad: ", m.grad)

# sum needs to return a scalar MaskedTensor because the input might be fully masked
data = torch.randn(2, 2, 3).mul(5).float()
mask = torch.zeros(2, 2, 3, dtype=torch.bool)
m = masked_tensor(data, mask, requires_grad=True)
print("\n", m)
s = torch.sum(m)
print("s: ", s)
s.backward()
print("m.grad: ", m.grad)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 3N1Y7QFJMrdz
outputId: da5f6016-23cb-46e2-d949-ea145aea0037
---
# Grad of multiplication of MaskedTensor and Tensor
x = masked_tensor(torch.tensor([3.0, 4.0]), torch.tensor([True, False]), requires_grad=True)
print("x:\n", x)
y = torch.tensor([2., 1.]).requires_grad_()
print("y:\n", y)
# The mask broadcast in the sense that the result is masked.
# In general a MaskedTensor is considered a generalization of Tensor's shape.
# The mask is a more complex, higher dimensional shape and thus the Tensor
# broadcasts to it. I'd love to find a more rigorous definition of this.
z = x * y
print("x * y:\n", z)
z.sum().backward()
print("\nx.grad: ", x.grad)
# The regular torch.Tensor now has a MaskedTensor grad
print("y.grad: ", y.grad)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ITWFisob5xNG
outputId: f6a6c891-212c-411d-923b-9d77fc5bcb4c
---
# is_contiguous doesn't work
t = torch.arange(4).reshape(1, 2, 2).float()
mask = (t > 0).bool().clone()
t = t.clone()
mt = masked_tensor(t, mask)
mt = mt.view(mt.size())
mt = mt.transpose(0, 1)
print(mt.is_contiguous(), mt.size(), mt.stride())
print(mt.masked_data.is_contiguous(), mt.masked_data.size(), mt.masked_data.stride())
mt = mt.view(mt.size())
print(mt.is_contiguous(), mt.size(), mt.stride())
mt = mt.contiguous()
print(mt.is_contiguous(), mt.size(), mt.stride())
```

```{code-cell} ipython3
:id: kHqeS_pwvQkD

# Because .contiguous doesn't work we need to modify view to use reshape instead
mask = (t > 0).bool().clone()
t = t.clone()
mt = masked_tensor(t, mask, requires_grad=True)
mt = mt.view([4])
mt.sum().backward()
```

+++ {"id": "bPI00yqwWQ-j"}

## Resolving or revisiting some issues
In some cases MaskedTensors can provide a resolution, in others it can provide an alternative or best case more principled approach.

+++ {"id": "7VQSfHpU5LvV"}

### [1369](https://github.com/pytorch/pytorch/issues/1369)
This issue discussed the additional lines of code that were introduce while writing "sparse" semantics for Adagrad. But really the code doesn't use sparsity as a compression and optimization technique, it wants to use masked semantics. We worked around this by introducing one-off semantics and operators that encode this behavior while forcing users to be aware of storage details such as indices and values. Let's look at the current implementation of [Adagrad](https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py) [(functional)](https://github.com/pytorch/pytorch/blob/6c2f235d368b697072699e5ca9485fd97d0b9bcc/torch/optim/_functional.py#L16-L51) to illustrate that.

In particular we'll point out when sparsity is used as a semantic extension, i.e. unspecified values are not zero and when it is just used to compress zeros. We'll also compare and contrast this with equivalent code written using MaskedTensor. In the end the code snippets are repeat without additional comments to show the difference in brevity.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: DS8rsWq15GbB
outputId: 94ecd1a8-df2f-41ef-b5f0-05f4dc7ee3b3
---
def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

# We don't support sparse gradients
param = torch.arange(8).reshape(2, 4).float()
i = torch.tensor([[0, 1, 1],
                  [2, 0, 2]])
v = torch.tensor([3, 4, 5], dtype=torch.float32)
grad = torch.sparse_coo_tensor(i, v, [2, 4])
state_sum = torch.full_like(param, 0.5) # initial value for state sum

print("param:\n", param)
print("grad:\n", grad.to_dense())
print("state_sum:\n", state_sum)

# Some hyperparameters
eps = 1e-10
clr = 0.1
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1G4c6XpQAGcA
outputId: 31335fe7-7f11-4e32-bb31-97b1ab776609
---
state_sum = torch.full_like(param, 0.5) # initial value for state sum
print(state_sum)

grad = grad.coalesce()  # the update is non-linear so indices must be unique
grad_indices = grad._indices()
grad_values = grad._values()
size = grad.size()

# pow(2) has the same semantics for both sparse and dense memory layouts since
# 0^2 is zero
state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
# We take care to make std sparse, even though state_sum clearly is not.
# This means that we're only applying the gradient to parts of the state_sum
# for which it is specified. This even drives the point home a lot more that
# the passed gradient is not sparse, but masked. 
std = state_sum.sparse_mask(grad)
print("state_sum:\n", state_sum)
print("std:\n", std.to_dense())
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FGiIbAHYTX8K
outputId: 192ec140-1d90-4ef5-abd1-d2d7668bd31d
---
# This is where we have a very important divergence. The addition of eps
# should technically be applied to all values, but instead is only applied to
# specified values. Here we're using sparsity as a semantic extension and
# to enforce a certain pattern of defined and undefined values. If parts
# of the values of the gradient are zero they are still included if materialized.
# Even though they could be compressed by other sparse storage layouts.
# This is technically quite brittle even though someone could argue that eps is
# always very small.

# More so an implementation add_ for sparsity as a storage layout and compression
# scheme should cause densification, but we force it not to. For this one-off
# case it is fine until we want to introduce new compression schemes such as
# CSR, BSR or 2:4 block sparsity. We'll then need to introduce separate Tensor
# types for each and write variations for gradients compressed using different
# storage formats.

# We currently dodge all these concerns using the private method values.
std_values = std._values().sqrt_().add_(eps)

# We currently don't support div for sparse Tensors because zero / zero is
# not well defined. For a MaskedTensor undefined / undefined is undefined.
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
print("param:\n", param)
```

+++ {"id": "jKgg43BY_SQG"}

We've been conflating sparsity as an optimization with sparsity as a semantic extension to PyTorch. MaskedTensor proposes to call the semantic extension through sparsity masked. Currently we can't have dense semantics with sparse storage or masked semantics with dense storage. MaskedTensor fixes that because it separates the storage from the semantics. Let's look at above example using a masked gradient.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: hDJl-NCHScL5
outputId: cd976bf5-3074-44ad-a368-6b878e7ba84e
---
# Of course we can add sparsity as a storage layout for MaskedTensor which
# compresses repeated undefined values. We can recycle SparseTensor and SparseCSR
# by setting data and mask to an instance of each that share indices.
# However, ideally we'd just have regular torch.Tensors with a sparse layout
# and use those to back MaskedTensor.
masked_grad = masked_tensor(grad.to_dense(), grad.to_dense() != 0)
print("masked_grad:\n", masked_grad)
```

```{code-cell} ipython3
:id: yljgf492EuAj

# Create an entirely new set of parameters to avoid errors
param2 = torch.arange(8).reshape(2, 4).float()
state_sum2 = torch.full_like(param, 0.5) # initial value for state sum
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 4zkyzQU4Pccn
outputId: 9d1f19d6-1038-4343-adb7-c1f76a1e8a49
---
# This is an excellent example of why to_tensor is important. We don't
# want to propagate the mask to state_sum, but still maintain the compression.
# to_tensor could eventually return a Tensor with sparse layout for the
# special value of zero or first require explicit densification if it can't
# maintain the layout.

# This is also a value proposition for sparsity
# as a separate layout and a SparseTensor with dense semantics. MaskedTensor
# can be much simpler without having to introduce complex maske union and intersection
# semantics for binary operations.
state_sum2 = state_sum2 + masked_grad.pow(2).to_tensor(0)
# We can eventually construct a masked std backed by a sparse layout
std2 = masked_tensor(state_sum2, masked_grad.mask()) #, layout=torch.layout.coo)
# Let's print both this version and the regular version for easier comparison
print("state_sum:\n", state_sum)
print("std:\n", std)
print("state_sum2:\n", state_sum2)
print("std2:\n", std2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lQ_ywlvZUbVu
outputId: 9987db25-9776-4360-ee50-5086663109b5
---
# We can add support for in-place operations later. Notice how this doesn't
# need to access any storage internals and is in general a lot shorter
std2 = std2.sqrt().add(eps)

print("std:\n", std)
print("std2:\n", std2)

# to_tensor ideally eventually returns a torch.Tensor with sparse layout
# but would currently return a SparseTensor.
param2 = param2.add((masked_grad / std2).to_tensor(0), alpha=-clr)

# The final results is the same
print("param:\n", param)
print("param2:\n", param2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 0lCv9h3EVj1_
outputId: 26793681-2bac-417f-a6f2-adf492feeeb2
---
# # For reference, this is the regular, dense code path without masked gradients or sparsity
# state_sum.addcmul_(grad, grad, value=1)
# std = state_sum.sqrt().add_(eps)
# param.addcdiv_(grad, std, value=-clr)

# Compare this to the original for sparse
grad = grad.coalesce()  # the update is non-linear so indices must be unique
grad_indices = grad._indices()
grad_values = grad._values()
size = grad.size()

state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
std = state_sum.sparse_mask(grad)
std_values = std._values().sqrt_().add_(eps)
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)

# All in all MaskedTensor minimizes the code to the follwing snippet
state_sum2 = state_sum2 + masked_grad.pow(2).to_tensor(0)
std2 = masked_tensor(state_sum2, masked_grad.mask()) #, layout=torch.layout.coo)
std2 = std2.sqrt().add(eps)
param2 = param2.add((masked_grad / std2).to_tensor(0), alpha=-clr)

# We ran this code again so let's check that the results again match
print("param:\n", param)
print("param2:\n", param2)
```

+++ {"id": "gMqxnPslmTCI"}

### [21987](https://github.com/pytorch/pytorch/issues/21987)
Was closed by inclusion into [Implement missing torch.nan* operators](https://github.com/pytorch/pytorch/issues/61474). This proposes an alternative, which is to use masked tensors instead of introducing additional operators. Since nanmean [has already landed](https://github.com/pytorch/pytorch/issues/21987) we can use it as a comparison point.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: X-b9rVOzmNH0
outputId: 1ab8b7eb-57f7-4a70-bc73-abbbabee7b6a
---
y = torch.arange(32).float()
x = y * y.fmod(4)
x = x.masked_fill(x == 0, float('nan'))
print(x)
print(torch.nanmean(x))
print(torch.mean(masked_tensor(x, ~torch.isnan(x))))
```

+++ {"id": "R8zHciFup_C-"}

MaskedTensor can further support reduction when fully masked out, as would be the case when a given Tensor is completetely nan. nanmean on the other hand returns nan when the input is entirely nan.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Y9_ZT63Mm96y
outputId: dccda564-ff2a-4e70-b58b-62668c2fdde3
---
x = torch.empty(32)
x.fill_(float('nan'))
print(x)
print(torch.nanmean(x))
print(torch.mean(masked_tensor(x, ~torch.isnan(x))))
```

+++ {"id": "jZwH_xn308Rw"}

Further [some users](https://github.com/pytorch/pytorch/issues/63870) already want to use nan reductions to encode masked semantics.
