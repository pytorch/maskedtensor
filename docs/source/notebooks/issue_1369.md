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

# Efficiency of writing "sparse" semantics for Adagrad

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/issue_1369.ipynb)

+++

## Motivation

+++

[Issue 1369](https://github.com/pytorch/pytorch/issues/1369) discussed the additional lines of code that were introduce while writing "sparse" semantics for Adagrad. But really the code doesn't use sparsity as a compression and optimization technique, it wants to use masked semantics. We worked around this by introducing one-off semantics and operators that encode this behavior while forcing users to be aware of storage details such as indices and values. Let's look at the current implementation of [Adagrad](https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py) [(functional)](https://github.com/pytorch/pytorch/blob/6c2f235d368b697072699e5ca9485fd97d0b9bcc/torch/optim/_functional.py#L16-L51) to illustrate that.

In particular we'll point out when sparsity is used as a semantic extension, i.e. unspecified values are not zero and when it is just used to compress zeros. We'll also compare and contrast this with equivalent code written using MaskedTensor. In the end the code snippets are repeat without additional comments to show the difference in brevity.

+++

## Code

```{code-cell} ipython3
import torch
if "1.11.0" not in torch.__version__:
    !pip uninstall --y torch
    !pip install torch -f https://download.pytorch.org/whl/test/cu102/torch_test.html --pre
```

```{code-cell} ipython3
# Import factory function
from maskedtensor import masked_tensor
from maskedtensor import as_masked_tensor
```

```{code-cell} ipython3
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

This is where we have a very important divergence. The addition of eps
should technically be applied to all values, but instead is only applied to
specified values. Here we're using sparsity as a semantic extension and
to enforce a certain pattern of defined and undefined values. If parts
of the values of the gradient are zero they are still included if materialized.
Even though they could be compressed by other sparse storage layouts.
This is technically quite brittle even though someone could argue that eps is
always very small.

More so an implementation add\_ for sparsity as a storage layout and compression
scheme should cause densification, but we force it not to. For this one-off
case it is fine until we want to introduce new compression schemes such as
CSR, BSR or 2:4 block sparsity. We'll then need to introduce separate Tensor
types for each and write variations for gradients compressed using different
storage formats.

```{code-cell} ipython3
# We currently dodge all these concerns using the private method values.
std_values = std._values().sqrt_().add_(eps)

# We currently don't support div for sparse Tensors because zero / zero is
# not well defined. For a MaskedTensor undefined / undefined is undefined.
param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
print("param:\n", param)
```

We've been conflating sparsity as an optimization with sparsity as a semantic extension to PyTorch. MaskedTensor proposes to call the semantic extension through sparsity masked. Currently we can't have dense semantics with sparse storage or masked semantics with dense storage. MaskedTensor fixes that because it separates the storage from the semantics. Let's look at above example using a masked gradient.

+++

Of course we can add sparsity as a storage layout for MaskedTensor which
compresses repeated undefined values. We can recycle SparseTensor and SparseCSR
by setting data and mask to an instance of each that share indices.
However, ideally we'd just have regular torch.Tensors with a sparse layout
and use those to back MaskedTensor.

```{code-cell} ipython3
masked_grad = masked_tensor(grad.to_dense(), grad.to_dense() != 0)
print("masked_grad:\n", masked_grad)
```

```{code-cell} ipython3
# Create an entirely new set of parameters to avoid errors
param2 = torch.arange(8).reshape(2, 4).float()
state_sum2 = torch.full_like(param, 0.5) # initial value for state sum
```

```{code-cell} ipython3
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

```{code-cell} ipython3

```
