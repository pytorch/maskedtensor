---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3.9.7 ('pytorch_env')
  language: python
  name: python3
---

# Sparse semantics

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/sparse.ipynb)

+++

## Introduction

+++

[Sparsity in PyTorch](https://pytorch.org/docs/stable/sparse.html) is a quickly growing area that has found a lot of support and demand due to its efficiency in both memory and compute. This tutorial is meant to be used in conjunction with the the PyTorch link above, as the sparse tensors are ultimately the building blocks for MaskedTensors (just as regular `torch.Tensor`s are as well).

Sparse storage formats are particularly powerful in scenarios where the majority of elements are equal to zero. There are a number of different [sparse storage formats](https://en.wikipedia.org/wiki/Sparse_matrix) that can be leveraged with various tradeoffs and degrees of adoption.

Noting PyTorch's terminology of "specified" and "unspecified" elements (e.g. the elements that are actually stored vs. not), the parallel to MaskedTensor's usage is clear. However, by allowing a mask as well, MaskedTensors are even more generalizable, as we'l show through the tutorial - e.g. when the mask all `True`, most operations will result in the same result, but when the mask indicates unspecified values, then values in the sparse tensor will be masked out.

<div class="alert alert-info">

**Note:** Currently, only the COO sparse storage format is supported in MaskedTensor ([CSR is being developed](https://github.com/pytorch/maskedtensor/pull/65)). If you have another format that you would like supported, please file an issue!

</div>

+++

## Principles

+++

1. `input` and `mask` must have the same storage format, whether that's `torch.strided`, `torch.sparse_coo`, or `torch.sparse_csr`.

2. `input` and `mask` must have the same size, indicated by `t.size()`

3. `input` and `mask` - only for sparse formats - can have a different number of elements (`tensor.nnz()`) **at creation**, the indices of `mask` must then be a subset of the indices from `input`. In this case, `input` will assume the shape of mask using the function `input.sparse_mask(mask)`; in other words, any of the elements in `input` that are not `True` in `mask` will be thrown away

+++

## Sparse COO Tensors

```{code-cell} ipython3
import torch
from maskedtensor import masked_tensor
```

In according with Principle #1, a MaskedTensor is created by passing in two sparse tensors, which can be initialized with any of the constructors, e.g. `torch.sparse_coo_tensor`.

```{code-cell} ipython3
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
m = torch.tensor([True, False, True])

values = torch.sparse_coo_tensor(i, v, (2, 3))
mask = torch.sparse_coo_tensor(i, m, (2, 3))

mt = masked_tensor(values, mask)  

print("values:\n", values.to_dense())
print("mask:\n", mask.to_dense())
print("mt:\n", mt)
```

A word of warning: when using a function like `.to_sparse_coo()`, if the user does not specify the indices like in the above example, then 0 values will be default "unspecified"

```{code-cell} ipython3
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
m = torch.tensor(
     [[False, False, True],
      [False, False, True]]
)

values = torch.sparse_coo_tensor(i, v, (2, 3))
mask = m.to_sparse_coo()
mt2 = masked_tensor(values, mask)

print("values:\n", values)
print("mask:\n", mask)
print("mt2:\n", mt2)
```

#### Principle 3: 

+++

Note that `mt` and `mt2` will have the same value in the vast majority of operations, but it's worth noting that in line with Principle #3, under the hood, the data looks slightly different; `mt` has the 4 value masked out and `mt2` is completely without it. In other words, their underlying data still has different shapes, so `mt + mt2` is invalid.

```{code-cell} ipython3
print("mt.masked_data:\n", mt.masked_data)
print("mt2.masked_data:\n", mt2.masked_data)
```

## Supported Operations

+++

### Unary

+++

All unary operations are supported; for a list of operations, please refer to [here](https://pytorch.org/maskedtensor/main/unary.html).

```{code-cell} ipython3
mt.sin()
```

### Binary

+++

As in the usual case of masked binary operations, the input masks from the two masked tensors must match. For a list of operations, please refer [here](https://pytorch.org/maskedtensor/main/binary.html).

```{code-cell} ipython3
i = [[0, 1, 1],
     [2, 0, 2]]
v1 = [3, 4, 5]
v2 = [20, 30, 40]
m = torch.tensor([True, False, True])

s1 = torch.sparse_coo_tensor(i, v1, (2, 3))
s2 = torch.sparse_coo_tensor(i, v2, (2, 3))
mask = torch.sparse_coo_tensor(i, m, (2, 3))

mt1 = masked_tensor(s1, mask)
mt2 = masked_tensor(s2, mask)
```

```{code-cell} ipython3
print("mt1:\n", mt1)
print("mt2:\n", mt2)
print("torch.div(mt2, mt1):\n", torch.div(mt2, mt1))
print("torch.mul(mt1, mt2):\n", torch.mul(mt1, mt2))
```

### Reductions

+++

Unfortunately, only reductions across all dimensions are supported and not a particular dimension (e.g. `mt.sum()` is supported but not `mt.sum(dim=1)`). For a list of supported reductions, please refer [here](https://pytorch.org/maskedtensor/main/reductions.html).

```{code-cell} ipython3
print("mt:\n", mt)
print("mt.sum():\n", mt.sum())
print("mt.amin():\n", mt.amin())
```

## MaskedTensor methods and sparse

+++

`to_dense()`

```{code-cell} ipython3
mt.to_dense()
```

`to_sparse_coo()`

```{code-cell} ipython3
v = [[3, 0, 0],
     [0, 4, 5]]
m = [[True, False, False],
     [False, True, True]]
mt = masked_tensor(torch.tensor(v), torch.tensor(m))

mt_sparse = mt.to_sparse_coo()
```

`is_sparse` / `is_sparse_coo`

```{code-cell} ipython3
print("mt.is_sparse: ", mt.is_sparse())
print("mt_sparse.is_sparse: ", mt_sparse.is_sparse())

print("mt.is_sparse_coo: ", mt.is_sparse_coo())
print("mt_sparse.is_sparse_coo: ", mt_sparse.is_sparse_coo())
```
