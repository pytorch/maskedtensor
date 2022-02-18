#!/usr/bin/env python
# coding: utf-8

# # Resolving NaN Grad

# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/pytorch/maskedtensor/tree/main/docs/notebooks/nan_grad.ipynb)

# In[1]:


import torch
import numpy as np
if "1.11.0" not in torch.__version__:
    get_ipython().system('pip uninstall --y torch')
    get_ipython().system('pip install torch -f https://download.pytorch.org/whl/test/cu102/torch_test.html --pre')


# In[2]:


get_ipython().system('pip install -i https://test.pypi.org/simple/ maskedtensor')


# In[3]:


# Import factory function
from maskedtensor import masked_tensor
from maskedtensor import as_masked_tensor


# ## Resolving Issues

# One issue that vanilla tensors run into is the inability to differentiate between gradients that are not defined (nan) vs. gradients that are actually 0.
# 
# Below, we show by example several different issues where MaskedTensor can resolve and/or work around these issues.

# ### [10729 - torch.where](https://github.com/pytorch/pytorch/issues/10729)

# **PyTorch result**:

# In[4]:


# This behavior underlies the fix to clamp, which uses where in its derivative
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
print("y:", y)
y.sum().backward()
print("x.grad:", x.grad)
print("y.grad:", y.grad)


# **MaskedTensor result**:

# In[5]:


x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
mask = x < 0
mx = masked_tensor(x, mask, requires_grad=True)
my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
y = torch.where(mask, torch.exp(mx), my)
s = y.sum()
s.backward()
# Gradient is only provided to selected subset.
# Effectively this changes the gradient of where to mask out elements instead
# of setting them to zero.
print("mx.grad: ", mx.grad)


# The gradient here is only provided to the selected subset. Effectively, this changes the gradient of where to mask out elements instead of setting them to zero.

# ### [52248 - another torch.where](https://github.com/pytorch/pytorch/issues/52248)

# **PyTorch result**:

# In[6]:


# A more recent incarnation specific to where of this
# https://github.com/pytorch/pytorch/issues/52248

a = torch.randn((), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())

print(torch.where(b, a/0, c))
print(torch.autograd.grad(torch.where(b, a/0, c), a))


# **MaskedTensor result**:

# In[7]:


a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())

print(torch.where(b, a/0, c))
print(torch.autograd.grad(torch.where(b, a/0, c), a))


# ### [67180 - torch.nansum and torch.nanmean](https://github.com/pytorch/pytorch/issues/67180)

# **PyTorch result**:

# In[8]:


a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
c = a * b
c1 = torch.nansum(c)  # or torch.nanmean

bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1


# **MaskedTensor result**:

# In[9]:


a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
ma = masked_tensor(a, ~torch.isnan(a))
c = ma * b
c1 = torch.sum(c)  # or torch.nanmean

bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1


# ### [4132 - when using mask, x/0 yields NaN grad](https://github.com/pytorch/pytorch/issues/4132)

# PyTorch result:

# In[10]:


x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]

mask = (div != 0) # => mask is [0, 1]
loss = y[mask]
loss.backward()

x.grad # grad is [nan, 1], but expected [0, 1]


# MaskedTensor result:

# In[11]:


x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]

mask = (div != 0) # => mask is [0, 1]
loss = as_masked_tensor(y, mask)
# We could add autograd support for indexing here instead of using sum
loss = loss.sum()
loss.backward()

x.grad # grad is [nan, 1], but expected [0, 1]


# In[ ]:




