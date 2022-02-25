.. currentmodule:: torch

Reductions
============

As an intro to masked reductions, please find the document on reduction semantics `here <https://github.com/pytorch/rfcs/pull/27>`_.

In general, an operator is a reduction operator if it reduces one or more dimensions of the input tensor to a single value. 
For masked tensors, reduction operators must implement the following signature: :mod:`(input, mask=None, *args, *, dim=None, keepdim=False, dtype=None)`

Reductions are currently implemented in :mod:`torch._masked`. 

The available operators are:

.. autosummary::
    :toctree: generated
    :nosignatures:

    _masked.sum
    _masked.mean
    _masked.amin
    _masked.amax
    _masked.prod

The next ops to be implemented will be in the ones in the `MaskedTensor Reduction RFC <https://github.com/pytorch/rfcs/blob/8cb9ce7fe84724099138dc281080c74ad1bc2cca/RFC-0016-Masked-reductions-and-normalizations.md>`_. 
If you would like any others implemented, please create a feature request with a proposed input/output semantics.
