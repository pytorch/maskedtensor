.. currentmodule:: torch

Pass through functions
============

Pass through functions are simply functions that should be applied to both the mask and the data. 

By way of example, consider :mod:`select`; this operation can be applied to both the data
and the mask of a :mod:`MaskedTensor`, and the result will then be wrapped into a new :mod:`MaskedTensor`. 

A quick example of this: 

::

    >>> data = torch.arange(12, dtype=torch.float).reshape((3,4))
    >>> mask = torch.tensor([
            [True, False, False, True],
            [False, True, False, False],
            [True, True, True, True]])
    >>> mt = masked_tensor(data, mask)
    >>> data.select(0, 1)
    tensor([4., 5., 6., 7.])
    >>> mask.select(0, 1)
    tensor([False,  True, False, False])
    >>> mt.select(0, 1)
    masked_tensor(
    [      --,   5.0000,       --,       --]
    )

Below is a list of the ops that are currently implemented as pass through functions.

.. autosummary::
    :toctree: generated
    :nosignatures:

    ops.aten.cat
    ops.aten.expand
    ops.aten.index
    ops.aten.slice
    ops.aten.slice_backward
    ops.aten.select
    ops.aten.select_backward
    ops.aten.split
    ops.aten.t
    ops.aten.transpose
    ops.aten.view
    ops.aten._reshape_alias
    ops.aten._unsafe_view

