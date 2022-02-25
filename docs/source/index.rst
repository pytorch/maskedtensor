.. MaskedTensor documentation master file, created by
   sphinx-quickstart on Wed Feb 16 11:48:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MaskedTensor
========================================

This library is a part of the `PyTorch <http://pytorch.org/>`_ project. The purpose of :mod:`maskedtensor` is to serve as an extension to `torch.Tensor`, especially in cases of:

* Using any masked semantics
* Differentiation between 0 and NaN gradients
* Various sparse applications

More details can be found in the Overview tutorial.

Please note that this library is very much in its early development stages. The Github can be found `here <https://github.com/pytorch/maskedtensor>`_, where we welcome any feature requests, issues, etc.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   install

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/overview
   notebooks/nan_grad

.. toctree::
   :maxdepth: 1
   :caption: Python API

   passthrough
   reductions
   binary
