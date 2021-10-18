from .maskedtensor import MaskedTensor

# Basic factory function
def masked_tensor(data, mask, requires_grad=False):
    data = data.clone().detach()
    mask = mask.clone().detach()
    return MaskedTensor(data, mask, requires_grad)
