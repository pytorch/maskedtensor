import torch
from torch.nn.functional import linear
from maskedtensor import masked_tensor

class ExampleTensor(torch.Tensor):
  @staticmethod
  def __new__(cls, data):
    # Use a Tensor that of the give size for the wrapper.
    kwargs = {}
    kwargs['requires_grad'] = True
    kwargs['dispatch_strides'] = True
    return torch.Tensor._make_subclass(cls, torch.empty(data.size(), device='meta'), **kwargs)

  @classmethod
  def __torch_dispatch__(cls, func, types, args, kwargs):
    print("Calling func: ", func)
    return NotImplemented

torch.manual_seed(0)
attn_nn = torch.nn.MultiheadAttention(1, 1, bias=False)
attn_mt = torch.nn.MultiheadAttention(1, 1, bias=False)
for (na, a), (nb, b) in zip(
    attn_nn.named_parameters(), attn_mt.named_parameters()
):
    a.data.copy_(b.data)

x = torch.rand(3, 2, 1)
key_padding_mask = torch.as_tensor(
    [
        [False, False, False],
        [False, True, True],
    ]
)
attn_mask = torch.as_tensor(
    [
        [False, True, True],
        [False, False, True],
        [True, False, False],
    ]
)
output, scores = attn_nn(
    x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
)
loss0 = output[0, :].sum()

x_mt = masked_tensor(
    x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
)

output, scores = attn_mt(x, x_mt, x, attn_mask=attn_mask)
