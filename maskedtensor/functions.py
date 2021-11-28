import torch
from .creation import masked_tensor

Tensor = torch.Tensor


class MaskedBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, attn_mask):
        from maskedtensor import is_masked_tensor

        assert not is_masked_tensor(q)
        assert is_masked_tensor(k)
        k_mask = k.mask()
        ctx.mark_non_differentiable(attn_mask, k_mask)
        ctx.save_for_backward(attn_mask, k_mask, q, k)
        attn = torch.bmm(q, k)
        return_mask = attn_mask.expand_as(attn.masked_data)
        return masked_tensor(attn.masked_data + return_mask, return_mask == 0)

    @staticmethod
    def backward(ctx, grad):
        attn_mask, k_mask, q, k = ctx.saved_tensors
        grad_data = grad.masked_data

        k_trans = k.transpose(1, 2)
        q_grad = torch.bmm(grad_data, k_trans)

        q_trans = q.transpose(1, 2)
        k_grad = torch.bmm(q_trans, grad)
        k_grad = masked_tensor(k_grad.masked_data, k_mask)

        return q_grad, k_grad, None


def masked_bmm(q, k, attn_mask):
    return MaskedBmm.apply(q, k, attn_mask)
