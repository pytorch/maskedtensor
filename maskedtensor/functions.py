import torch
import math
from .creation import masked_tensor
from torch.nn.functional import linear

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


def _in_projection_packed(q, k, v, w, b):
    # if is_masked_tensor(k):
    #     assert not is_masked_tensor(q)
    #     assert not is_masked_tensor(v)
    E = q.size(-1)
    w_q, w_k, w_v = w.chunk(3)
    assert b is None
    b_q = b_k = b_v = None
    # print("k: ", k)
    # print("w_k: ", w_k)
    # print("k.size(): ", k.size())
    # print("w_k.size(): ", w_k.size())
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0):
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = masked_bmm(q, k.transpose(-2, -1), attn_mask)
    attn = torch.nn.functional.softmax(attn, dim=-1)
    print("attn: ", attn)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training,
    key_padding_mask,
    need_weights,
    attn_mask,
    use_separate_proj_weight,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    static_k,
    static_v,
):
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert not use_separate_proj_weight
    assert q_proj_weight is None
    assert k_proj_weight is None
    assert v_proj_weight is None
    assert in_proj_bias is None
    assert bias_k is None
    assert bias_v is None
    assert static_k is None
    assert static_v is None
    assert not add_zero_attn
    assert key_padding_mask is None

    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # prep attention mask
    if attn_mask is not None:
        assert (
            attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
        ), f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        assert attn_mask.dim() == 2
        correct_2d_size = (tgt_len, src_len)
        if attn_mask.shape != correct_2d_size:
            raise RuntimeError(
                f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
            )
        attn_mask = attn_mask.unsqueeze(0)

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    print("attn_output0: ", attn_output)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    print("attn_output1: ", attn_output)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        print("attn_output_weights: ", attn_output_weights)
        print("attn_output_weights.sum(dim=1): ", attn_output_weights.sum(dim=1))
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
