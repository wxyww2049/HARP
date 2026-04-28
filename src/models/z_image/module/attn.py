import math

import torch
from torch import nn
import torch.distributed as dist

# from src.utils.rope import apply_rotary_emb
# from kp_rope import kp_apply_rotary_emb
from async_rt.async_ops import ApplyRotaryComplexFn

from managers.ring_attn_manager import CPUWaitEngine
from managers.mem_manager import use_hbm_allocator


class CPURingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_q, local_k, local_v, sp_size):
        ctx.sp_size = sp_size
        # print(sp_size)
        scale = 1.0 / math.sqrt(local_q.size(-1))
        ctx.scale = scale
        mmType = local_q.dtype

        with use_hbm_allocator():
            out = torch.zeros_like(local_q, dtype=torch.float32)
        m_i = torch.full((local_q.shape[0], local_q.shape[1], local_q.shape[2], 1), float('-inf'), device=local_q.device, dtype=torch.bfloat16)
        l_i = torch.empty((local_q.shape[0], local_q.shape[1], local_q.shape[2], 1), device=local_q.device, dtype=torch.float32)

        torch.ops.ring_attention_ops.ring_attention_forward(local_q, local_k, local_v, out, m_i, l_i, sp_size, scale)
        
        ctx.save_for_backward(local_q, local_k, local_v, out, m_i, l_i)
        out = out.to(mmType)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        local_q, local_k, local_v, out, m_i, l_i = ctx.saved_tensors
        scale = ctx.scale
        sp_size = ctx.sp_size

        D = torch.sum(grad_out.to(torch.float32) * out, dim=-1, keepdim=True)
        grad_q = torch.empty_like(local_q, dtype=torch.bfloat16)
        grad_k = torch.empty_like(local_k, dtype=torch.bfloat16)
        grad_v = torch.empty_like(local_v, dtype=torch.bfloat16)

        torch.ops.ring_attention_ops.ring_attention_backward(local_q, grad_q,
            local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale)

        return grad_q, grad_k, grad_v, None

class Z_Attention(nn.Module):
    def __init__(
        self,
        pretrained_attn,
        sp_size
    ):
        super().__init__()
        self.attn = pretrained_attn
        self.sp_size = sp_size
    def forward(
        self,
        hidden_states,
        RoPE_embeds,
        attn_mask,
    ):
        attn = self.attn
        # hidden_states: (bs, seqlen, 3840)
        # q,k,v: (bs, seqlen, 3840)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # q,k,v: (bs, 30, seqlen, 128)
        query = query.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(-1, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if RoPE_embeds is not None:
            # query = kp_apply_rotary_emb(query, RoPE_embeds, use_real=False)
            # key = kp_apply_rotary_emb(key, RoPE_embeds, use_real=False)
            # query = apply_rotary_emb(query, RoPE_embeds, use_real=False)
            # key = apply_rotary_emb(key, RoPE_embeds, use_real=False)
            query = ApplyRotaryComplexFn.apply(query, RoPE_embeds)
            key = ApplyRotaryComplexFn.apply(key, RoPE_embeds)

        # dtype = query.dtype
        # query, key = query.to(dtype), key.to(dtype)
        # if attn_mask is not None and attn_mask.ndim < 4:
        #     if attn_mask.ndim == 2:
        #         attn_mask = attn_mask[:, None, None, :]
        #     elif attn_mask.ndim == 3:
        #         attn_mask = attn_mask[:, None, :, :]

        hidden_states = CPURingAttention.apply(query,key,value,self.sp_size).transpose(1,2).flatten(2,3)

        hidden_states = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
