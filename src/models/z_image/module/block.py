from typing import Optional

import torch
from torch import nn

from .attn import Z_Attention

from managers.mem_manager import use_hbm_allocator

class Z_SingleBlock(nn.Module):
    def __init__(self, pretrained_block, sp_size):
        super().__init__()

        self.attention = Z_Attention(
            pretrained_attn=pretrained_block.attention,
            sp_size=sp_size
        )
        pretrained_block.attention = None

        self.z_block = pretrained_block

    def forward(
        self,
        hidden_states,
        attn_mask: torch.Tensor,
        RoPE_embeds,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        block = self.z_block

        if block.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            # with use_hbm_allocator():
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
    
            attn_out = self.attention(
                block.attention_norm1(hidden_states) * scale_msa,
                attn_mask=attn_mask,
                RoPE_embeds=RoPE_embeds,
            )
            # with use_hbm_allocator():
            hidden_states = hidden_states + gate_msa * block.attention_norm2(attn_out)
            hidden_states = block.ffn_norm1(hidden_states) * scale_mlp
    
            hidden_states = block.feed_forward(hidden_states)
        
            # with use_hbm_allocator():
            hidden_states = block.ffn_norm2(hidden_states) * gate_mlp + hidden_states
            # hidden_states = hidden_states + gate_mlp * block.ffn_norm2(block.feed_forward(block.ffn_norm1(hidden_states) * scale_mlp))
        else:
            attn_out = self.attention(
                block.attention_norm1(hidden_states),
                attn_mask=attn_mask,
                RoPE_embeds=RoPE_embeds,)
            hidden_states = hidden_states + block.attention_norm2(attn_out)

            hidden_states = hidden_states + block.ffn_norm2(block.feed_forward(block.ffn_norm1(hidden_states)))

        return hidden_states

class Z_JointBlock(nn.Module):
    def __init__(self, pretrained_block,sp_size):
        super().__init__()

        self.attention = Z_Attention(
            pretrained_attn=pretrained_block.attention,
            sp_size=sp_size
        )
        pretrained_block.attention = None

        self.z_block = pretrained_block

    def forward(
        self,
        hidden_states,
        attn_mask: torch.Tensor,
        RoPE_embeds,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        block = self.z_block

        if block.modulation:

            scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            # with use_hbm_allocator():
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
    
            attn_out = self.attention(
                block.attention_norm1(hidden_states) * scale_msa,
                attn_mask=attn_mask,
                RoPE_embeds=RoPE_embeds,
            )
        
            # with use_hbm_allocator():
            hidden_states = hidden_states + gate_msa * block.attention_norm2(attn_out)
            hidden_states = block.ffn_norm1(hidden_states) * scale_mlp

            hidden_states = block.feed_forward(hidden_states)

            # with use_hbm_allocator():
            hidden_states = block.ffn_norm2(hidden_states) * gate_mlp + hidden_states

            # hidden_states = hidden_states + gate_mlp * block.ffn_norm2(block.feed_forward(block.ffn_norm1(hidden_states) * scale_mlp))
        else:
            attn_out = self.attention(
                block.attention_norm1(hidden_states),
                attn_mask=attn_mask,
                RoPE_embeds=RoPE_embeds,)
            # with use_hbm_allocator():
            hidden_states = hidden_states + block.attention_norm2(attn_out)
            hidden_states = block.ffn_norm1(hidden_states)

            hidden_states = block.feed_forward(hidden_states)
            
            # with use_hbm_allocator():
            hidden_states = hidden_states + block.ffn_norm2(hidden_states)
            # hidden_states = hidden_states + block.ffn_norm2(block.feed_forward(block.ffn_norm1(hidden_states)))

        return hidden_states