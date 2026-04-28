import copy
from typing import List, Optional

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers import ZImageTransformer2DModel
from torch.nn.utils.rnn import pad_sequence
from torch import nn

from .module.block import Z_JointBlock, Z_SingleBlock
from ..embedding import TimestepEmbedder
from .base import Z_Mixin

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

class Z_Transformer(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    Z_Mixin,
):
    def __init__(
        self,
        z_transformer: ZImageTransformer2DModel,
        num_metadata: int = 0,
        in_channels: Optional[int] = None,
        use_cond: bool = False,
    ):
        super().__init__()
        if in_channels is None:
            self.out_channels = z_transformer.out_channels
            self.in_channels = z_transformer.in_channels
        else:
            self.out_channels = in_channels
            self.in_channels = in_channels

        if use_cond:
            cond_refiner = copy.deepcopy(z_transformer.noise_refiner)
            self.cond_refiner = nn.ModuleList(
                [
                    Z_SingleBlock(layer)
                    for layer in cond_refiner
                ]
            )
            self.cond_embedder = nn.Linear(2 * 2 * 192, z_transformer.dim)
            self.cond_pad_token = copy.deepcopy(z_transformer.x_pad_token)
        
        self.noise_refiner = nn.ModuleList(
            [
                Z_SingleBlock(layer)
                for layer in z_transformer.noise_refiner
            ]
        )
        z_transformer.noise_refiner = None
        
        self.context_refiner = nn.ModuleList(
            [
                Z_SingleBlock(layer)
                for layer in z_transformer.context_refiner
            ]
        )
        z_transformer.context_refiner = None

        self.layers = nn.ModuleList(
            [
                Z_JointBlock(layer)
                for layer in z_transformer.layers
            ]
        )
        z_transformer.layers = None

        self.z_transformer = z_transformer
        
        self.n_metadata = num_metadata
        if num_metadata > 0:
            dim = z_transformer.dim
            self.meta_embeddings = nn.ModuleList(
                TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
                for _ in range(num_metadata)
            )
        else:
            self.meta_embeddings = None
    
    @property
    def config(self):
        return self.z_transformer.config

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        text_embedding: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
        metadata: List[torch.Tensor] = None,
        cond_list: List[torch.Tensor] = None,
    ):

        print(f"len(cond_list) {len(cond_list)}")
        transformer = self.z_transformer

        bsz = len(x)
        device = x[0].device
        t = t * transformer.t_scale
        t = transformer.t_embedder(t)

        if metadata is not None:
            # metadata shape: [B, num_metadata]
            for i, md_embedder in enumerate(self.meta_embeddings):
                cond_emb = md_embedder(metadata[:, i])
                t = t + cond_emb

        (
            x,
            text_embedding,
            x_size,
            x_pos_ids,
            text_pos_ids,
            x_inner_pad_mask,
            text_inner_pad_mask,
            cond_out_list,
        ) = self.patchify_and_embed(x, text_embedding, patch_size, f_patch_size, cond_list)
        #print(conda_out_list[0].shape)
        # x embed
        x_item_seqlens = [len(_) for _ in x]
        x_max_item_seqlen = max(x_item_seqlens)
        x = torch.cat(x, dim=0)
        x = transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = transformer.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_RoPE_embeds = list(transformer.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        # x_padded & x_RoPE_padded
        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_RoPE_embeds = pad_sequence(x_RoPE_embeds, batch_first=True, padding_value=0.0)
        x_RoPE_embeds = x_RoPE_embeds[:, : x.shape[1]]

        # x_attn_mask
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # x refiner
        for layer in self.noise_refiner:
            x = layer(
                hidden_states=x,
                attn_mask=x_attn_mask,
                RoPE_embeds=x_RoPE_embeds,
                adaln_input=adaln_input,
            )
        
        # text embed
        text_item_seqlens = [len(_) for _ in text_embedding]
        text_max_item_seqlen = max(text_item_seqlens)

        text_embedding = torch.cat(text_embedding, dim=0)
        text_embedding = transformer.cap_embedder(text_embedding)
        text_embedding[torch.cat(text_inner_pad_mask)] = transformer.cap_pad_token
        text_embedding = list(text_embedding.split(text_item_seqlens, dim=0))
        text_RoPE_embeds = list(
            transformer.rope_embedder(
                torch.cat(text_pos_ids, dim=0)
            ).split([len(_) for _ in text_pos_ids], dim=0)
        )

        # text_padded & text_RoPE_padded
        text_embedding = pad_sequence(text_embedding, batch_first=True, padding_value=0.0)
        text_RoPE_embeds = pad_sequence(text_RoPE_embeds, batch_first=True, padding_value=0.0)
        text_RoPE_embeds = text_RoPE_embeds[:, : text_embedding.shape[1]]

        # text attn mask
        text_attn_mask = torch.zeros((bsz, text_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(text_item_seqlens):
            text_attn_mask[i, :seq_len] = 1

        # text refiner
        for layer in self.context_refiner:
            text_embedding = layer(
                hidden_states=text_embedding,
                attn_mask=text_attn_mask,
                RoPE_embeds=text_RoPE_embeds,
            )
        # print(text_embedding.shape)
        # cond process
        if cond_list is not None:
            # cond_padded & cond_RoPE_padded
            cond_padded, cond_pad_mask, cond_padded_pos_ids = cond_out_list
            cond_item_seqlens = [len(_) for _ in cond_padded]
            cond_max_item_seqlen = max(cond_item_seqlens)
            
            cond_padded = torch.cat(cond_padded, dim=0)
            cond_padded = self.cond_embedder(cond_padded)
            cond_padded[torch.cat(cond_pad_mask)] = self.cond_pad_token
            cond_padded = list(cond_padded.split(cond_item_seqlens, dim=0))
            print(cond_padded[0].shape)
            cond_RoPE_embeds = list(
                transformer.rope_embedder(
                    torch.cat(cond_padded_pos_ids, dim=0)
                ).split([len(_) for _ in cond_padded_pos_ids], dim=0)
            )
            
            # cond_padded & cond_RoPE_padded
            cond_padded = pad_sequence(cond_padded, batch_first=True, padding_value=0.0)
            cond_RoPE_embeds = pad_sequence(cond_RoPE_embeds, batch_first=True, padding_value=0.0)
            cond_RoPE_embeds = cond_RoPE_embeds[:, : cond_padded.shape[1]]

            # cond attn mask
            cond_attn_mask = torch.zeros((bsz, cond_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(cond_item_seqlens):
                cond_attn_mask[i, :seq_len] = 1

            # cond refiner
            for layer in self.cond_refiner:
                cond_padded = layer(
                    hidden_states=cond_padded,
                    attn_mask=cond_attn_mask,
                    RoPE_embeds=cond_RoPE_embeds,
                    adaln_input=adaln_input,
                )

            # unified sequence prepare
            unified = []
            unified_RoPE_embeds = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                text_len = text_item_seqlens[i]
                cond_len = cond_item_seqlens[i]
                unified.append(torch.cat([x[i][:x_len], text_embedding[i][:text_len], cond_padded[i][:cond_len]]))
                unified_RoPE_embeds.append(torch.cat([x_RoPE_embeds[i][:x_len], text_RoPE_embeds[i][:text_len], cond_RoPE_embeds[i][:cond_len]]))
            
            unified_item_seqlens = [a + b + c for a, b, c in zip(text_item_seqlens, x_item_seqlens, cond_item_seqlens)]
            unified_max_item_seqlen = max(unified_item_seqlens)

            # padding and attn mask
            unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
            unified_RoPE_embeds = pad_sequence(unified_RoPE_embeds, batch_first=True, padding_value=0.0)
            unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_item_seqlens):
                unified_attn_mask[i, :seq_len] = 1

        else:
            # unified sequence prepare
            unified = []
            unified_RoPE_embeds = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                text_len = text_item_seqlens[i]
                unified.append(torch.cat([x[i][:x_len], text_embedding[i][:text_len]]))
                unified_RoPE_embeds.append(torch.cat([x_RoPE_embeds[i][:x_len], text_RoPE_embeds[i][:text_len]]))
            
            unified_item_seqlens = [a + b for a, b in zip(text_item_seqlens, x_item_seqlens)]
            unified_max_item_seqlen = max(unified_item_seqlens)

            # padding and attn mask
            unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
            unified_RoPE_embeds = pad_sequence(unified_RoPE_embeds, batch_first=True, padding_value=0.0)
            unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(unified_item_seqlens):
                unified_attn_mask[i, :seq_len] = 1
        # print(unified.shape)
        # backbone
        print(f"unified shape {unified.shape}")
        
        for layer in self.layers:
            unified = layer(
                hidden_states=unified,
                attn_mask=unified_attn_mask,
                RoPE_embeds=unified_RoPE_embeds,
                adaln_input=adaln_input,
            )
        
        unified = transformer.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size, self.out_channels)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)
    
    def switch_meta_embedding_grad(self, grad_stat: bool = True):
        total_switched_params = 0
        if self.meta_embeddings is not None:
            for param in self.meta_embeddings.parameters():
                param.requires_grad_(grad_stat)
                total_switched_params += param.numel()
        return total_switched_params
    
    def get_branch_lora_modules(self, branch_name: str):
        blocks_lora_modules = []
        
        blocks = getattr(self, branch_name)
        for i, block in enumerate(blocks):
            blocks_lora_modules.extend([
                f'{branch_name}.{i}.attention.to_q',
                f'{branch_name}.{i}.attention.to_k',
                f'{branch_name}.{i}.attention.to_v',
                f'{branch_name}.{i}.attention.to_out.0',
                f'{branch_name}.{i}.z_block.feed_forward.w1',
                f'{branch_name}.{i}.z_block.feed_forward.w2',
                f'{branch_name}.{i}.z_block.feed_forward.w3',
            ])
            if block.z_block.modulation:
                blocks_lora_modules.append(
                    f'{branch_name}.{i}.z_block.adaLN_modulation.0',
                )
            return blocks_lora_modules
        
    def get_noise_lora_branch(self, patch_size = 2, f_patch_size = 1):
        lora_branch = self.get_branch_lora_modules("noise_refiner")
        lora_branch.append(f"all_x_embedder.{patch_size}-{f_patch_size}")
        lora_branch.extend([
            f"all_final_layer.{patch_size}-{f_patch_size}.linear",
            f"all_final_layer.{patch_size}-{f_patch_size}.adaLN_modulation.1",
        ])
        return lora_branch
    
    def get_context_lora_branch(self):
        lora_branch = ['cap_embedder.1']
        lora_branch.extend(self.get_branch_lora_modules("context_refiner"))
        return lora_branch

    def get_main_lora_branch(self):
        lora_branch = self.get_branch_lora_modules("layers")
        lora_branch.extend(["t_embedder.mlp.0", "t_embedder.mlp.2"])
        return lora_branch

    def switch_cond_branch_grad(self, grad_stat: bool = True):
        total_switched_params = 0
        cond_modules = ['cond_embedder']
        if self.cond_embedder is not None:
            for param in self.cond_embedder.parameters():
                param.requires_grad_(grad_stat)
                total_switched_params += param.numel()

            for param in self.cond_refiner.parameters():
                param.requires_grad_(grad_stat)
                total_switched_params += param.numel()
        
        return total_switched_params
