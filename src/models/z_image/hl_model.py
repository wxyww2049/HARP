import copy
from typing import List, Optional

import torch

from .standalone_transformer import (
    ConfigMixin,
    FromOriginalModelMixin,
    ModelMixin,
    PeftAdapterMixin,
    Transformer2DModelOutput,
    ZImageTransformer2DModel,
)

from torch.nn.utils.rnn import pad_sequence
from torch import nn

from .module.block import Z_JointBlock, Z_SingleBlock
from ..embedding import TimestepEmbedder
from .hl_base import Z_Mixin
from managers.shard_manager import SequenceScatter, SequenceAllGather, SPBroadcast
import torch.distributed as dist

from async_rt.runtime import AsyncSync, AsyncModeStart, AsyncModeEnd


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
        sp_group_model = None,
        sp_size = None,
    ):
        super().__init__()
        self.sp_group_model = sp_group_model
        self.sp_rank = dist.get_rank(self.sp_group_model)
        self.sp_world_size = dist.get_world_size(self.sp_group_model)
        # print(f"sp rank: {self.sp_rank}, sp size: {self.sp_world_size}")
        if in_channels is None:
            self.out_channels = z_transformer.out_channels
            self.in_channels = z_transformer.in_channels
        else:
            self.out_channels = in_channels
            self.in_channels = in_channels

        if use_cond:
            self.cond_refiner_1 = nn.ModuleList(
                [
                    Z_SingleBlock(layer,sp_size)
                    for layer in copy.deepcopy(z_transformer.noise_refiner)
                ]
            )
            self.cond_refiner_2 = nn.ModuleList(
                [
                    Z_SingleBlock(layer,sp_size)
                    for layer in copy.deepcopy(z_transformer.noise_refiner)
                ]
            )
            self.cond_embedder_1 = nn.Linear(2 * 2 * 192, z_transformer.dim)
            self.cond_embedder_2 = nn.Linear(2 * 2 * 192, z_transformer.dim)
            self.cond_pad_token = copy.deepcopy(z_transformer.x_pad_token)
        
        self.noise_refiner = nn.ModuleList(
            [
                Z_SingleBlock(layer,sp_size)
                for layer in z_transformer.noise_refiner
            ]
        )
        z_transformer.noise_refiner = None
        
        self.context_refiner = nn.ModuleList(
            [
                Z_SingleBlock(layer,sp_size)
                for layer in z_transformer.context_refiner
            ]
        )
        z_transformer.context_refiner = None

        self.layers = nn.ModuleList(
            [
                Z_JointBlock(layer,sp_size)
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
        cond_list: List[List[torch.Tensor]] = None,
    ):
    
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
        # print(f"len(cond_list) in model.py {len(cond_list[0])}")

        (
            x,
            text_embedding,
            x_size,
            x_pos_ids,
            text_pos_ids,
            x_inner_pad_mask,
            text_inner_pad_mask,
            cond_out_dict,
        ) = self.patchify_and_embed(x, text_embedding, patch_size, f_patch_size, cond_list)

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

        x = SequenceScatter.apply(x, self.sp_group_model)
        adaln_input = SPBroadcast.apply(adaln_input, self.sp_group_model)
        tmp_x_attn_mask = torch.chunk(x_attn_mask, self.sp_world_size, dim=1)[self.sp_rank]
        tmp_x_RoPE_embeds = torch.chunk(x_RoPE_embeds, self.sp_world_size, dim=1)[self.sp_rank]
        # x refiner
        x = AsyncModeStart.apply(x)
        for layer in self.noise_refiner:
            x = AsyncSync.apply(x, False, True)
            x = layer(
                hidden_states=x,
                attn_mask=tmp_x_attn_mask,
                RoPE_embeds=tmp_x_RoPE_embeds,
                adaln_input=adaln_input,
            )
        x = AsyncModeEnd.apply(x)
        
        x = SequenceAllGather.apply(x, self.sp_group_model)

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

        text_embedding = SequenceScatter.apply(text_embedding, self.sp_group_model)
        tmp_text_attn_mask = torch.chunk(text_attn_mask, self.sp_world_size, dim=1)[self.sp_rank]
        tmp_text_RoPE_embeds = torch.chunk(text_RoPE_embeds, self.sp_world_size, dim=1)[self.sp_rank]
        # text refiner
        text_embedding = AsyncModeStart.apply(text_embedding)
        for layer in self.context_refiner:
            text_embedding = AsyncSync.apply(text_embedding, False, True)
            text_embedding = layer(
                hidden_states=text_embedding,
                attn_mask=tmp_text_attn_mask,
                RoPE_embeds=tmp_text_RoPE_embeds,
            )
        text_embedding = AsyncModeEnd.apply(text_embedding)

        text_embedding = SequenceAllGather.apply(text_embedding, self.sp_group_model)
        # cond process
        if cond_list is not None:
            # cond_padded_list, cond_RoPE_embeds_list, cond_item_seqlens_list
            c_p_list, c_RoPE_e_list, c_i_s_list = [], [], []
            for i in range(len(cond_list)):
                (
                    cond_padded,
                    cond_RoPE_embeds,
                    cond_item_seqlens
                ) = self.cond_refine(
                    cond_padded=cond_out_dict["c_p"][i],
                    cond_pad_mask=cond_out_dict["c_p_m"][i],
                    cond_padded_pos_ids=cond_out_dict["c_p_p_id"][i],
                    adaln_input=adaln_input,
                    cond_idx=i+1,
                    device=device,
                )
                c_p_list.append(cond_padded)
                c_RoPE_e_list.append(cond_RoPE_embeds)
                c_i_s_list.append(cond_item_seqlens)

            # unified sequence prepare
            unified = []
            unified_RoPE_embeds = []
            cond_seq_lens = []
            for i in range(bsz):
                x_len = x_item_seqlens[i]
                text_len = text_item_seqlens[i]
                cond_seq_lens.append(0)

                cond_padded = []
                cond_RoPE_embeds = []
                for j, (ith_c_p, ith_c_RoPE_e, ith_c_i_lens) in enumerate(
                    zip(c_p_list, c_RoPE_e_list, c_i_s_list)
                ):
                    # print(f"ith_c_p shape {ith_c_p.shape}")
                    # print(f"ith_c_RoPE_e shape {ith_c_RoPE_e.shape}")

                    ith_c_i_len = ith_c_i_lens[i]
                    cond_padded.append(ith_c_p[i][:ith_c_i_len])
                    cond_RoPE_embeds.append(ith_c_RoPE_e[i][:ith_c_i_len])
                    cond_seq_lens[i] += ith_c_i_len

                cond_padded = torch.concat(cond_padded, dim=0)
                # print(f"cond_padded shape {cond_padded.shape}")
                cond_RoPE_embeds = torch.concat(cond_RoPE_embeds, dim=0)
                unified.append(torch.cat([x[i][:x_len], text_embedding[i][:text_len], cond_padded]))
                unified_RoPE_embeds.append(torch.cat([x_RoPE_embeds[i][:x_len], text_RoPE_embeds[i][:text_len], cond_RoPE_embeds]))
            
            unified_item_seqlens = [a + b + c for a, b, c in zip(text_item_seqlens, x_item_seqlens, cond_seq_lens)]
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

        # print(f"unified shape {unified.shape}")
        # backbone
        unified = SequenceScatter.apply(unified, self.sp_group_model)
        adaln_input = SPBroadcast.apply(adaln_input, self.sp_group_model)
        unified_attn_mask = torch.chunk(unified_attn_mask, self.sp_world_size, dim=1)[self.sp_rank]
        unified_RoPE_embeds = torch.chunk(unified_RoPE_embeds, self.sp_world_size, dim=1)[self.sp_rank]

        # print(f"unified shape: {unified.shape}, ")
        unified = AsyncModeStart.apply(unified)
        for idx, layer in enumerate(self.layers):
            if idx % 1 == 0:
                unified = AsyncSync.apply(unified, False, True)
            unified = layer(
                hidden_states=unified,
                attn_mask=unified_attn_mask,
                RoPE_embeds=unified_RoPE_embeds,
                adaln_input=adaln_input,
            )
        unified = AsyncModeEnd.apply(unified)

        unified = SequenceAllGather.apply(unified, self.sp_group_model)

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
        cond_index_list = [1, 2]
        for module_idx in cond_index_list:
            cond_embedder = getattr(self, f"cond_embedder_{module_idx}", None)
            for param in cond_embedder.parameters():
                param.requires_grad_(grad_stat)
                total_switched_params += param.numel()
            
            cond_refiner = getattr(self, f"cond_refiner_{module_idx}", None)
            for param in cond_refiner.parameters():
                param.requires_grad_(grad_stat)
                total_switched_params += param.numel()
        
        return total_switched_params
    
    def cond_refine(
        self,
        cond_padded,
        cond_pad_mask,
        cond_padded_pos_ids,
        adaln_input,
        cond_idx,
        device,
        ):
        cond_embedder = getattr(self, f"cond_embedder_{cond_idx}")
        cond_refiner = getattr(self, f"cond_refiner_{cond_idx}")

        cond_item_seqlens = [len(_) for _ in cond_padded]
        cond_max_item_seqlen = max(cond_item_seqlens)
        bsz = len(cond_item_seqlens)

        cond_padded = torch.cat(cond_padded, dim=0)
        cond_padded = cond_embedder(cond_padded)
        cond_padded[torch.cat(cond_pad_mask)] = self.cond_pad_token
        cond_padded = list(cond_padded.split(cond_item_seqlens, dim=0))
        cond_RoPE_embeds = list(
            self.z_transformer.rope_embedder(
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
        cond_padded = SequenceScatter.apply(cond_padded, self.sp_group_model)
        adaln_input = SPBroadcast.apply(adaln_input, self.sp_group_model)
        tmp_cond_attn_mask = torch.chunk(cond_attn_mask, self.sp_world_size, dim=1)[self.sp_rank]
        tmp_cond_RoPE_embeds = torch.chunk(cond_RoPE_embeds, self.sp_world_size, dim=1)[self.sp_rank]
        # cond refiner
        cond_padded = AsyncModeStart.apply(cond_padded)
        for layer in cond_refiner:
            cond_padded = AsyncSync.apply(cond_padded, False, True)
            cond_padded = layer(
                hidden_states=cond_padded,
                attn_mask=tmp_cond_attn_mask,
                RoPE_embeds=tmp_cond_RoPE_embeds,
                adaln_input=adaln_input,
            )
        cond_padded = AsyncModeEnd.apply(cond_padded)
        cond_padded = SequenceAllGather.apply(cond_padded, self.sp_group_model)
        return cond_padded, cond_RoPE_embeds, cond_item_seqlens