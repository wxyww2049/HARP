from typing import List, Tuple, Optional

import torch

SEQ_MULTI_OF = 32

class Z_Mixin:
    def unpatchify(
        self,
        x: List[torch.Tensor],
        size: List[Tuple],
        patch_size,
        f_patch_size,
        out_channels,
    ) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        # assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(out_channels, F, H, W)
            )
        return x
    
    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def cond_patchify_and_embed(
        self,
        cond: torch.Tensor,
        patch_size: int,
        start: int,
        device,
    ):
        pH = pW = patch_size

        C, H, W = cond.size()
        H_tokens, W_tokens = H // pH, W // pW
        cond = cond.view(C, H_tokens, pH, W_tokens, pW)
        cond = cond.permute(1, 3, 2, 4, 0).reshape(H_tokens * W_tokens, pH * pW * C)
        cond_ori_len = len(cond)
        cap_padding_len = (-cond_ori_len) % SEQ_MULTI_OF
        
        cond_ori_pos_ids = self.create_coordinate_grid(
            size=(1, H_tokens, W_tokens),
            start=(start, 0, 0),
            device=device,
        ).flatten(0, 2)
        cond_padded_pos_ids = torch.cat(
            [
                cond_ori_pos_ids,
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(cap_padding_len, 1),
            ],
            dim=0,
        )
        # pad mask
        cond_pad_mask = torch.cat(
            [
                torch.zeros((cond_ori_len,), dtype=torch.bool, device=device),
                torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
            ],
            dim=0,
        )
        cond_padded = torch.cat(
            [cond, cond[-1:].repeat(cap_padding_len, 1)],
            dim=0,
        )
        return cond_padded, cond_pad_mask, cond_padded_pos_ids


    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
        cond_list: Optional[List[torch.Tensor]]
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []
        cond_out_list = None

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask if cap_padding_len > 0 else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)
            )

            # padded feature
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padded_pos_ids = torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

            if cond_list is not None:
                if cond_out_list is None:
                    cond_out_list = ([], [], []) # (cond_padded, cond_pad_mask, cond_padded_pos_ids)

                cond_padded, cond_pad_mask, cond_padded_pos_ids = self.cond_patchify_and_embed(
                        cond_list[i], patch_size, cap_ori_len + cap_padding_len + 2, device
                )
                cond_out_list[0].append(cond_padded)
                cond_out_list[1].append(cond_pad_mask)
                cond_out_list[2].append(cond_padded_pos_ids)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
            cond_out_list,
        )


