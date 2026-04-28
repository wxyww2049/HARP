from typing import Tuple, Union
import torch
from managers.mem_manager import use_hbm_allocator
def kp_apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        raise NotImplementedError("kp_apply_rotary_emb not implement use_real")
    else:
        # with use_hbm_allocator(True):
        return torch.ops.kprope.rope_forward_impl(x, freqs_cis)