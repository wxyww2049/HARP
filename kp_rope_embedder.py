import os
from typing import List

import torch

class RopeEmbedderReference:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for d, e in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.dtype == torch.int32
        assert ids.device.type == "cpu"
        assert ids.shape[-1] == len(self.axes_dims)

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)

        result = []
        for axis in range(len(self.axes_dims)):
            index = ids[:, axis]
            result.append(self.freqs_cis[axis][index])
        return torch.cat(result, dim=-1)


class RopeEmbedderSVE(RopeEmbedderReference):
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        super().__init__(theta=theta, axes_dims=axes_dims, axes_lens=axes_lens)
        self.freqs_cis = None

    def _ensure_tables(self):
        if self.freqs_cis is None:
            self.freqs_cis = list(
                torch.ops.wkrope.precompute_freqs_cis_impl(self.axes_dims, self.axes_lens, float(self.theta))
            )
        return self.freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.dtype == torch.int32
        assert ids.device.type == "cpu"
        assert ids.shape[-1] == len(self.axes_dims)
        return torch.ops.wkrope.rope_lookup_impl(ids.contiguous(), self._ensure_tables())

def replace_ropeembedder_with_sverope(model, rope_class=RopeEmbedderSVE):
    rope_embedder = getattr(model, "rope_embedder", None)
    if rope_embedder is not None and not isinstance(rope_embedder, rope_class):
        theta = getattr(rope_embedder, "theta")
        axes_dims = getattr(rope_embedder, "axes_dims")
        axes_lens = getattr(rope_embedder, "axes_lens")
        setattr(
            model,
            "rope_embedder",
            rope_class(theta=theta, axes_dims=axes_dims, axes_lens=axes_lens),
        )

    for _, module in list(model.named_children()):
        replace_ropeembedder_with_sverope(module, rope_class)
    return model
