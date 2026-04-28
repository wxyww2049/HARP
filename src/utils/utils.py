from contextlib import contextmanager
import torch

def encode_images(vae: torch.nn.Module, pixels: torch.Tensor, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

@contextmanager
def preserve_requires_grad(model):
    requires_grad_backup = {name: param.requires_grad for name, param in model.named_parameters()}
    yield

    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_backup[name]