import inspect
from typing import List, Optional, Union, Dict, Any

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers import (
    ZImageTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def prepare_latents(
    batch_size,
    num_channels_latents,
    latent_height,
    latent_width,
    dtype,
    device,
    generator,
    latents=None,
):
    # height = 2 * (int(height) // (self.vae_scale_factor * 2))
    # width = 2 * (int(width) // (self.vae_scale_factor * 2))
    shape = (batch_size, num_channels_latents, latent_height, latent_width)
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        if latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents.to(device)
    return latents

@torch.no_grad()
def batch_image_inference(
    transformer: ZImageTransformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    image_processor: VaeImageProcessor,
    prompt_embeds: List[torch.FloatTensor],
    negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 5.0,
    cfg_normalization: bool = False,
    cfg_truncation: float = 1.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    initial_latents: Optional[torch.Tensor] = None,
    vae_scale_factor: Optional[int] = None, 
):
    if vae_scale_factor is None:
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_scale = vae_scale_factor * 2
    device = transformer.device
    _dtype = prompt_embeds[0].dtype
    batch_size = len(prompt_embeds)

    if initial_latents is None:
        latents_height = 2 * (int(height) // (vae_scale_factor * 2))
        latents_width = 2 * (int(width) // (vae_scale_factor * 2))
        num_channels_latents = transformer.in_channels
        latents = prepare_latents(
            batch_size,
            num_channels_latents,
            latents_height,
            latents_width,
            _dtype,
            device,
            generator,
        )
    else:
        latents = initial_latents

    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    if negative_prompt_embeds is None:
        use_cfg = False
    else:
        use_cfg = True

    with tqdm(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end)
            t_norm = timestep[0].item()

            if use_cfg:
                current_guidance_scale = guidance_scale
                if t_norm > cfg_truncation:
                    current_guidance_scale = 0.0
                latent_model_input = latents.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep
            
            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            # transformer forward
            model_out_list = transformer(
                x=latent_model_input_list,
                t=timestep_model_input,
                text_embedding=prompt_embeds_model_input,
                return_dict=False,
            )[0]

            if use_cfg:
                pos_out = model_out_list[:batch_size]
                neg_out = model_out_list[batch_size:]
                noise_pred = []

                for j in range(batch_size):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()
                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization
                    if cfg_normalization:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(cfg_normalization)
                        if new_pos_norm > max_new_norm:
                            pred = pred * (max_new_norm / new_pos_norm)
                        
                    noise_pred.append(pred.to(_dtype))

                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t for t in model_out_list], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents != _dtype:
                latents = latents.to(dtype=_dtype)
                
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        latents = latents.to(_dtype)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        image = vae.decode(latents, return_dict=False)[0]
        image = image_processor.postprocess(image, output_type=output_type)

    if not return_dict:
        return (image,)

    return ZImagePipelineOutput(images=image)