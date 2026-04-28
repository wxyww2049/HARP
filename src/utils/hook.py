import os
from typing import List
from safetensors.torch import save_file

def save_lora_adapter_hook(models, weights, output_dir, accelerator, target_adapters: List[str]):
    if accelerator.is_main_process:
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            # Save lora modules
            for target_adapter in target_adapters:
                unwrapped.save_lora_adapter(
                    save_directory=os.path.join(output_dir, target_adapter),
                    adapter_name=target_adapter,
                )

            # Save model params
            trainable_sd = {
                k: v
                for k, v in unwrapped.named_parameters()
                if v.requires_grad and 'lora_' not in k
            }
            # print(trainable_sd)
            if len(trainable_sd) !=0:
                save_file(trainable_sd, os.path.join(output_dir, 'model.safetensors'))
            if weights:
                weights.pop()

def save_trainable_param_hook(models, weights, output_dir, accelerator):
    if accelerator.is_main_process:
        for model in models:
            unwrapped = accelerator.unwrap_model(model)
            trainable_sd = {
                k: v
                for k, v in unwrapped.named_parameters()
                if v.requires_grad and 'lora_' not in k
            }
            
            save_file(trainable_sd, os.path.join(output_dir, 'model.safetensors'))
            if weights:
                weights.pop()