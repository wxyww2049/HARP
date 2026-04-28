from typing import Union, List

from transformers import AutoTokenizer, Qwen3ForCausalLM

import torch

@torch.no_grad
def encode_prompt(
    prompt: Union[str, List[str]],
    tokenizer,
    text_encoder,
    max_sequence_length: int = 512,
):
    device = text_encoder.device

    if isinstance(prompt, str):
        prompt = [prompt]

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()
    
    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []

    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list

def load_z_image_text_processer(ckpt_path, device, weight_dtype, requires_grad: bool = False):
    text_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        subfolder="tokenizer",
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ckpt_path,
        subfolder="text_encoder",
    ).to(device, dtype=weight_dtype)

    text_encoder.requires_grad_(requires_grad)

    return text_tokenizer, text_encoder