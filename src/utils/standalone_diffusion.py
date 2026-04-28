import json
import logging
import math
import os
from typing import Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConfigDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class FlowMatchEulerDiscreteScheduler:
    config_name = "scheduler_config.json"
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
        **extra_config,
    ):
        self.config = ConfigDict(
            {
                "_class_name": self.__class__.__name__,
                "num_train_timesteps": num_train_timesteps,
                "shift": shift,
                "use_dynamic_shifting": use_dynamic_shifting,
                "base_shift": base_shift,
                "max_shift": max_shift,
                "base_image_seq_len": base_image_seq_len,
                "max_image_seq_len": max_image_seq_len,
                "invert_sigmas": invert_sigmas,
                "shift_terminal": shift_terminal,
                "use_karras_sigmas": use_karras_sigmas,
                "use_exponential_sigmas": use_exponential_sigmas,
                "use_beta_sigmas": use_beta_sigmas,
                "time_shift_type": time_shift_type,
                "stochastic_sampling": stochastic_sampling,
            }
        )
        self.config.update(extra_config)

        timesteps = torch.linspace(num_train_timesteps, 1, num_train_timesteps, dtype=torch.float32)
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self._step_index = None
        self._begin_index = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None):
        config_path = os.path.expanduser(pretrained_model_name_or_path)
        if subfolder is not None:
            config_path = os.path.join(config_path, subfolder)
        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, cls.config_name)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(**config)


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_piecewise_constant_schedule(optimizer: Optimizer, step_rules: str, last_epoch: int = -1) -> LambdaLR:
    rules_dict = {}
    rule_list = step_rules.split(",")
    for rule_str in rule_list[:-1]:
        value_str, steps_str = rule_str.split(":")
        rules_dict[int(steps_str)] = float(value_str)
    last_lr_multiple = float(rule_list[-1])

    def rule_func(steps: int) -> float:
        for sorted_step in sorted(rules_dict.keys()):
            if steps < sorted_step:
                return rules_dict[sorted_step]
        return last_lr_multiple

    return LambdaLR(optimizer, rule_func, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init

        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_with_warmup": get_constant_schedule_with_warmup,
    "piecewise_constant": get_piecewise_constant_schedule,
}


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    name = name.lower()
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == "constant":
        return schedule_func(optimizer, last_epoch=last_epoch)

    if name == "piecewise_constant":
        return schedule_func(optimizer, step_rules=step_rules, last_epoch=last_epoch)

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == "constant_with_warmup":
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, last_epoch=last_epoch)

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == "cosine_with_restarts":
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            last_epoch=last_epoch,
        )

    if name == "polynomial":
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
        )

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch,
    )


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def set_library_verbosity_info():
    logging.getLogger(__name__).setLevel(logging.INFO)


def set_library_verbosity_error():
    logging.getLogger(__name__).setLevel(logging.ERROR)

