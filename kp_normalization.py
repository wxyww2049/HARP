import numbers

import torch
import torch.nn as nn
from src.models.z_image.standalone_transformer import RMSNorm


class CustomNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputx, weight, eps):
        output = torch.empty_strided(inputx.size(), inputx.stride(), dtype=inputx.dtype, device=inputx.device)
        rows = int(inputx.numel() / inputx.shape[-1])
        rms = torch.empty((rows,), device=inputx.device, dtype=torch.float32)
        torch.ops.async_op.rmsnorm_forward_out(inputx, weight, output, rms, eps)
        ctx.save_for_backward(inputx, weight, rms)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_bf16, weight_bf16, sve_rms = ctx.saved_tensors
        grad_input = torch.empty_strided(input_bf16.size(), input_bf16.stride(), dtype=input_bf16.dtype, device=input_bf16.device)
        grad_weight = torch.zeros((input_bf16.shape[-1],), dtype=input_bf16.dtype, device=input_bf16.device)
        torch.ops.async_op.rmsnorm_backward_out(input_bf16, grad_output, weight_bf16, sve_rms, grad_input, grad_weight)
        return grad_input, grad_weight, None

class kpRMSNorm(nn.Module):

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None
        self.bias = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                raise NotImplementedError("Bias is not implemented for kpRMSNorm yet.")

    def forward(self, hidden_states):
        return CustomNormFunction.apply(hidden_states, self.weight, self.eps)


def _is_rmsnorm_module(module: nn.Module) -> bool:
    if isinstance(module, RMSNorm):
        return True
    if isinstance(module, nn.RMSNorm):
        print("replacing nn.RMSNorm")
        return True
    return type(module).__name__ == "RMSNorm"


def _build_kprmsnorm_from(module: nn.Module) -> kpRMSNorm:
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)

    if hasattr(module, "dim"):
        dim = module.dim
    else:
        raise ValueError(f"Cannot infer RMSNorm dim from {type(module).__name__}.")

    elementwise_affine = getattr(module, "elementwise_affine")
    eps = getattr(module, "eps")
    if eps is None:
        raise ValueError(f"Cannot infer eps from {type(module).__name__}.")

    new_module = kpRMSNorm(
        dim=dim,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias is not None,
    )

    ref_tensor = weight if weight is not None else bias
    if ref_tensor is not None:
        new_module = new_module.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    new_module.train(module.training)

    if new_module.weight is not None and weight is not None:
        new_module.weight.data.copy_(weight.data)
        new_module.weight.requires_grad_(weight.requires_grad)

    return new_module


def replace_rmsNorm_with_kpRMSNorm(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if _is_rmsnorm_module(child):
            setattr(module, name, _build_kprmsnorm_from(child))
            continue
        replace_rmsNorm_with_kpRMSNorm(child)
    return module