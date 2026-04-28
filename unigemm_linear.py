import math

import torch
import torch.nn as nn

from managers.mem_manager import use_hbm_allocator

class CustomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputx, weight, bias=None):
        input_shape = inputx.shape
        input_flat = inputx.view(-1, inputx.shape[-1])

        ctx.save_for_backward(input_flat, weight, bias)
        ctx.input_shape = input_shape
        batch_size = input_flat.size(0)
        out_features = weight.size(0)
        output = torch.empty(
            *input_shape[:-1],
            out_features,
            dtype=torch.bfloat16,
            device=inputx.device
        )
        output_flat = output.view(batch_size, out_features)
        torch.ops.xhpops.kp_linear_forward(
            input_flat,
            weight,
            bias,
            output_flat
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        inputx, weight, bias = ctx.saved_tensors
        
        grad_output = grad_output.to(torch.bfloat16)
        grad_output_flat = grad_output.view(-1, grad_output.shape[-1])

        grad_input = torch.empty_like(inputx)
        grad_bias = torch.empty_like(bias) if bias is not None else None
        grad_weight = torch.empty_like(weight)

        torch.ops.xhpops.kp_linear_backward(
            grad_output_flat,
            inputx,
            weight,
            grad_input,
            grad_weight,
            grad_bias
        )
        grad_input = grad_input.view(*ctx.input_shape)

        return grad_input, grad_weight, grad_bias

class kpLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)


    def reset_parameters(self):

        w = torch.empty_like(self.weight, dtype=torch.float32)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight.data.copy_(w.to(torch.bfloat16))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty_like(self.bias, dtype=torch.float32)
            nn.init.uniform_(b, -bound, bound)
            self.bias.data.copy_(b.to(torch.bfloat16))

    def forward(self, x):
        x = x.to(torch.bfloat16)
        return CustomLinearFunction.apply(
            x,
            self.weight,
            self.bias
        )


def replace_linear_with_kplinear(model, kplinear_class):
    for name, module in list(model.named_children()):
        if isinstance(module, torch.nn.Linear):
            new_layer = kplinear_class(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )   
            if module.weight is not None:
                new_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None and new_layer.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)
            setattr(model, name, new_layer)
        else:
             replace_linear_with_kplinear(module, kplinear_class)
    return model