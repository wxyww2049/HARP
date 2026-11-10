import torch


ACCELERATE_OPS = {
    "kp_linear_forward.default",
    "ring_attention_forward.default",
    "rmsnorm_forward_out.default",
    "rope_forward_impl.default",
    "silu_out.default",
    "tanh.default",

    "kp_linear_backward.default",
    "ring_attention_backward.default",
    "rmsnorm_backward_out.default",
    "rope_backward_impl.default",
    "silu_backward_out.default",
    "tanh_backward.default",
    "copy_.default",
    "add_.Tensor",

    "mul.Tensor",
    "add.Tensor",
    "sum.dim_IntList",
    "cat.default",
    "full.default",
    "_to_copy.default",
}

NO_SYNC_OPS = {
    "ones_like.default",
    "zeros_like.default",
    "empty_like.default",
    "empty.memory_format",
    "empty_strided.default",
    "new_empty_strided.default",
    "ones.default",
    "zeros.default",

    "unsqueeze.default",
    "squeeze.dim",
    "permute.default",
    "view.default",
    "view_as_complex.default",
    "t.default",
    "transpose.int",
    "unflatten.int",

    "detach.default",
    "chunk.default",
    "split.Tensor",
    "slice.Tensor",
}


class MyMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        out = torch.empty(torch.broadcast_shapes(a.shape, b.shape),
                          dtype=a.dtype,
                          device=a.device)
        torch.ops.aten.mul.out(a, b, out=out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_output * b
            grad_a = grad_a.sum_to_size(a.shape)

        if ctx.needs_input_grad[1]:
            grad_b = grad_output * a
            grad_b = grad_b.sum_to_size(b.shape)

        return grad_a, grad_b


class MyAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a_is_tensor = isinstance(a, torch.Tensor)
        ctx.a_is_tensor = a_is_tensor

        if a_is_tensor:
            ctx.save_for_backward(a, b)
            ctx.a_shape = a.shape
            out_shape = torch.broadcast_shapes(a.shape, b.shape)
            out = torch.empty(out_shape, dtype=b.dtype, device=b.device)
            torch.ops.aten.add.out(a, b, out=out)
            return out
        else:
            ctx.save_for_backward(b)
            ctx.a = a
            out = torch.empty(b.shape, dtype=b.dtype, device=b.device)
            torch.ops.aten.add.out(a, b, out=out)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = grad_b = None

        if ctx.a_is_tensor:
            a, b = ctx.saved_tensors

            if ctx.needs_input_grad[0]:
                grad_a = grad_output.sum_to_size(a.shape)

            if ctx.needs_input_grad[1]:
                grad_b = grad_output.sum_to_size(b.shape)
        else:
            (b,) = ctx.saved_tensors

            grad_a = None

            if ctx.needs_input_grad[1]:
                grad_b = grad_output.sum_to_size(b.shape)

        return grad_a, grad_b


class MyTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputx):
        output = torch.empty_like(inputx)
        torch.ops.async_op.tanh_out(inputx, output)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = torch.ops.aten.tanh_backward(grad_output, output)
        return grad_input
    

class MySiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputx):
        output = torch.empty_like(inputx)
        torch.ops.async_op.silu_out(inputx, output)
        ctx.save_for_backward(inputx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (inputx,) = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        torch.ops.async_op.silu_backward_out(grad_output, inputx, grad_input)
        return grad_input


class ApplyRotaryComplexFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        x_dtype = x.dtype
        x_shape = x.shape
        y = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
        torch.ops.async_op.rope_forward_impl(x, freqs_cis, y)
        ctx.save_for_backward(freqs_cis)
        ctx.x_dtype = x_dtype
        ctx.x_shape = x_shape
        return y

    @staticmethod
    def backward(ctx, grad_output):
        freqs_cis = ctx.saved_tensors[0]
        grad_x = torch.empty_strided(
            ctx.x_shape,
            grad_output.stride(),
            dtype=ctx.x_dtype,
            device=grad_output.device,
        )
        torch.ops.async_op.rope_backward_impl(grad_output, freqs_cis, grad_x)
        return grad_x, None


def mm_kernel(output, input, weight, bias):
    torch.ops.xhpops.kp_linear_forward(
        input,
        weight,
        bias,
        output
    )
    return 

def mm_backward_kernel(grad_input, grad_output_flat, inputx, weight, grad_weight, grad_bias):
    torch.ops.xhpops.kp_linear_backward(
        grad_output_flat,
        inputx,
        weight,
        grad_input,
        grad_weight,
        grad_bias
    )
    return 

def silu_kernel(output, input):
    torch.ops.async_op.silu_out(input, output)
    return 

def silu_backward_kernel(grad_input, grad_output, input):
    torch.ops.async_op.silu_backward_out(grad_output, input, grad_input)
    return

def mul_kernel(output, input_a, input_b):
    torch.ops.aten.mul.out(input_a, input_b, out=output)
    return

def add_kernel(output, input_a, input_b):
    torch.ops.aten.add.out(input_a, input_b, out=output)
    return

def mul_fast(output, input_a, input_b):
    torch.ops.async_op.mul_out(input_a, input_b, output)
    return

def add_fast(output, input_a, input_b):
    torch.ops.async_op.add_out(input_a, input_b, output)
    return

def rms_norm_kernel(output, inputx, weight, rms, eps):
    torch.ops.async_op.rmsnorm_forward_out(inputx, weight, output, rms, eps)
    return

def rmsnorm_backward_kernel(grad_input, input_bf16, grad_output, weight_bf16, sve_rms, grad_weight):
    torch.ops.async_op.rmsnorm_backward_out(input_bf16, grad_output, weight_bf16, sve_rms, grad_input, grad_weight)
    return

def tanh_kernel(output, inputx):
    torch.ops.async_op.tanh_out(inputx, output)
    return

def tanh_backward_kernel(grad_input, grad_output, output):
    torch.ops.async_op.tanh_backward_out(grad_output, output, grad_input)
    return

def rope_forward_kernel(output, inputx, freqs_cis):
    torch.ops.async_op.rope_forward_impl(inputx, freqs_cis, output)
    return

def rope_backward_kernel(grad_input, grad_output, freqs_cis):
    torch.ops.async_op.rope_backward_impl(grad_output, freqs_cis, grad_input)
    return

def copy_kernel(_self):
    # torch.ops.aten.copy_(_self, _src)
    if hasattr(_self, "ready_event"):
        _self.ready_event.set()
    return 

def add_inplace_kernel(_self, _other):
    torch.ops.aten.add_(_self, _other)
    return

def sum_dim_kernel(output, _self, dim, keepdim):
    torch.ops.aten.sum.IntList_out(_self, dim=dim, keepdim=keepdim, out=output)
    return

def cat_kernel(output, tensors, dim):
    torch.ops.aten.cat.out(tensors, dim, out=output)
    return 

def full_kernel(output, size, fill_value):
    torch.ops.aten.full.out(size, fill_value, out=output)
    return 

def to_copy_kernel(output, input):
    torch.ops.aten._to_copy.out(input, out=output)
    return

def ring_attention_forward_kernel(out, local_q, local_k, local_v, m_i, l_i, sp_size, scale):
    torch.ops.ring_attention_ops.ring_attention_forward(local_q, local_k, local_v, out, m_i, l_i, sp_size, scale)
    return

def ring_attention_backward_kernel(grad_q, local_q, local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale):
    torch.ops.ring_attention_ops.ring_attention_backward(local_q, grad_q, local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale)
    return
