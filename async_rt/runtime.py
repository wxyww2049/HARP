import os
import threading
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from async_rt import async_ops

import sys
sys.path.insert(0, "kernels")

import cpp_async_runtime


import time
def _set_thread_affinity(core_id):
    if core_id is not None:
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, {core_id})


def infer_sum_shape(shape, dim=None, keepdim=False):
    if isinstance(dim, int):
        dim = [dim]

    dim = set([d if d >= 0 else d + len(shape) for d in dim])

    out_shape = []
    for i in range(len(shape)):
        if i in dim:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(shape[i])

    return tuple(out_shape)


class CppRuntimeWrapper:
    def __init__(self):
        self.impl = cpp_async_runtime.get_runtime()
        self.rank = self.impl.rank

    def sync(self):
        self.impl.sync()

    def shutdown(self):
        self.impl.shutdown()

    def bind(self, core_id):
        self.impl.bind_worker(0, core_id)


RUNTIME = CppRuntimeWrapper()


class AsyncRunMode(TorchDispatchMode):
    mode_enable = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = func.__name__

        if AsyncRunMode.mode_enable is False:
            return func(*args, **kwargs)

        if op_name in async_ops.ACCELERATE_OPS:

            if op_name == "kp_linear_forward.default":
                input, weight, bias, output = args
                if hasattr(weight,"state") and hasattr(weight,"chunk_part"):
                    RUNTIME.impl.submit_wait_event(weight.state)    
                RUNTIME.impl.submit_kp_linear_forward(output, input, weight, bias)
                return None

            elif op_name == "kp_linear_backward.default":
                grad_output_flat, inputx, weight, grad_input, grad_weight, grad_bias = args
                RUNTIME.impl.submit_kp_linear_backward(
                    grad_input, grad_output_flat, inputx, weight, grad_weight, grad_bias
                )
                return None

            elif op_name == "ring_attention_forward.default":
                local_q, local_k, local_v, out, m_i, l_i, sp_size, scale = args
                return RUNTIME.impl.submit_ring_attention_forward(
                    out, local_q, local_k, local_v, m_i, l_i, sp_size, scale
                )

            elif op_name == "ring_attention_backward.default":
                local_q, grad_q, local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale = args
                return RUNTIME.impl.submit_ring_attention_backward(
                    grad_q, local_q, local_k, grad_k, local_v, grad_v, grad_out, m_i, l_i, D, sp_size, scale
                )

            elif op_name == "rmsnorm_forward_out.default":
                inputx, weight, output, rms, eps = args
                if hasattr(weight,"state") and hasattr(weight,"chunk_part"):
                    RUNTIME.impl.submit_wait_event(weight.state)
                RUNTIME.impl.submit_rmsnorm_forward(output, inputx, weight, rms, eps)
                return None

            elif op_name == "rmsnorm_backward_out.default":
                input_bf16, grad_output, weight_bf16, sve_rms, grad_input, grad_weight = args
                RUNTIME.impl.submit_rmsnorm_backward(
                    grad_input, input_bf16, grad_output, weight_bf16, sve_rms, grad_weight
                )
                return None

            elif op_name == "rope_forward_impl.default":
                inputx, freqs_cis, output = args
                RUNTIME.impl.submit_rope_forward(output, inputx, freqs_cis)
                return None

            elif op_name == "rope_backward_impl.default":
                grad_output, freqs_cis, grad_input = args
                RUNTIME.impl.submit_rope_backward(grad_input, grad_output, freqs_cis)
                return None

            elif op_name == "silu_out.default":
                inputx, output = args
                RUNTIME.impl.submit_silu(output, inputx)
                return None

            elif op_name == "silu_backward_out.default":
                grad_output, inputx, grad_input = args
                RUNTIME.impl.submit_silu_backward(grad_input, grad_output, inputx)
                return None

            elif op_name == "tanh.default":
                inputx = args[0]
                output = torch.empty_like(inputx)
                RUNTIME.impl.submit_tanh(output.detach(), inputx.detach())
                return output

            elif op_name == "tanh_backward.default":
                grad_output, output = args
                grad_input = torch.empty_like(output)
                RUNTIME.impl.submit_tanh_backward(
                    grad_input.detach(), grad_output.detach(), output.detach()
                )
                return grad_input

            elif op_name == "mul.Tensor":
                input_a, input_b = args
                if isinstance(input_b, torch.Tensor) and isinstance(input_a, torch.Tensor):
                    output = torch.empty(
                        torch.broadcast_shapes(input_a.shape, input_b.shape),
                        dtype=input_a.dtype,
                        device=input_a.device,
                    )
                    if input_a.shape == input_b.shape:
                        RUNTIME.impl.submit_mul_fast(output, input_a, input_b)
                    else:
                        RUNTIME.impl.submit_mul(output.detach(), input_a.detach(), input_b.detach())
                else:
                    if isinstance(input_a, torch.Tensor):
                        output = torch.empty_like(input_a)
                        RUNTIME.impl.submit_mul(output.detach(), input_a.detach(), input_b)
                    else:
                        output = torch.empty_like(input_b)
                        RUNTIME.impl.submit_mul(output.detach(), input_a, input_b.detach())
                return output

            elif op_name == "add.Tensor":
                input_a, input_b = args
                if isinstance(input_b, torch.Tensor) and isinstance(input_a, torch.Tensor):
                    output = torch.empty(
                        torch.broadcast_shapes(input_a.shape, input_b.shape),
                        dtype=input_a.dtype,
                        device=input_a.device,
                    )
                    if input_a.shape == input_b.shape:
                        RUNTIME.impl.submit_add_fast(output, input_a, input_b)
                    else:
                        RUNTIME.impl.submit_add(output.detach(), input_a.detach(), input_b.detach())
                else:
                    if isinstance(input_a, torch.Tensor):
                        output = torch.empty_like(input_a)
                        RUNTIME.impl.submit_add_scalar(output.detach(), input_a.detach(), input_b)
                    else:
                        output = torch.empty_like(input_b)
                        RUNTIME.impl.submit_add_scalar(output.detach(), input_b.detach(), input_a)
                return output

            elif op_name == "copy_.default":
                _self, _src = args
                _self.set_(
                    source=_src.untyped_storage(),
                    storage_offset=_src.storage_offset(),
                    size=_src.size(),
                    stride=_src.stride(),
                )
                _self.ready_event = RUNTIME.impl.submit_copy_ready_only()
                del _src
                return _self

            elif op_name == "add_.Tensor":
                _self, _other = args
                if _self.shape == _other.shape:
                    RUNTIME.impl.submit_add_fast(_self, _self, _other)
                else:
                    RUNTIME.impl.submit_add_inplace(_self, _other)
                return _self

            elif op_name == "sum.dim_IntList":
                _self, dim, keepdim = args
                return RUNTIME.impl.submit_sum_dim(_self, list(dim), keepdim).detach()

            elif op_name == "cat.default":
                tensors, dim = args
                return RUNTIME.impl.submit_cat(list(tensors), dim).detach()

            elif op_name == "full.default":
                size, full_value = args[0], args[1]
                device, dtype = kwargs["device"], kwargs["dtype"]
                return RUNTIME.impl.submit_full(list(size), full_value, dtype, device).detach()

            elif op_name == "_to_copy.default":
                _self = args[0]
                dtype = kwargs["dtype"]
                return RUNTIME.impl.submit_to_copy(_self.detach(), dtype).detach()

        if op_name not in async_ops.NO_SYNC_OPS:
            RUNTIME.sync()

        return func(*args, **kwargs)


GAsyncRunMode = AsyncRunMode()


class AsyncModeStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x):
        AsyncRunMode.mode_enable = True
        _set_thread_affinity(RUNTIME.rank % 16 * 38 + 36)
        return input_x

    @staticmethod
    def backward(ctx, grad_x):
        AsyncRunMode.mode_enable = False
        RUNTIME.sync()
        _set_thread_affinity(RUNTIME.rank % 16 * 38)

        return grad_x


class AsyncModeEnd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x):
        AsyncRunMode.mode_enable = False
        RUNTIME.sync()
        _set_thread_affinity(RUNTIME.rank % 16 * 38)
        return input_x

    @staticmethod
    def backward(ctx, grad_x):
        AsyncRunMode.mode_enable = True
        _set_thread_affinity(RUNTIME.rank % 16 * 38 + 36)
        return grad_x


class AsyncSync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_x, forward_sync, backward_sync):
        if forward_sync is True:
            RUNTIME.sync()
        ctx.backward_sync = backward_sync
        return input_x

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.backward_sync is True:
            RUNTIME.sync()
        return grad_x, None, None