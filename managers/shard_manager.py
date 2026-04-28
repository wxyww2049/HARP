import torch
import torch.distributed as dist

class SequenceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, global_unified, sp_group):
        ctx.sp_group = sp_group
        rank = dist.get_rank(sp_group)
        world_size = dist.get_world_size(sp_group)
        
        chunks = torch.chunk(global_unified, world_size, dim=1)
        local_unified = chunks[rank]
        
        return local_unified

    @staticmethod
    def backward(ctx, grad_local_unified):
        world_size = dist.get_world_size(ctx.sp_group)
        
        gathered_grads = torch.empty(
            (world_size,) + tuple(grad_local_unified.shape),
            dtype=grad_local_unified.dtype,
            device=grad_local_unified.device,
        )
        dist.barrier(group=ctx.sp_group)
        dist.all_gather_into_tensor(gathered_grads, grad_local_unified, group=ctx.sp_group)
        
        if grad_local_unified.shape[0] == 1:
            target_shape = list(grad_local_unified.shape)
            target_shape[1] *= world_size
            grad_global_unified = gathered_grads.view(*target_shape)
        else:
            grad_global_unified = torch.cat(list(gathered_grads.unbind(0)), dim=1)
        return grad_global_unified, None

class SequenceAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_unified, sp_group):
        ctx.sp_group = sp_group
        
        world_size = dist.get_world_size(sp_group)
        gathered_tensors = torch.empty(
            (world_size,) + tuple(local_unified.shape),
            dtype=local_unified.dtype,
            device=local_unified.device,
        )

        dist.barrier(group=sp_group)
        dist.all_gather_into_tensor(gathered_tensors,local_unified, group=sp_group)
        if local_unified.shape[0] == 1:
            target_shape = list(local_unified.shape)
            target_shape[1] *= world_size
            global_unified = gathered_tensors.view(*target_shape)
        else:
            global_unified = torch.cat(list(gathered_tensors.unbind(0)), dim=1)
        return global_unified

    @staticmethod
    def backward(ctx, grad_global_unified):
        world_size = dist.get_world_size(ctx.sp_group)
        rank = dist.get_rank(ctx.sp_group)
        
        grad_chunks = torch.chunk(grad_global_unified, world_size, dim=1)
        
        grad_local_unified = grad_chunks[rank]
        
        return grad_local_unified, None

class SPBroadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, sp_group):
        ctx.sp_group = sp_group
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        dist.barrier(group=ctx.sp_group)
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.sp_group)
        return grad_output, None
