import torch
from multiprocessing import shared_memory
import torch.distributed as dist
import queue
import threading
import os
import time
def is_norm_param(name):
    return "norm" in name.lower()
def is_no_shard_param(name):
    return False

def get_shm(shm_name, size,local_rank, master_rank):
    if local_rank == master_rank:
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            shm.unlink()
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
    dist.barrier()
    if local_rank != master_rank:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
    return shm


def setup_shared_model(model, param_rank, rank, local_rank, master_rank = 3, grad_master_rank = 2):
    shm_name = f"zimage_model_shm_{rank // 4}"
    shm_grad_name = f"zimage_grad_shm_{rank // 4}"
    shm_state_name = f"zimage_state_shm_{rank // 4}"
    offsets = []
    num_parameters = len(list(model.parameters()))
    current_offset = 0
    for p in model.parameters():
        offsets.append(current_offset)
        current_offset += p.numel() * p.element_size()
    total_bytes = current_offset

    shm = get_shm(shm_name, total_bytes, local_rank, master_rank)
    shm_grad = get_shm(shm_grad_name, total_bytes, local_rank, grad_master_rank)
    shm_state = get_shm(shm_state_name, num_parameters, local_rank, master_rank)
    shm_state.buf[:num_parameters] = bytearray(num_parameters)
    shm_state_buffer = torch.frombuffer(shm_state.buf, dtype=torch.uint8)

    for i, (name,p) in enumerate(model.named_parameters()):

        shared_grad_view = torch.frombuffer(
            shm_grad.buf, 
            dtype=p.dtype, 
            count=p.numel(), 
            offset=offsets[i]
        ).view(p.shape)
        p.shared_grad = shared_grad_view.detach()
        
        if param_rank[id(p)] == local_rank:
            p.shared_grad.zero_()

        if is_norm_param(name):
            p.state = torch.ones((),dtype=torch.uint8)
            continue

        p.state = shm_state_buffer[i]
        p.state.fill_(1)
        shared_tensor_view = torch.frombuffer(
            shm.buf, 
            dtype=p.dtype, 
            count=p.numel(), 
            offset=offsets[i]
        ).view(p.shape)
        
        if param_rank[id(p)] == local_rank:
            shared_tensor_view.copy_(p.data)
        p.data = shared_tensor_view
    dist.barrier()
    
    return shm,shm_grad,shm_state


def build_balanced_param_groups(model, sp_size, local_rank,block_size=100*1024*1024):
    norm_params = []
    non_norm_trainable = []
    total_bytes = 0

    for name, p in model.named_parameters():
        if is_norm_param(name):
            norm_params.append((name, p))
            total_bytes += p.numel() * p.element_size()
            continue

        if p.requires_grad:
            non_norm_trainable.append((name, p))

    param_rank = {}
    for i,(name,p) in enumerate(norm_params):
        param_rank[id(p)] = i % sp_size

    rank_bytes = [0] * sp_size
    current_rank = 0

    for name, p in non_norm_trainable:
        param_bytes = p.numel() * p.element_size()

        if (
            rank_bytes[current_rank] > 0
            and rank_bytes[current_rank] > block_size
        ):
            current_rank = (current_rank + 1) % sp_size
            rank_bytes[current_rank] = 0

        param_rank[id(p)] = current_rank
        rank_bytes[current_rank] += param_bytes
        if current_rank == local_rank:
            total_bytes += param_bytes

    my_params = []
    opt_params = []
        
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_norm_param(name):
            my_params.append(p)
        elif param_rank[id(p)] == local_rank:
            my_params.append(p)
    return my_params, param_rank

def build_chunk_param(my_params, model, rs_size, rs_rank):
    opt_params=[]
    def set_chunk_param(name,p):
        if p.ndim == 0 or p.shape[0] % rs_size != 0 or is_no_shard_param(name):
            return p
        chunk = p.shape[0] // rs_size
        begin = rs_rank * chunk
        end = (rs_rank + 1) * chunk
        src_part = p.detach().narrow(0, begin, chunk)
        p_part = torch.empty_like(src_part)
        p_part.copy_(src_part)
        p_part.chunk_begin = begin
        p_part.chunk_end = end
        p_part.requires_grad = True
        p_part.grad = p.shared_grad[p_part.chunk_begin : p_part.chunk_end]
        p.chunk_part = p_part
        return p_part
    for name,p in model.named_parameters():
        if p in my_params:
            opt_params.append(set_chunk_param(name,p))
        
    return opt_params

class AsyncParamAllGatherManager:
    def __init__(self, rs_group, rs_size, core_id=None):
        self.rs_group = rs_group
        self.rs_size = rs_size
        self.core_id = core_id
        self.task_queue = queue.Queue()
        self.exc = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def push(self, p):
        if self.exc is not None:
            raise self.exc
        self.task_queue.put(p)

    def wait_all(self):
        self.task_queue.join()
        if self.exc is not None:
            raise self.exc

    def shutdown(self):
        self.task_queue.put(None)
        self.worker.join()
        if self.exc is not None:
            raise self.exc

    def _run(self):
        if self.core_id is not None:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, {self.core_id})

        while True:
            p = self.task_queue.get()
            try:
                if p is None:
                    return
                self._all_gather_param(p)
            except Exception as e:
                raise e
            finally:
                self.task_queue.task_done()

    @torch.no_grad()
    def _all_gather_param(self, p):
        if p.ndim == 0 or p.shape[0] % self.rs_size != 0:
            raise ValueError(f"param shape {tuple(p.shape)} cannot be evenly split by rs_size={self.rs_size}")

        expected_rows = p.shape[0] // self.rs_size
        if p.chunk_part.shape[0] != expected_rows:
            raise ValueError(
                f"chunk_part.shape[0]={p.chunk_part.shape[0]}, expected {expected_rows}, param shape={tuple(p.shape)}"
            )

        full_param = p.detach()
        local_chunk = p.chunk_part.detach()

        dist.barrier(group=self.rs_group)
        dist.all_gather_into_tensor(full_param,local_chunk,group=self.rs_group)
        p.state.fill_(1)



def set_opt_hook(model):
    for name, module in model.named_modules():
        def check_param_available_hook(module, inputs, module_name=name):
            for p_name, p in module.named_parameters(recurse=False):
                if hasattr(p, "state") and hasattr(p, "chunk_part"):
                    while p.state.item() != 1:
                        time.sleep(0.00001)

        if len(list(module.parameters(recurse=False))) > 0:
            module.register_forward_pre_hook(check_param_available_hook)
