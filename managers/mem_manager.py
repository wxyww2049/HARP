from contextlib import contextmanager
import my_allocator
import os
import queue
import threading
import torch

@contextmanager
def use_hbm_allocator(state:bool = True):
    if my_allocator.get_allocate_state() == state:
        yield
        return
    try:
        my_allocator.set_allocate_on_hbm(state)
        yield
    finally:
        my_allocator.set_allocate_on_hbm(not state)


class CopyWork:
    def __init__(self):
        self._done_event = threading.Event()
        self._exception = None
        self.tensor = None

    def reset(self):
        self._done_event.clear()
        self._exception = None
        self.tensor = None

    def mark_done(self, tensor=None):
        self.tensor = tensor
        self._done_event.set()

    def mark_failed(self, exc):
        self._exception = exc
        self._done_event.set()

    def wait(self):
        self._done_event.wait()
        if self._exception is not None:
            raise self._exception
        return self.tensor
    def __del__(self):
        if self.tensor is not None:
            self.tensor = None

class CopyEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.num_workers = 1
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        self.total_size = 0
        self.max_pending_bytes = int(
            os.environ.get("COPY_ENGINE_MAX_PENDING_BYTES", 1 * 200 * 1024 * 1024)
        )
        self._size_cv = threading.Condition()
        self.work_pool = queue.Queue()
        for _ in range(5000): 
            self.work_pool.put(CopyWork())

        self.worker_threads = []
        for worker_idx in range(self.num_workers):
            worker_thread = threading.Thread(
                target=self._run,
                args=(worker_idx,),
                daemon=True,
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_work_handle(self):
        try:
            work = self.work_pool.get_nowait()
            work.reset()
            return work
        except queue.Empty:
            return CopyWork()

    def recycle_work_handle(self, work):
        work.reset()
        self.work_pool.put(work)

    def submit(self, src_storage, src_meta, cache_key, storage_cache, param=None, dp_queue = None, dp_group = None):

        task = self._get_work_handle()
        task_size = src_storage.size() * src_storage.element_size()
        with self._size_cv:
            while self.total_size + task_size > self.max_pending_bytes:
                print("offload queue is full, waiting")
                self._size_cv.wait()
            self.total_size += task_size
        self.task_queue.put((src_storage, src_meta, cache_key, storage_cache, task, param, dp_queue,dp_group))
        
        return task

    def _get_core_id(self, worker_idx):
        core_ids = os.environ.get("COPY_ENGINE_CORE_IDS")
        if core_ids:
            parsed_core_ids = [
                int(core_id.strip()) for core_id in core_ids.split(",") if core_id.strip()
            ]
            if worker_idx < len(parsed_core_ids):
                return parsed_core_ids[worker_idx]
        return (self.rank % 16) * 38 + 32 + 3 + worker_idx

    def _run(self, worker_idx):
        core_id = self._get_core_id(worker_idx)
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, {core_id})

        while not self.stop_event.is_set():
            try:
                src_storage, src_meta, cache_key, storage_cache, task, param, dp_queue, dp_group = self.task_queue.get(timeout=0.05)
                
                try:
                    if cache_key not in storage_cache:
                        ddr_storage = torch.UntypedStorage(src_storage.size())
                        my_allocator.copy_from_hbm(src_storage, ddr_storage)
                        storage_cache[cache_key] = ddr_storage

                    ddr_storage = storage_cache[cache_key]
                    
                    dst_tensor = torch.empty(0, dtype=src_meta['dtype'])
                    dst_tensor.set_(
                        source=ddr_storage,
                        storage_offset=src_meta['offset'],
                        size=src_meta['size'],
                        stride=src_meta['stride']
                    )
                    if param is not None and param.grad is not None:
                        param.grad = dst_tensor
                        if dp_queue is not None and dp_group is not None:
                            dp_queue.put((param, dp_group))
                        task.mark_done()
                        self.recycle_work_handle(task)
                    else:
                        task.mark_done(dst_tensor)
                except Exception as exc:
                    task.mark_failed(exc)
                finally:
                    self.task_queue.task_done()
                    with self._size_cv:
                        self.total_size -= src_storage.size() * src_storage.element_size()
                        del src_storage, src_meta, cache_key, param, ddr_storage, dst_tensor
                        self._size_cv.notify_all()
            except queue.Empty:
                continue

    def wait_all(self):
        self.task_queue.join()

    def shutdown(self):
        self.stop_event.set()
        for worker_thread in self.worker_threads:
            worker_thread.join()


class HBMToDDROffloader:
    def __init__(self):
        self.storage_cache = {}
        self.copyEngine = CopyEngine.get_instance()

    def wait_all_copy_tasks(self):
        self.copyEngine.wait_all()
        self.storage_cache.clear()

    def  _offload_param(self, param, dp_queue, dp_group):
        if param.grad is None:
            return None
        
        storage = param.grad.untyped_storage()
        storage_ptr = storage.data_ptr()
        
        unique_seq_id = my_allocator.get_alloc_id_on_hbm(storage_ptr) 
        
        if unique_seq_id == -1:
            return None

        cache_key = (storage_ptr, unique_seq_id)
        
        src_meta = {
            'dtype': param.grad.dtype,
            'offset': param.grad.storage_offset(),
            'size': param.grad.size(),
            'stride': param.grad.stride()
        }
        
        return self.copyEngine.submit(
            storage, src_meta, cache_key, self.storage_cache, param, dp_queue,dp_group
        )

    def _pack_hook_offload(self, tensor):
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        
        unique_seq_id = my_allocator.get_alloc_id_on_hbm(storage_ptr) 
        
        if unique_seq_id == -1:
            return tensor.detach()
        cache_key = (storage_ptr, unique_seq_id)
        
        src_meta = {
            'dtype': tensor.dtype,
            'offset': tensor.storage_offset(),
            'size': tensor.size(),
            'stride': tensor.stride()
        }
        
        return self.copyEngine.submit(
            storage, src_meta, cache_key, self.storage_cache
        )

    def pack_hook(self, tensor):
        return self._pack_hook_offload(tensor)
    def recycle(self,handle):
        self.copyEngine.recycle_work_handle(handle)

    def unpack_hook(self, tensor_or_handle):
        if isinstance(tensor_or_handle, torch.Tensor):
            return tensor_or_handle
        tensor = tensor_or_handle.wait()
        
        self.recycle(tensor_or_handle)
        return tensor
