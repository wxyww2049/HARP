import torch
import torch.distributed as dist
import threading
import queue
from time import time
import os

from managers.mem_manager import HBMToDDROffloader
import accumulate_sp

def _set_thread_affinity(core_id):
    if core_id is not None:
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, {core_id})


class AsyncGradSyncManager:
    def __init__(self, model, local_rank, local_die, param_rank, my_params_set):
        self.local_die = local_die
        self.local_rank = local_rank
        self.run_id_base = os.environ.get("TORCH_SHM_ID", "")
        self.param_rank = param_rank
        self.my_params_set = my_params_set

        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_exc = None

        self._comm_pending = {}
        self._next_comm_seq = 0
        self._num_comm_tasks = 0
        self._worker = threading.Thread(
            target=self._grad_worker_loop,
            name=f"grad-sync-worker-rank{local_rank}",
            daemon=True,
        )
        self._worker.start()

        self._register_hooks(model)

    def _register_hooks(self, model):
        named_params = list(model.named_parameters())[::-1]

        comm_seq = 0

        for name, param in named_params:
            if not param.requires_grad:
                continue

            is_mine = self.param_rank[id(param)] == self.local_rank
            is_my_grad = param in self.my_params_set

            if (
                "layers." in name
                or "context_refiner." in name
                or "noise_refiner." in name
                or "cond_refiner_" in name
            ):
                expected_count = 4
            else:
                expected_count = 1

            my_comm_seq = None
            if is_mine:
                my_comm_seq = comm_seq
                comm_seq += 1

            hook = self._make_hook(
                is_mine=is_mine,
                name=name,
                is_my_grad=is_my_grad,
                expected_count=expected_count,
                comm_seq=my_comm_seq,
            )
            param.register_post_accumulate_grad_hook(hook)

        self._num_comm_tasks = comm_seq

    def _make_hook(self, is_mine, name, is_my_grad, expected_count, comm_seq):
        def hook(p):
            task = {
                "param": p,
                "grad": p.grad,
                "shared_grad": p.shared_grad,
                "tag": f"{self.run_id_base}_{self.local_die}_{name}",
                "expected_count": expected_count,
                "is_mine": is_mine,
                "comm_seq": comm_seq,
            }
            self._task_queue.put(task)

        return hook

    def _flush_ready_comm_tasks_in_order(self):
        while self._next_comm_seq in self._comm_pending:
            payload = self._comm_pending.pop(self._next_comm_seq)

            if payload is not None:
                shared_grad, chunk_grad, tag = payload
                accumulate_sp.push_comm_task(shared_grad, chunk_grad, tag)

            self._next_comm_seq += 1

    def _grad_worker_loop(self):
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        _set_thread_affinity(rank % 16 * 38 + 35)

        try:
            while not self._stop_event.is_set():
                try:
                    task = self._task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if task is None:
                    self._task_queue.task_done()
                    break

                p = task["param"]
                grad = task["grad"]
                shared_grad = task["shared_grad"]
                tag = task["tag"]
                expected_count = task["expected_count"]
                is_mine = task["is_mine"]
                comm_seq = task["comm_seq"]

                if hasattr(grad, "ready_event"):
                    grad.ready_event.wait()

                if is_mine or expected_count == 4:
                    accumulate_sp.push_task(
                        grad,
                        shared_grad,
                        tag,
                        expected_count,
                    )

                if is_mine:
                    if hasattr(p, "chunk_part") and hasattr(p.chunk_part, "grad"):
                        self._comm_pending[comm_seq] = (
                            shared_grad,
                            p.chunk_part.grad,
                            tag,
                        )
                    else:
                        self._comm_pending[comm_seq] = None

                    self._flush_ready_comm_tasks_in_order()

                p.grad = None
                self._task_queue.task_done()

        except Exception as e:
            self._worker_exc = e

    def wait_for_completion(self):
        tq_time = time()
        self._task_queue.join()
        tq_time = time() - tq_time
        if self._worker_exc is not None:
            raise RuntimeError("Background grad worker failed") from self._worker_exc

        # Optional conditioning branches may have no gradient this step.
        # All hooks have drained: submit the remaining active parameters in order.
        while self._comm_pending:
            self._next_comm_seq = min(self._comm_pending)
            self._flush_ready_comm_tasks_in_order()
        self._next_comm_seq = 0

        sp_time = time()
        accumulate_sp.wait_all()
        sp_time = time() - sp_time
        dp_time = time()
        accumulate_sp.wait_comm_all()
        dp_time = time() - dp_time
        barr_time = time()
        dist.barrier()
        barr_time = time() - barr_time
        return tq_time, sp_time, dp_time, barr_time

    def shutdown(self):
        self._stop_event.set()
        self._task_queue.put(None)
        self._worker.join()

        if self._worker_exc is not None:
            raise RuntimeError("Background grad worker failed") from self._worker_exc

        accumulate_sp.shutdown()