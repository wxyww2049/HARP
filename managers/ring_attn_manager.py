
import threading
import queue
import os

class CommWork:
    def __init__(self):
        self._done_event = threading.Event()
        self.work_handle = None

    def mark_done(self):
        self._done_event.set()

    def wait(self):
        self._done_event.wait()

class CPUWaitEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _run(self):
        core_id = self.rank % 16 * 38 + 34
        if core_id is not None:
            tid = threading.get_native_id()
            os.sched_setaffinity(tid, {core_id})

        while not self.stop_event.is_set():
            try:
                req, task = self.task_queue.get(timeout=0.1)
                try:
                    if req is not None:
                        req.wait()
                    task.mark_done()
                except Exception as e:
                    print(f"Dist wait error: {e}")
                
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def submit(self, req):
        task = CommWork()
        self.task_queue.put((req, task))
        return task

    def shutdown(self):
        self.stop_event.set()
        self.worker_thread.join()