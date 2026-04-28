import os

from .runtime import RUNTIME, AsyncRunMode

def is_asyncrt_enabled():
    return os.environ.get("ASYNC_RUNTIME_ENABLE", "0") == "1"