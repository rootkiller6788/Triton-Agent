"""vLLM mock adapter: integrates Triton-agent best kernels into vLLM-style interface.

Phase 2: mock only — no real vLLM dependency. Registers the kernel with a
vLLM-compatible signature so it can be swapped in later.
"""

from typing import Any, Callable, Optional


class VLLMMockAdapter:
    """Mock adapter that exposes a vLLM-compatible custom op API.

    In Phase 3+, this will be replaced with a real vLLM custom op registration.
    """

    def __init__(self, op_name: str, kernel_fn: Callable) -> None:
        self.op_name = op_name
        self.kernel_fn = kernel_fn

    def dispatch(self, *args, **kwargs) -> Any:
        return self.kernel_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.dispatch(*args, **kwargs)
