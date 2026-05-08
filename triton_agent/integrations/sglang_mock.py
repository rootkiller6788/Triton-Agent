"""SGLang mock adapter: integrates Triton-agent kernels into SGLang-style interface.

Phase 2: mock only — no real SGLang dependency.
"""

from typing import Any, Callable


class SGLangMockAdapter:
    """Mock adapter for SGLang-style kernel dispatch."""

    def __init__(self, op_name: str, kernel_fn: Callable) -> None:
        self.op_name = op_name
        self.kernel_fn = kernel_fn

    def dispatch(self, *args, **kwargs) -> Any:
        return self.kernel_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.dispatch(*args, **kwargs)
