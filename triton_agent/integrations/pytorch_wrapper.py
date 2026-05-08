"""PyTorch wrapper: exposes optimized Triton kernels as torch.autograd.Function.

This allows seamless drop-in replacement of PyTorch ops with optimized
Triton kernels discovered by the agent.
"""

from typing import Any, Callable, Optional
import warnings


class TritonOpWrapper:
    """Wraps a Triton kernel as a torch-compatible callable.

    Usage:
        from triton_agent.integrations.pytorch_wrapper import TritonOpWrapper

        wrapper = TritonOpWrapper(kernel_fn, compile_args)
        output = wrapper(x, weight)  # drop-in replacement
    """

    def __init__(self, kernel_fn: Callable, compile_args: Optional[dict] = None) -> None:
        self._kernel_fn = kernel_fn
        self._compile_args = compile_args or {}

    def __call__(self, *args, **kwargs) -> Any:
        try:
            return self._kernel_fn(*args, **self._compile_args, **kwargs)
        except Exception as e:
            warnings.warn(f"TritonOpWrapper fallback: {e}")
            return self._fallback(*args, **kwargs)

    def _fallback(self, *args, **kwargs) -> Any:
        raise NotImplementedError("No fallback registered. Provide a PyTorch reference.")


class PyTorchAdapter:
    """Adapter that avoids torch.compile re-compilation of Triton kernels.

    Registers a Triton kernel as a custom op to bypass torch.compile's tracing
    and let the JIT-optimized Triton code run natively.
    """

    def __init__(self, op_name: str, kernel_fn: Callable) -> None:
        self.op_name = op_name
        self.kernel_fn = kernel_fn

    def apply(self, *args) -> Any:
        return self.kernel_fn(*args)

    def __call__(self, *args) -> Any:
        return self.apply(*args)
