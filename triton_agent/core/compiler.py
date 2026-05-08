"""Triton JIT compiler wrapper with error capture."""

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class CompileResult:
    success: bool
    kernel_fn: Optional[Callable] = None
    compile_time_s: float = 0.0
    error_log: str = ""


class TritonCompiler:
    """Wraps Triton JIT compilation with error handling."""

    def compile(
        self,
        kernel_template: Callable,
        compile_args: dict[str, Any],
    ) -> CompileResult:
        """Attempt to JIT-compile a Triton kernel template.

        Args:
            kernel_template: callable that returns a compiled kernel when invoked
            compile_args: kwargs forwarded to the template (B, T, D, BLOCK_SIZE, etc.)

        Returns:
            CompileResult with success flag, kernel_fn, and diagnostic info.
        """
        import time
        result = CompileResult()
        try:
            start = time.perf_counter()
            result.kernel_fn = kernel_template(**compile_args)
            result.compile_time_s = time.perf_counter() - start
            result.success = True
        except Exception:
            result.success = False
            result.error_log = traceback.format_exc()
        return result


def compile_kernel(
    kernel_template: Callable,
    compile_args: dict[str, Any],
) -> CompileResult:
    """Convenience wrapper for single-shot compilation."""
    compiler = TritonCompiler()
    return compiler.compile(kernel_template, compile_args)
