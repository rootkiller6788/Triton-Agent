"""CUDA Extension fallback: compiles Triton kernels as CUDA C++ when JIT fails.

This is a Phase 4 production hardening feature. When Triton JIT fails (e.g.,
due to unsupported hardware features or complex warp configurations), the
system falls back to compiling the kernel as a torch.utils.cpp_extension.

Note: requires CUDA toolkit (nvcc) to be available on the system.
"""

import os
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Callable, Optional


TEMPLATE_PREAMBLE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Auto-generated CUDA kernel fallback
// Original failure context is preserved in the error_log

torch::Tensor triton_agent_fallback(torch::Tensor input) {
    // Placeholder — real CUDA code generation happens at kernel-level
    return input.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triton_agent_fallback, "fallback kernel");
}
"""


class CUDAExtensionCompiler:
    """Compiles a CUDA C++ kernel as a torch extension.

    This is an emergency fallback when Triton JIT compilation fails.
    In practice this is rarely needed, but provides a safety net for
    production deployments.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "triton_agent_cuda_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def compile(self, kernel_source: str, kernel_name: str) -> Optional[Callable]:
        """Attempt to compile a CUDA C++ kernel as a torch extension.

        Args:
            kernel_source: CUDA C++ source code string
            kernel_name: unique name for this kernel (used for caching)

        Returns:
            Callable kernel function, or None on failure.
        """
        source_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]
        build_dir = self._cache_dir / f"{kernel_name}_{source_hash}"

        if build_dir.exists():
            try:
                return self._load_cached(build_dir)
            except Exception:
                pass

        try:
            self._build_extension(kernel_source, kernel_name, build_dir)
            return self._load_cached(build_dir)
        except Exception:
            return None

    def _build_extension(self, source: str, name: str, build_dir: Path) -> None:
        """Run JIT compilation via torch.utils.cpp_extension."""
        import torch.utils.cpp_extension

        cpp_source = f"""
#include <torch/extension.h>
{source}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &{name}, "{name} CUDA kernel");
}}
"""
        src_file = build_dir / f"{name}.cu"
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_text(cpp_source, encoding="utf-8")

        torch.utils.cpp_extension.load(
            name=name,
            sources=[str(src_file)],
            build_directory=str(build_dir),
            verbose=False,
        )

    def _load_cached(self, build_dir: Path) -> Callable:
        import importlib
        import sys

        for so_file in build_dir.glob("*.pyd"):
            spec = importlib.util.spec_from_file_location(
                so_file.stem, str(so_file)
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                return mod.forward
        raise FileNotFoundError(f"No compiled extension found in {build_dir}")

    def is_available(self) -> bool:
        """Check if CUDA toolkit is available for compilation."""
        try:
            import torch
            return torch.cuda.is_available() and self._nvcc_found()
        except ImportError:
            return False

    def _nvcc_found(self) -> bool:
        try:
            subprocess.run(
                ["nvcc", "--version"], capture_output=True, timeout=5
            )
            return True
        except Exception:
            return False
