"""Shape/dtype/device profile and core data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpState:
    op_name: str
    B: int
    T: int
    D: int
    dtype: str
    device: str
    gpu_name: str = ""
    baseline_latency_us: float = 0.0
    historical_best_config: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "B": self.B,
            "T": self.T,
            "D": self.D,
            "dtype": self.dtype,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "baseline_latency_us": self.baseline_latency_us,
            "historical_best_config": self.historical_best_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpState":
        return cls(
            op_name=data["op_name"],
            B=data["B"],
            T=data["T"],
            D=data["D"],
            dtype=data["dtype"],
            device=data["device"],
            gpu_name=data.get("gpu_name", ""),
            baseline_latency_us=data.get("baseline_latency_us", 0.0),
            historical_best_config=data.get("historical_best_config"),
        )


@dataclass
class OpAction:
    template_id: str
    block_d: int = 128
    num_warps: int = 4
    num_stages: int = 3
    vectorize: bool = False
    fusion: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "vectorize": self.vectorize,
            "fusion": self.fusion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpAction":
        return cls(
            template_id=data["template_id"],
            block_d=data.get("block_d", 128),
            num_warps=data.get("num_warps", 4),
            num_stages=data.get("num_stages", 3),
            vectorize=data.get("vectorize", False),
            fusion=data.get("fusion", False),
        )


@dataclass
class CandidateResult:
    compile_pass: bool = False
    verify_pass: bool = False
    latency_us_p50: float = 0.0
    latency_us_p90: float = 0.0
    latency_us_p99: float = 0.0
    speedup: float = 1.0
    variance: float = 0.0
    memory_peak_mb: float = 0.0
    reward: float = 0.0
    promoted: bool = False
    compile_log: str = ""
    verify_log: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_pass": self.compile_pass,
            "verify_pass": self.verify_pass,
            "latency_us_p50": self.latency_us_p50,
            "latency_us_p90": self.latency_us_p90,
            "latency_us_p99": self.latency_us_p99,
            "speedup": self.speedup,
            "variance": self.variance,
            "memory_peak_mb": self.memory_peak_mb,
            "reward": self.reward,
            "promoted": self.promoted,
            "compile_log": self.compile_log,
            "verify_log": self.verify_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateResult":
        return cls(
            compile_pass=data.get("compile_pass", False),
            verify_pass=data.get("verify_pass", False),
            latency_us_p50=data.get("latency_us_p50", 0.0),
            latency_us_p90=data.get("latency_us_p90", 0.0),
            latency_us_p99=data.get("latency_us_p99", 0.0),
            speedup=data.get("speedup", 1.0),
            variance=data.get("variance", 0.0),
            memory_peak_mb=data.get("memory_peak_mb", 0.0),
            reward=data.get("reward", 0.0),
            promoted=data.get("promoted", False),
            compile_log=data.get("compile_log", ""),
            verify_log=data.get("verify_log", ""),
        )
