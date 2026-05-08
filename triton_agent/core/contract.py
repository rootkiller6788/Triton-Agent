"""Operator contract parser and data structures."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ContractError(Exception):
    """Raised when a contract is invalid or missing required fields."""


@dataclass
class ShapeSpec:
    dims: list[str]

    @classmethod
    def from_raw(cls, raw: list[str]) -> "ShapeSpec":
        return cls(dims=raw)


@dataclass
class InputSpec:
    name: str
    shape: ShapeSpec

    @classmethod
    def from_raw(cls, name: str, raw: list[str]) -> "InputSpec":
        return cls(name=name, shape=ShapeSpec.from_raw(raw))


@dataclass
class OutputSpec:
    name: str
    shape: ShapeSpec

    @classmethod
    def from_raw(cls, name: str, raw: list[str]) -> "OutputSpec":
        return cls(name=name, shape=ShapeSpec.from_raw(raw))


@dataclass
class ToleranceSpec:
    max_abs_error: float
    mean_abs_error: float

    @classmethod
    def from_raw(cls, raw: dict) -> "ToleranceSpec":
        return cls(
            max_abs_error=float(raw["max_abs_error"]),
            mean_abs_error=float(raw["mean_abs_error"]),
        )


@dataclass
class SearchSpace:
    BLOCK_D: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_warps: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_stages: list[int] = field(default_factory=lambda: [3, 4])
    vectorize: list[bool] = field(default_factory=lambda: [True, False])

    @classmethod
    def from_raw(cls, raw: dict) -> "SearchSpace":
        return cls(
            BLOCK_D=raw.get("BLOCK_D", [64, 128, 256, 512]),
            num_warps=raw.get("num_warps", [1, 2, 4, 8]),
            num_stages=raw.get("num_stages", [3, 4]),
            vectorize=raw.get("vectorize", [True, False]),
        )


@dataclass
class BenchmarkSpec:
    warmup: int = 20
    repeat: int = 100
    metric: str = "latency_us_p50"

    @classmethod
    def from_raw(cls, raw: dict) -> "BenchmarkSpec":
        return cls(
            warmup=int(raw.get("warmup", 20)),
            repeat=int(raw.get("repeat", 100)),
            metric=raw.get("metric", "latency_us_p50"),
        )


@dataclass
class PromotionSpec:
    min_speedup: float = 1.05
    max_variance: float = 0.10
    regression_required: bool = True

    @classmethod
    def from_raw(cls, raw: dict) -> "PromotionSpec":
        return cls(
            min_speedup=float(raw.get("min_speedup", 1.05)),
            max_variance=float(raw.get("max_variance", 0.10)),
            regression_required=bool(raw.get("regression_required", True)),
        )


@dataclass
class OpContract:
    op: str
    inputs: list[InputSpec]
    outputs: list[OutputSpec]
    dtype: list[str]
    device: list[str]
    tolerance: ToleranceSpec
    search_space: SearchSpace
    benchmark: BenchmarkSpec
    promotion: PromotionSpec

    @classmethod
    def from_dict(cls, data: dict) -> "OpContract":
        try:
            inputs = []
            for name, shape in data["inputs"].items():
                inputs.append(InputSpec.from_raw(name, shape))
            outputs = []
            for name, shape in data["outputs"].items():
                outputs.append(OutputSpec.from_raw(name, shape))
            return cls(
                op=data["op"],
                inputs=inputs,
                outputs=outputs,
                dtype=data.get("dtype", []),
                device=data.get("device", []),
                tolerance=ToleranceSpec.from_raw(data["tolerance"]),
                search_space=SearchSpace.from_raw(data.get("search_space", {})),
                benchmark=BenchmarkSpec.from_raw(data.get("benchmark", {})),
                promotion=PromotionSpec.from_raw(data.get("promotion", {})),
            )
        except KeyError as e:
            raise ContractError(f"Missing required field: {e}") from e

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpContract":
        path = Path(path)
        if not path.exists():
            raise ContractError(f"Contract file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ContractError(f"Invalid contract YAML: expected dict, got {type(data)}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "inputs": {inp.name: inp.shape.dims for inp in self.inputs},
            "outputs": {out.name: out.shape.dims for out in self.outputs},
            "dtype": self.dtype,
            "device": self.device,
            "tolerance": {
                "max_abs_error": self.tolerance.max_abs_error,
                "mean_abs_error": self.tolerance.mean_abs_error,
            },
            "search_space": {
                "BLOCK_D": self.search_space.BLOCK_D,
                "num_warps": self.search_space.num_warps,
                "num_stages": self.search_space.num_stages,
                "vectorize": self.search_space.vectorize,
            },
            "benchmark": {
                "warmup": self.benchmark.warmup,
                "repeat": self.benchmark.repeat,
                "metric": self.benchmark.metric,
            },
            "promotion": {
                "min_speedup": self.promotion.min_speedup,
                "max_variance": self.promotion.max_variance,
                "regression_required": self.promotion.regression_required,
            },
        }
