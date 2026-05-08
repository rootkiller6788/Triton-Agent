import pytest
import yaml
import tempfile
import os

from triton_agent.core.contract import OpContract, ContractError, SearchSpace, ToleranceSpec, BenchmarkSpec, PromotionSpec


VALID_CONTRACT_YAML = """
op: rmsnorm

inputs:
  x: [B, T, D]
  weight: [D]

outputs:
  y: [B, T, D]

dtype:
  - fp16
  - bf16

device:
  - cuda

tolerance:
  max_abs_error: 1e-3
  mean_abs_error: 1e-4

search_space:
  BLOCK_D: [64, 128, 256, 512]
  num_warps: [1, 2, 4, 8]
  num_stages: [3, 4]
  vectorize: [true, false]

benchmark:
  warmup: 20
  repeat: 100
  metric: latency_us_p50

promotion:
  min_speedup: 1.05
  max_variance: 0.10
  regression_required: true
"""


class TestOpContract:
    def test_from_dict_valid(self):
        data = yaml.safe_load(VALID_CONTRACT_YAML)
        contract = OpContract.from_dict(data)
        assert contract.op == "rmsnorm"
        assert len(contract.inputs) == 2
        assert contract.inputs[0].name == "x"
        assert contract.inputs[0].shape.dims == ["B", "T", "D"]
        assert contract.inputs[1].name == "weight"
        assert contract.inputs[1].shape.dims == ["D"]
        assert len(contract.outputs) == 1
        assert contract.outputs[0].name == "y"
        assert contract.outputs[0].shape.dims == ["B", "T", "D"]
        assert contract.dtype == ["fp16", "bf16"]
        assert contract.device == ["cuda"]
        assert contract.tolerance.max_abs_error == 1e-3
        assert contract.tolerance.mean_abs_error == 1e-4

    def test_from_dict_missing_op(self):
        with pytest.raises(ContractError, match="Missing required field"):
            OpContract.from_dict({})

    def test_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(VALID_CONTRACT_YAML)
            tmp_path = f.name
        try:
            contract = OpContract.from_yaml(tmp_path)
            assert contract.op == "rmsnorm"
            assert contract.tolerance.max_abs_error == 1e-3
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_missing_file(self):
        with pytest.raises(ContractError, match="not found"):
            OpContract.from_yaml("/nonexistent/contract.yaml")

    def test_to_dict_roundtrip(self):
        data = yaml.safe_load(VALID_CONTRACT_YAML)
        contract = OpContract.from_dict(data)
        result = contract.to_dict()
        assert result["op"] == "rmsnorm"
        assert result["tolerance"]["max_abs_error"] == 1e-3

    def test_search_space_defaults(self):
        data = yaml.safe_load(VALID_CONTRACT_YAML)
        contract = OpContract.from_dict(data)
        assert contract.search_space.BLOCK_D == [64, 128, 256, 512]
        assert contract.search_space.num_warps == [1, 2, 4, 8]

    def test_benchmark_spec(self):
        data = yaml.safe_load(VALID_CONTRACT_YAML)
        contract = OpContract.from_dict(data)
        assert contract.benchmark.warmup == 20
        assert contract.benchmark.repeat == 100
        assert contract.benchmark.metric == "latency_us_p50"

    def test_promotion_spec(self):
        data = yaml.safe_load(VALID_CONTRACT_YAML)
        contract = OpContract.from_dict(data)
        assert contract.promotion.min_speedup == 1.05
        assert contract.promotion.max_variance == 0.10
        assert contract.promotion.regression_required is True

    def test_minimal_contract_defaults(self):
        minimal = {
            "op": "test_op",
            "inputs": {"x": ["N"]},
            "outputs": {"y": ["N"]},
            "tolerance": {"max_abs_error": 0.01, "mean_abs_error": 0.001},
        }
        contract = OpContract.from_dict(minimal)
        assert contract.dtype == []
        assert contract.device == []
        assert contract.search_space.BLOCK_D == [64, 128, 256, 512]
        assert contract.benchmark.warmup == 20
        assert contract.promotion.min_speedup == 1.05
