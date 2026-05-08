import pytest
from pathlib import Path
from triton_agent.core.registry import OpRegistry, RegistryError, get_registry
from triton_agent.core.contract import OpContract


class TestOpRegistry:
    def test_register_and_get(self):
        reg = OpRegistry()
        contract = OpContract.from_dict({
            "op": "rmsnorm",
            "inputs": {"x": ["B", "T", "D"]},
            "outputs": {"y": ["B", "T", "D"]},
            "tolerance": {"max_abs_error": 1e-3, "mean_abs_error": 1e-4},
        })
        reg.register(contract)
        assert reg.has("rmsnorm")
        assert reg.get("rmsnorm").op == "rmsnorm"

    def test_get_missing(self):
        reg = OpRegistry()
        with pytest.raises(RegistryError, match="not registered"):
            reg.get("nonexistent")

    def test_list_empty(self):
        reg = OpRegistry()
        assert reg.list() == []

    def test_list_sorted(self):
        reg = OpRegistry()
        for name in ["rope", "rmsnorm", "fused_bias_gelu"]:
            contract = OpContract.from_dict({
                "op": name,
                "inputs": {"x": ["N"]},
                "outputs": {"y": ["N"]},
                "tolerance": {"max_abs_error": 0.01, "mean_abs_error": 0.001},
            })
            reg.register(contract)
        assert reg.list() == ["fused_bias_gelu", "rmsnorm", "rope"]

    def test_has_false(self):
        reg = OpRegistry()
        assert not reg.has("anything")

    def test_get_registry_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
