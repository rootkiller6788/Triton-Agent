import pytest
from triton_agent.core.spec import OpState, OpAction, CandidateResult


class TestOpState:
    def test_create(self):
        state = OpState(
            op_name="rmsnorm",
            B=8, T=2048, D=4096,
            dtype="fp16",
            device="cuda",
            gpu_name="A100",
        )
        assert state.op_name == "rmsnorm"
        assert state.B == 8
        assert state.D == 4096

    def test_to_from_dict_roundtrip(self):
        state = OpState(
            op_name="rope",
            B=4, T=1024, D=2048,
            dtype="bf16",
            device="cuda",
            historical_best_config={"block_d": 256, "num_warps": 4},
        )
        data = state.to_dict()
        restored = OpState.from_dict(data)
        assert restored.op_name == "rope"
        assert restored.historical_best_config == {"block_d": 256, "num_warps": 4}

    def test_defaults(self):
        state = OpState.from_dict({
            "op_name": "test", "B": 1, "T": 1, "D": 1,
            "dtype": "fp32", "device": "cpu",
        })
        assert state.gpu_name == ""
        assert state.baseline_latency_us == 0.0
        assert state.historical_best_config is None


class TestOpAction:
    def test_create(self):
        action = OpAction(
            template_id="triton_v2",
            block_d=256,
            num_warps=4,
            num_stages=3,
            vectorize=True,
        )
        assert action.template_id == "triton_v2"
        assert action.block_d == 256
        assert action.vectorize is True
        assert action.fusion is False

    def test_to_from_dict_roundtrip(self):
        action = OpAction(
            template_id="triton_v1",
            block_d=128,
            num_warps=4,
            num_stages=4,
            vectorize=False,
            fusion=True,
        )
        data = action.to_dict()
        restored = OpAction.from_dict(data)
        assert restored.template_id == "triton_v1"
        assert restored.fusion is True

    def test_defaults(self):
        action = OpAction.from_dict({"template_id": "triton_v3"})
        assert action.block_d == 128
        assert action.num_warps == 4
        assert action.num_stages == 3
        assert action.vectorize is False
        assert action.fusion is False


class TestCandidateResult:
    def test_create_defaults(self):
        result = CandidateResult()
        assert result.compile_pass is False
        assert result.verify_pass is False
        assert result.speedup == 1.0
        assert result.promoted is False

    def test_to_from_dict_roundtrip(self):
        result = CandidateResult(
            compile_pass=True,
            verify_pass=True,
            latency_us_p50=23.4,
            latency_us_p90=25.1,
            latency_us_p99=28.0,
            speedup=1.18,
            variance=0.05,
            memory_peak_mb=512.0,
            reward=0.88,
            promoted=True,
        )
        data = result.to_dict()
        restored = CandidateResult.from_dict(data)
        assert restored.compile_pass is True
        assert restored.latency_us_p50 == 23.4
        assert restored.speedup == 1.18
        assert restored.promoted is True
