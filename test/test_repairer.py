import pytest

from triton_agent.core.spec import OpAction, CandidateResult
from triton_agent.agent.repairer import repair_action, should_retry


class TestRepairer:
    def test_compile_register_spill(self):
        action = OpAction(template_id="v1", block_d=512, num_warps=8, num_stages=4)
        result = CandidateResult(compile_pass=False, compile_log="out of registers spill detected")

        repaired = repair_action(action, result)
        assert repaired is not None
        assert repaired.block_d < 512
        assert repaired.num_stages <= 4

    def test_compile_shared_memory(self):
        action = OpAction(template_id="v1", block_d=1024, num_warps=4, num_stages=4)
        result = CandidateResult(compile_pass=False, compile_log="shared memory out of memory")

        repaired = repair_action(action, result)
        assert repaired is not None
        assert repaired.block_d < 1024
        assert repaired.num_stages <= 4

    def test_compile_not_supported(self):
        action = OpAction(template_id="v1", block_d=2048, num_warps=16, num_stages=3)
        result = CandidateResult(compile_pass=False, compile_log="not supported configuration")

        repaired = repair_action(action, result, max_block_size=1024)
        assert repaired is not None
        assert repaired.block_d <= 256
        assert repaired.num_warps <= 8

    def test_verify_nan(self):
        action = OpAction(template_id="v1", block_d=256, vectorize=True)
        result = CandidateResult(
            compile_pass=True, verify_pass=False,
            verify_log='{"max_abs_error": 0.01, "has_nan": true, "has_inf": false}',
        )

        repaired = repair_action(action, result)
        assert repaired is not None
        assert repaired.vectorize is False

    def test_verify_numerical_drift(self):
        action = OpAction(template_id="v1", block_d=512, vectorize=True)
        result = CandidateResult(
            compile_pass=True, verify_pass=False,
            verify_log='{"max_abs_error": 0.05, "has_nan": false, "has_inf": false}',
        )

        repaired = repair_action(action, result)
        assert repaired is not None
        assert repaired.vectorize is False
        assert repaired.block_d < 512

    def test_no_repair_on_pass(self):
        action = OpAction(template_id="v1", block_d=256)
        result = CandidateResult(compile_pass=True, verify_pass=True)
        assert repair_action(action, result) is None

    def test_should_retry(self):
        orig = OpAction(template_id="v1", block_d=512, num_warps=4)
        repaired = OpAction(template_id="v1", block_d=256, num_warps=4)
        result = CandidateResult(compile_pass=True, verify_pass=False)
        assert should_retry(repaired, orig, result) is True

    def test_should_not_retry_same(self):
        action = OpAction(template_id="v1", block_d=256)
        result = CandidateResult(compile_pass=False)
        assert should_retry(action, action, result) is False
