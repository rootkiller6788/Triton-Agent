"""FlashAttention-like Triton kernel v1: tiled attention with online softmax."""

import triton
import triton.language as tl


@triton.jit
def _flash_attn_v1_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    q_base = (pid_b * H + pid_h) * T * D + m_offs[:, None] * D + d_offs[None, :]
    q = tl.load(q_ptr + q_base, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    max_logit = tl.full((BLOCK_M, 1), -float("inf"), dtype=tl.float32)
    softmax_denom = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for n_start in range(0, S, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < S

        k_base = (pid_b * H + pid_h) * S * D + n_offs[:, None] * D + d_offs[None, :]
        v_base = (pid_b * H + pid_h) * S * D + n_offs[:, None] * D + d_offs[None, :]

        k = tl.load(k_ptr + k_base, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptr + v_base, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        scores = tl.dot(q, tl.trans(k))
        scale = 1.0 / tl.sqrt(tl.full((1,), float(D), dtype=tl.float32))
        scores = scores * scale

        new_max = tl.maximum(max_logit, tl.max(scores, axis=1, keep_dims=True))
        exp_old = tl.exp(max_logit - new_max)
        exp_scores = tl.exp(scores - new_max)

        softmax_denom = softmax_denom * exp_old + tl.sum(exp_scores, axis=1, keep_dims=True)
        acc = acc * exp_old + tl.dot(exp_scores.to(tl.float32), v)
        max_logit = new_max

    acc = acc / softmax_denom

    out_base = (pid_b * H + pid_h) * T * D + m_offs[:, None] * D + d_offs[None, :]
    tl.store(out_ptr + out_base, acc.to(tl.float16), mask=m_mask[:, None] & d_mask[None, :])


def flash_attn_like_v1(
    q, k, v,
    B: int, H: int, T: int, S: int, D: int,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_D: int = 64,
    NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    out = triton.empty((B, H, T, D), dtype=q.dtype, device=q.device)
    grid = (B, H, (T + BLOCK_M - 1) // BLOCK_M)
    _flash_attn_v1_kernel[grid](
        q, k, v, out,
        B=B, H=H, T=T, S=S, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
