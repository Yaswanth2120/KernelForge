"""GPU-only causal attention forward (FlashAttention-2 style). Import through the public wrapper.

Structure follows the published FlashAttention-2 forward algorithm (Dao, 2023): one program per
(query block, batch*head), an online softmax over key/value blocks, and normalization at the end.
"""
import itertools

import triton
import triton.language as tl

BLOCKS = (32, 64, 128)
WARPS = (4, 8)
STAGES = (2, 3, 4)
LOG2E = 1.4426950408889634
KEY = ["S", "D"]  # Triton also appends every tensor argument's dtype to the autotune key.
# Winner of the most recent launch, and per-tuning config validity keyed by
# (S, D, dtype, return_lse), for tests and benchmark result records.
LAST_LAUNCH = {}
TUNING_LOG = {}


def _prune_blocks(configs, named_args, **kwargs):
    """The diagonal split needs BLOCK_N <= BLOCK_M; a block larger than S only adds padding."""
    s = named_args["S"]
    kept = [c for c in configs if c.kwargs["BLOCK_N"] <= c.kwargs["BLOCK_M"]
            and (c.kwargs["BLOCK_M"] == 32 or c.kwargs["BLOCK_M"] // 2 < s)]
    return kept or configs[:1]


@triton.jit
def _attend_blocks(acc, l_i, m_i, q, k_base, v_base, stride_ks, stride_vs, offs_m, lo, hi, S,
                   scale_log2, MASK: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    """Fold key/value blocks [lo, hi) into the running (max m, sum l, unnormalized acc)."""
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        start = tl.cast(start_n, tl.int64)
        k_ptrs = k_base + start * stride_ks + offs_n[None, :] * stride_ks + offs_d[:, None]
        v_ptrs = v_base + start * stride_vs + offs_n[:, None] * stride_vs + offs_d[None, :]
        if MASK:
            k = tl.load(k_ptrs, mask=cols[None, :] < S, other=0.)
            v = tl.load(v_ptrs, mask=cols[:, None] < S, other=0.)
        else:  # every key here precedes the query block, so it is inside S and visible
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        scores = tl.dot(q, k) * scale_log2  # (BLOCK_M, BLOCK_N), log2 units
        if MASK:
            visible = (offs_m[:, None] >= cols[None, :]) & (cols[None, :] < S)
            scores = tl.where(visible, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.math.exp2(scores - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    return acc, l_i, m_i


@triton.autotune(configs=[triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
                          for bm, bn, w, s in itertools.product(BLOCKS, BLOCKS, WARPS, STAGES)],
                 key=KEY, prune_configs_by={"early_config_prune": _prune_blocks})
@triton.jit
def _causal_attention(Q, K, V, O, LSE, scale_log2, H, S,
                      stride_qb, stride_qh, stride_qs, stride_kb, stride_kh, stride_ks,
                      stride_vb, stride_vh, stride_vs, stride_ob, stride_oh, stride_os,
                      D: tl.constexpr, RETURN_LSE: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """One query block of one (batch, head): unmasked blocks, then diagonal blocks, skip the rest."""
    block_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b, h = bh // H, bh % H
    first_row = tl.cast(block_m, tl.int64) * BLOCK_M
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_base = Q + b * stride_qb + h * stride_qh + first_row * stride_qs
    q = tl.load(q_base + rows[:, None] * stride_qs + offs_d[None, :],
                mask=offs_m[:, None] < S, other=0.)
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    diagonal = block_m * BLOCK_M
    # Keys before this query block are visible to every row in it: no mask needed.
    acc, l_i, m_i = _attend_blocks(acc, l_i, m_i, q, k_base, v_base, stride_ks, stride_vs,
                                   offs_m, 0, diagonal, S, scale_log2, False, BLOCK_N, D)
    # Keys inside the query block's span need the causal (j <= i) and tail (j < S) masks.
    # Key blocks after it are entirely above the diagonal and are never visited.
    acc, l_i, m_i = _attend_blocks(acc, l_i, m_i, q, k_base, v_base, stride_ks, stride_vs,
                                   offs_m, diagonal, tl.minimum(diagonal + BLOCK_M, S), S,
                                   scale_log2, True, BLOCK_N, D)
    out = acc / l_i[:, None]
    o_base = O + b * stride_ob + h * stride_oh + first_row * stride_os
    tl.store(o_base + rows[:, None] * stride_os + offs_d[None, :], out.to(O.dtype.element_ty),
             mask=offs_m[:, None] < S)
    if RETURN_LSE:
        # m and l are in log2 units of the scaled scores; convert to natural log.
        lse = (m_i + tl.math.log2(l_i)) / 1.4426950408889634
        tl.store(LSE + bh * S + offs_m, lse, mask=offs_m < S)


def launch_causal_attention(q, k, v, out, lse, sm_scale):
    """Launch with grid (query blocks, B*H) and record the winner and invalid configurations."""
    batch, heads, seq, dim = q.shape

    def grid(meta):
        return (triton.cdiv(seq, meta["BLOCK_M"]), batch * heads)

    tuning_key = (seq, dim, str(q.dtype), lse is not None)
    tuned_before = len(_causal_attention.cache)
    _causal_attention.configs_timings = None  # never report a previous tuning's timings
    _causal_attention[grid](
        q, k, v, out, out if lse is None else lse, sm_scale * LOG2E, heads, seq,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *out.stride()[:3],
        D=dim, RETURN_LSE=lse is not None)
    timings = _causal_attention.configs_timings
    if len(_causal_attention.cache) > tuned_before and timings:
        TUNING_LOG[tuning_key] = [
            {"block_m": c.kwargs["BLOCK_M"], "block_n": c.kwargs["BLOCK_N"],
             "num_warps": c.num_warps, "num_stages": c.num_stages,
             "status": "invalid" if _is_failed(t) else "valid"} for c, t in timings.items()]
    best = _causal_attention.best_config
    LAST_LAUNCH.clear()
    LAST_LAUNCH.update({
        "best_config": {"block_m": best.kwargs["BLOCK_M"], "block_n": best.kwargs["BLOCK_N"],
                        "num_warps": best.num_warps, "num_stages": best.num_stages},
        "configs": TUNING_LOG.get(tuning_key, [])})
    return dict(LAST_LAUNCH)


def _is_failed(timing):
    """Triton records a config that failed to compile or launch with infinite time."""
    values = timing if isinstance(timing, (list, tuple)) else [timing]
    return any(value == float("inf") for value in values)
