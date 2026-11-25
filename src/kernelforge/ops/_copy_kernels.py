"""GPU-only copy kernel used as the memory-bandwidth baseline (BW_copy)."""
import triton
import triton.language as tl


@triton.autotune(configs=[triton.Config({"BLOCK": block}, num_warps=warps)
                          for block in (1024, 2048, 4096, 8192) for warps in (4, 8)],
                 key=["n_elements"])
@triton.jit
def _copy(X, Y, n_elements, BLOCK: tl.constexpr):
    """Each program copies one contiguous tile; int64 offsets allow buffers over 2^31 elements."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(Y + offsets, tl.load(X + offsets, mask=mask), mask=mask)


def triton_copy(x, y):
    """Copy contiguous x into y with the autotuned Triton kernel; returns the best config."""
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]),)

    _copy[grid](x, y, n)
    best = _copy.best_config
    return {"block": best.kwargs["BLOCK"], "num_warps": best.num_warps,
            "num_stages": best.num_stages}
