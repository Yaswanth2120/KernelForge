"""GPU-only LayerNorm forward kernels. Import through the CPU-safe public wrapper."""
import triton
import triton.language as tl

WARPS = (1, 2, 4, 8, 16)
LOOP_BLOCKS = (1024, 2048, 4096)
# Triton also appends every tensor argument's dtype to the autotune key.
KEY = ["N", "HAS_WEIGHT", "HAS_BIAS", "STATS"]
# Winner of the most recent launch, read by the benchmark runner for its result records.
LAST_LAUNCH = {}


def _prune_idle_warps(configs, named_args, **kwargs):
    """Drop configs with more threads than padded columns; Triton skips resource failures."""
    fixed = kwargs.get("BLOCK", named_args.get("BLOCK"))
    kept = [c for c in configs if c.num_warps * 32 <= max(32, c.kwargs.get("BLOCK", fixed))]
    return kept or configs[:1]


@triton.jit
def _row_id():
    """Logical row from a 2-D grid, in int64 so row * stride cannot overflow."""
    return tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * tl.num_programs(0)


@triton.jit
def _rstd(m2, count, eps):
    """1/sqrt(M2/n + eps) with IEEE round-to-nearest ops; the default `/` and sqrt are approximate."""
    return tl.div_rn(tl.full((), 1., tl.float32), tl.sqrt_rn(tl.div_rn(m2, count) + eps))


@triton.jit
def _store_outputs(Y, row, offsets, active, centered, rstd, W, B, N,
                   HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Apply the optional affine transform in fp32 and store one tile of y."""
    y = centered * rstd
    if HAS_WEIGHT:
        y = y * tl.load(W + offsets, mask=offsets < N, other=0.).to(tl.float32)
    if HAS_BIAS:
        y = y + tl.load(B + offsets, mask=offsets < N, other=0.).to(tl.float32)
    tl.store(Y + row * N + offsets, y, mask=active)


@triton.autotune(configs=[triton.Config({}, num_warps=w) for w in WARPS], key=KEY,
                 prune_configs_by={"early_config_prune": _prune_idle_warps})
@triton.jit
def _single_layer_norm(X, Y, W, B, MEAN, RSTD, M, N, row_stride, eps,
                       HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, STATS: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Hold one row in registers; centered two-pass variance avoids E[x^2] - E[x]^2."""
    row = _row_id()
    offsets = tl.arange(0, BLOCK)
    active = (row < M) & (offsets < N)
    x = tl.load(X + row * row_stride + offsets, mask=active, other=0.).to(tl.float32)
    count = tl.full((), N, tl.float32)
    # Exact division keeps constant rows exactly centered (e.g. 93/31 == 3).
    mean = tl.div_rn(tl.sum(x, axis=0), count)
    centered = tl.where(active, x - mean, 0.)
    rstd = _rstd(tl.sum(centered * centered, axis=0), count, eps)
    _store_outputs(Y, row, offsets, active, centered, rstd, W, B, N, HAS_WEIGHT, HAS_BIAS)
    if STATS:
        tl.store(MEAN + row, mean, mask=row < M)
        tl.store(RSTD + row, rstd, mask=row < M)


@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in LOOP_BLOCKS
                          for w in WARPS], key=KEY,
                 prune_configs_by={"early_config_prune": _prune_idle_warps})
@triton.jit
def _looped_layer_norm(X, Y, W, B, MEAN, RSTD, M, N, row_stride, eps,
                       HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, STATS: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Merge per-tile (count, mean, M2) with Chan's formula, then reread and normalize."""
    row = _row_id()
    columns = tl.arange(0, BLOCK).to(tl.int64)
    count = tl.full((), 0., tl.float32)
    mean = tl.full((), 0., tl.float32)
    m2 = tl.full((), 0., tl.float32)
    for tile in range(tl.cdiv(N, BLOCK)):
        offsets = tl.cast(tile, tl.int64) * BLOCK + columns
        active = (row < M) & (offsets < N)
        x = tl.load(X + row * row_stride + offsets, mask=active, other=0.).to(tl.float32)
        tile_count = tl.minimum(N - tile * BLOCK, BLOCK).to(tl.float32)
        tile_mean = tl.div_rn(tl.sum(x, axis=0), tile_count)
        tile_centered = tl.where(active, x - tile_mean, 0.)
        tile_m2 = tl.sum(tile_centered * tile_centered, axis=0)
        total = count + tile_count
        delta = tile_mean - mean
        mean = mean + delta * tl.div_rn(tile_count, total)
        m2 = m2 + tile_m2 + delta * delta * tl.div_rn(count * tile_count, total)
        count = total
    rstd = _rstd(m2, count, eps)
    for tile in range(tl.cdiv(N, BLOCK)):
        offsets = tl.cast(tile, tl.int64) * BLOCK + columns
        active = (row < M) & (offsets < N)
        x = tl.load(X + row * row_stride + offsets, mask=active, other=0.).to(tl.float32)
        _store_outputs(Y, row, offsets, active, x - mean, rstd, W, B, N, HAS_WEIGHT, HAS_BIAS)
    if STATS:
        tl.store(MEAN + row, mean, mask=row < M)
        tl.store(RSTD + row, rstd, mask=row < M)


def launch_layer_norm(rows, output, weight, bias, mean, rstd, eps, grid, single_block_limit):
    """Pick a path by N, launch it, and return the autotuned winner for result records."""
    m, n = rows.shape
    flags = {"HAS_WEIGHT": weight is not None, "HAS_BIAS": bias is not None,
             "STATS": mean is not None}
    # Unused pointers still need a valid tensor argument; their loads are compiled out.
    arguments = (rows, output, rows if weight is None else weight, rows if bias is None else bias,
                 output if mean is None else mean, output if rstd is None else rstd,
                 m, n, rows.stride(0), eps)
    if n <= single_block_limit:
        kernel = _single_layer_norm
        kernel[grid](*arguments, **flags, BLOCK=triton.next_power_of_2(n))
        block = triton.next_power_of_2(n)
    else:
        kernel = _looped_layer_norm
        kernel[grid](*arguments, **flags)
        block = kernel.best_config.kwargs["BLOCK"]
    LAST_LAUNCH.clear()
    LAST_LAUNCH.update({"path": "single" if n <= single_block_limit else "looped", "block": block,
                        "num_warps": kernel.best_config.num_warps,
                        "num_stages": kernel.best_config.num_stages})
    return dict(LAST_LAUNCH)
