"""GPU-only softmax kernels. Import through the CPU-safe public wrapper."""
import triton
import triton.language as tl
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError

from ._common import prune_softmax_configs

WARPS = (1, 2, 4, 8, 16)
LOOP_BLOCKS = (1024, 2048, 4096)
KEY_FIELDS = ("N", "DTYPE", "DEVICE", "M", "LEADING_SHAPE", "LEADING_STRIDES")
PRUNING_LOG = {}


@triton.jit
def _row_location(LEADING_SHAPE: tl.constexpr, LEADING_STRIDES: tl.constexpr):
    """Map a two-dimensional grid ID to an int64 logical row and input offset."""
    row = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * tl.num_programs(0)
    remaining = row
    offset = tl.full((), 0, tl.int64)
    # Iterate axes in reverse with static_range so `axis` stays a compile-time constant;
    # arithmetic on the loop index turns it into a tensor, which cannot index a tuple.
    for axis in tl.static_range(len(LEADING_SHAPE) - 1, -1, -1):
        offset += (remaining % LEADING_SHAPE[axis]) * LEADING_STRIDES[axis]
        remaining = remaining // LEADING_SHAPE[axis]
    return row, offset


def _prune(configs, named_args, kwargs, path, kernel):
    """Compile candidates without executing them and retain resource-fit evidence."""
    arguments = {**named_args, **kwargs}
    parameters = {name: arguments[name] for name in kernel.arg_names if name in arguments}

    def compile_config(config):
        """Load a candidate's compiled handles so resource failures precede tuning."""
        compiled = kernel.warmup(**parameters, **config.all_kwargs(), grid=(1, 1),
                                 enable_fp_fusion=False)
        compiled._init_handles()
        return compiled

    valid, records = prune_softmax_configs(
        configs, arguments.get("BLOCK"), compile_config,
        (OutOfResources, CompileTimeAssertionFailure, PTXASError))
    key = (path, *(arguments[name] for name in KEY_FIELDS))
    PRUNING_LOG[key] = {"path": path, **{name: arguments[name] for name in KEY_FIELDS},
                        "configs": records, "best_config": None}
    return valid


def _prune_single(configs, named_args, **kwargs):
    """Prune the row-in-registers autotune candidates."""
    return _prune(configs, named_args, kwargs, "single", _single_softmax.fn)


def _prune_looped(configs, named_args, **kwargs):
    """Prune the online-normalizer autotune candidates."""
    return _prune(configs, named_args, kwargs, "looped", _looped_softmax.fn)


@triton.autotune(configs=[triton.Config({}, num_warps=w, num_stages=1) for w in WARPS],
                 key=list(KEY_FIELDS), prune_configs_by={"early_config_prune": _prune_single})
@triton.jit
def _single_softmax(X, Y, M: tl.constexpr, N: tl.constexpr, LEADING_SHAPE: tl.constexpr,
                    LEADING_STRIDES: tl.constexpr, DTYPE: tl.constexpr, DEVICE: tl.constexpr,
                    BLOCK: tl.constexpr):
    """Keep one padded row in a program and perform a stable fp32 reduction."""
    row, source = _row_location(LEADING_SHAPE, LEADING_STRIDES)
    columns = tl.arange(0, BLOCK)
    active = (row < M) & (columns < N)
    values = tl.load(X + source + columns, mask=active, other=-float("inf")).to(tl.float32)
    maximum = tl.max(values, axis=0)
    numerators = tl.exp(values - maximum)
    denominator = tl.sum(numerators, axis=0)
    tl.store(Y + row * N + columns, numerators / denominator, mask=active)


@triton.autotune(configs=[triton.Config({"BLOCK": block}, num_warps=w, num_stages=1)
                         for block in LOOP_BLOCKS for w in WARPS],
                 key=list(KEY_FIELDS), prune_configs_by={"early_config_prune": _prune_looped})
@triton.jit
def _looped_softmax(X, Y, M: tl.constexpr, N: tl.constexpr, LEADING_SHAPE: tl.constexpr,
                    LEADING_STRIDES: tl.constexpr, DTYPE: tl.constexpr, DEVICE: tl.constexpr,
                    BLOCK: tl.constexpr):
    """Compute an online normalizer in bounded tiles, then reread and normalize."""
    row, source = _row_location(LEADING_SHAPE, LEADING_STRIDES)
    columns = tl.arange(0, BLOCK).to(tl.int64)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    correction = tl.full((), 0, tl.float32)
    for tile in range(tl.cdiv(N, BLOCK)):
        offsets = tl.cast(tile, tl.int64) * BLOCK + columns
        values = tl.load(X + source + offsets, mask=(row < M) & (offsets < N),
                         other=-float("inf")).to(tl.float32)
        merged_max = tl.maximum(maximum, tl.max(values, axis=0))
        # Empty prefixes must not inject NaN via -inf - -inf before a finite tile arrives.
        safe_max = tl.where(merged_max == -float("inf"), 0., merged_max)
        rescale = tl.exp(maximum - safe_max)
        scaled_sum = denominator * rescale
        increment = tl.sum(tl.exp(values - safe_max), axis=0) - correction * rescale
        updated_sum = scaled_sum + increment
        # Compensate the scalar sum across many tiles; fp fusion is disabled on launch.
        correction = (updated_sum - scaled_sum) - increment
        denominator = updated_sum
        maximum = merged_max
    for tile in range(tl.cdiv(N, BLOCK)):
        offsets = tl.cast(tile, tl.int64) * BLOCK + columns
        active = (row < M) & (offsets < N)
        values = tl.load(X + source + offsets, mask=active, other=-float("inf")).to(tl.float32)
        probabilities = tl.exp(values - maximum) / denominator
        tl.store(Y + row * N + offsets, probabilities, mask=active)


def _key_arguments(x):
    """Kernel arguments that also form the autotune key (everything except the tensors)."""
    return {"M": x.numel() // x.shape[-1], "N": x.shape[-1],
            "LEADING_SHAPE": tuple(x.shape[:-1]), "LEADING_STRIDES": tuple(x.stride()[:-1]),
            "DTYPE": str(x.dtype), "DEVICE": x.device.index}


def autotune_entry(x, single_block_limit):
    """Return the pruning/winner record for the last launch on a tensor shaped like x."""
    path = "single" if x.shape[-1] <= single_block_limit else "looped"
    arguments = _key_arguments(x)
    return PRUNING_LOG.get((path, *(arguments[name] for name in KEY_FIELDS)))


def launch_softmax(x, output, grid, single_block_limit):
    """Launch a path and retain its winning configuration for later GPU artifacts."""
    arguments = {"X": x, "Y": output, **_key_arguments(x)}
    path = "single" if x.shape[-1] <= single_block_limit else "looped"
    kernel = _single_softmax if path == "single" else _looped_softmax
    if path == "single":
        kernel[grid](**arguments, BLOCK=triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False)
    else:
        kernel[grid](**arguments, enable_fp_fusion=False)
    key = (path, *(arguments[name] for name in KEY_FIELDS))
    PRUNING_LOG[key]["best_config"] = {
        "block": kernel.best_config.kwargs.get("BLOCK", triton.next_power_of_2(x.shape[-1])),
        "num_warps": kernel.best_config.num_warps, "num_stages": kernel.best_config.num_stages}
