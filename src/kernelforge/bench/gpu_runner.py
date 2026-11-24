"""Real-GPU benchmark launches: do_bench timer, telemetry, environment, one process per launch.

Importing this module is CPU-safe; torch.cuda and Triton are touched only inside GPU functions.
Records go to <out>/<suite>/launch_<i>.jsonl; accuracy (launch 0 only) to <out>/<suite>/accuracy.json.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from kernelforge import metrics
from kernelforge.bench import suites
from kernelforge.bench.harness import Autotune, Environment, Telemetry, Timing, benchmark

SUITES = ("copy", "softmax", "layer_norm", "causal_attention")
# Suites whose criteria are bandwidth ratios and therefore need BW_copy first.
RATIO_SUITES = ("softmax", "layer_norm")
LAUNCHES = 3
TELEMETRY_FIELDS = ("clocks.sm", "clocks.max.sm", "clocks.mem", "power.draw", "temperature.gpu",
                    "clocks_throttle_reasons.active")
TORCH_DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
ELEM_SIZE = {"fp16": 2, "bf16": 2, "fp32": 4}


class DoBenchTimer:
    """Timer backed by triton.testing.do_bench (behavior recorded in DECISIONS.md)."""

    def measure(self, operation):
        """Return do_bench's p20/p50/p80 milliseconds for an already-tuned callable."""
        from triton.testing import do_bench

        p20, p50, p80 = do_bench(operation, quantiles=[0.2, 0.5, 0.8])
        return Timing(ms_p20=p20, ms_p50=p50, ms_p80=p80)


def parse_telemetry(line):
    """Turn one `nvidia-smi --format=csv,noheader,nounits` line into a field dictionary."""
    values = [value.strip() for value in line.strip().split(",")]
    if len(values) != len(TELEMETRY_FIELDS):
        return {"error": f"unexpected nvidia-smi output: {line.strip()!r}"}
    return dict(zip(TELEMETRY_FIELDS, values))


def sample_telemetry():
    """Sample clocks, power, temperature, and throttle reasons for GPU 0."""
    result = subprocess.run(
        ["nvidia-smi", "--id=0", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return parse_telemetry(result.stdout) if result.returncode == 0 else {"error": result.stderr}


def clock_drop(telemetry):
    """Flag a launch whose end sample shows a lowered SM clock or a non-idle throttle reason."""
    end = telemetry.get("end", {})
    try:
        sm, max_sm = float(end["clocks.sm"]), float(end["clocks.max.sm"])
        reasons = int(end["clocks_throttle_reasons.active"], 16)
    except (KeyError, ValueError):
        return True  # missing telemetry cannot rule out a drop
    idle_bit = 0x1
    return sm < 0.97 * max_sm or (reasons & ~idle_bit) != 0


def git_sha():
    """Short HEAD SHA, suffixed `-dirty` when tracked source differs from the commit."""
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout

    sha = run("rev-parse", "--short", "HEAD").strip() or "unknown"
    dirty = run("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts",
                "tests", "pyproject.toml").strip()
    return f"{sha}-dirty" if dirty else sha


def environment():
    """Capture the fields the result schema requires from the live CUDA environment."""
    import torch
    import triton

    major, minor = torch.cuda.get_device_capability()
    driver = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=driver_version",
                             "--format=csv,noheader"], capture_output=True, text=True,
                            check=False).stdout.strip() or "unknown"
    return Environment(device=torch.cuda.get_device_name(), cc=f"{major}.{minor}",
                       torch=torch.__version__, triton=triton.__version__,
                       cuda=torch.version.cuda or "unknown", driver=driver, git_sha=git_sha(),
                       date=datetime.datetime.now(datetime.timezone.utc).date().isoformat())


def launch_command(suite, index, out_dir):
    """Command for one independent launch; each launch is a fresh Python process."""
    return [sys.executable, "-m", "kernelforge.bench.gpu_runner", "--suite", suite,
            "--launch-index", str(index), "--out", str(out_dir)]


def run_suite(suite, out_dir, launches=LAUNCHES):
    """Run the registered number of launches sequentially, each in its own process."""
    if suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}; expected one of {SUITES}")
    if launches != LAUNCHES:
        raise ValueError("The registered protocol requires exactly three launches")
    for index in range(launches):
        subprocess.run(launch_command(suite, index, out_dir), check=True)


def copy_baseline(out_dir):
    """Read BW_copy produced by `kernelforge copy-probe`; benchmarks need it for ratios."""
    path = Path(out_dir) / "copy_probe.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `kernelforge copy-probe` first")
    return json.loads(path.read_text())["bw_copy_gbps"]


def _first_call_seconds(operation):
    """Time the first call, which compiles and autotunes; excluded from benchmark timing."""
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _peak_memory_delta(operation):
    """Peak bytes allocated by one call beyond the already-allocated inputs."""
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    operation()
    torch.cuda.synchronize()
    return max(0, torch.cuda.max_memory_allocated() - before)


def copy_operations(case):
    """Triton copy kernel and torch.Tensor.copy_ over the same >= 1 GiB fp32 buffer."""
    import torch

    from kernelforge.ops._copy_kernels import triton_copy

    x = torch.ones(case["buffer_bytes"] // 4, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    tuned = {}

    def run_triton():
        tuned["best"] = triton_copy(x, y)

    return [("triton_copy", run_triton, "kernelforge.ops._copy_kernels.triton_copy(x, y)",
             lambda: tuned.get("best", {}), list),
            ("torch_copy_", lambda: y.copy_(x), "y.copy_(x)", dict, list)]


def softmax_operations(case):
    """KernelForge softmax and torch.softmax on the same deterministic input."""
    import torch

    from kernelforge import softmax
    from kernelforge.ops._softmax_kernels import autotune_entry
    from kernelforge.ops.softmax import SOFTMAX_SINGLE_BLOCK_LIMIT

    generator = torch.Generator(device="cuda").manual_seed(case["N"])
    x = torch.randn((case["M"], case["N"]), generator=generator,
                    dtype=getattr(torch, TORCH_DTYPES[case["dtype"]]), device="cuda")

    def entry():
        return autotune_entry(x, SOFTMAX_SINGLE_BLOCK_LIMIT) or {}

    def best():
        return entry().get("best_config") or {}

    def rejected():
        return [config for config in entry().get("configs", []) if config["status"] != "valid"]

    return [("kernelforge", lambda: softmax(x), "kernelforge.softmax(x, dim=-1)", best, rejected),
            ("torch", lambda: torch.softmax(x, dim=-1), "torch.softmax(x, dim=-1)", dict, list)], x


def softmax_accuracy(case, x):
    """Apply the frozen tolerance rule to one benchmark input (launch 0 only)."""
    import torch

    from kernelforge import reference, softmax
    from kernelforge.tolerance import compare_accuracy

    result = compare_accuracy(softmax(x), torch.softmax(x, dim=-1), reference.softmax(x), x.dtype)
    return {"case": case, "kernel_max_abs": result.kernel.max_abs,
            "torch_max_abs": result.baseline.max_abs, "allowed_max_abs": result.allowed_max_abs,
            "passed": result.passed}


def layer_norm_operations(case):
    """KernelForge layer_norm and F.layer_norm with same-dtype weight and bias over one input."""
    import torch
    import torch.nn.functional as F

    from kernelforge import layer_norm
    from kernelforge.ops import _layernorm_kernels

    dtype = getattr(torch, TORCH_DTYPES[case["dtype"]])
    generator = torch.Generator(device="cuda").manual_seed(case["N"])
    x = torch.randn((case["M"], case["N"]), generator=generator, dtype=dtype, device="cuda")
    weight = torch.randn(case["N"], generator=generator, dtype=dtype, device="cuda")
    bias = torch.randn(case["N"], generator=generator, dtype=dtype, device="cuda")
    shape = (case["N"],)

    def best():
        return dict(_layernorm_kernels.LAST_LAUNCH)

    return [("kernelforge", lambda: layer_norm(x, shape, weight, bias),
             "kernelforge.layer_norm(x, (N,), weight, bias)", best, list),
            ("torch", lambda: F.layer_norm(x, shape, weight, bias),
             "F.layer_norm(x, (N,), weight, bias)", dict, list)], (x, weight, bias)


def layer_norm_accuracy(case, x, weight, bias):
    """Apply the frozen tolerance rule to one benchmark input (launch 0 only)."""
    import torch.nn.functional as F

    from kernelforge import layer_norm, reference
    from kernelforge.tolerance import compare_accuracy

    shape = (case["N"],)
    result = compare_accuracy(layer_norm(x, shape, weight, bias),
                              F.layer_norm(x, shape, weight, bias),
                              reference.layer_norm(x, shape, weight, bias), x.dtype)
    return {"case": case, "kernel_max_abs": result.kernel.max_abs,
            "torch_max_abs": result.baseline.max_abs, "allowed_max_abs": result.allowed_max_abs,
            "passed": result.passed}


def layer_norm_bytes(case):
    elem = ELEM_SIZE[case["dtype"]]
    return metrics.layer_norm_bytes(case["M"], case["N"], elem, elem, elem)


def _sdpa(q, k, v, backend):
    """F.scaled_dot_product_attention(is_causal=True), optionally forced to one backend."""
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel

    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def default_sdpa_kernel(q, k, v):
    """Name the CUDA kernel that dominates the default SDPA backend's run (torch.profiler)."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        _sdpa(q, k, v, None)
        torch.cuda.synchronize()
    events = [(getattr(e, "self_device_time_total", 0), e.key) for e in trace.key_averages()]
    events = [event for event in events if event[0] > 0]
    return max(events)[1][:120] if events else "unknown"


def attention_operations(case):
    """KernelForge, SDPA forced to FlashAttention, and SDPA's default backend on one input."""
    import torch
    from torch.nn.attention import SDPBackend

    from kernelforge import causal_attention
    from kernelforge.ops import _attention_kernels as kernels

    shape = (case["B"], case["H"], case["S"], case["D"])
    dtype = getattr(torch, TORCH_DTYPES[case["dtype"]])
    generator = torch.Generator(device="cuda").manual_seed(case["S"] * case["D"])
    q, k, v = (torch.randn(shape, generator=generator, dtype=dtype, device="cuda")
               for _ in range(3))
    flash = SDPBackend.FLASH_ATTENTION
    call = "F.scaled_dot_product_attention(q, k, v, is_causal=True)"

    def best():
        return dict(kernels.LAST_LAUNCH.get("best_config", {}))

    def rejected():
        return [c for c in kernels.LAST_LAUNCH.get("configs", []) if c["status"] != "valid"]

    return [("kernelforge", lambda: causal_attention(q, k, v),
             "kernelforge.causal_attention(q, k, v)", best, rejected),
            ("sdpa_flash", lambda: _sdpa(q, k, v, flash),
             f"{call} under sdpa_kernel(SDPBackend.FLASH_ATTENTION)", dict, list),
            ("sdpa_default", lambda: _sdpa(q, k, v, None),
             f"{call}, default backend; kernel {default_sdpa_kernel(q, k, v)}", dict, list)], \
        (q, k, v)


def attention_accuracy(case, q, k, v):
    """Frozen rule on the first batch's first two heads (the reference materializes S x S)."""
    from torch.nn.attention import SDPBackend

    from kernelforge import causal_attention, reference
    from kernelforge.tolerance import compare_accuracy

    q, k, v = (t[:1, :2] for t in (q, k, v))
    result = compare_accuracy(causal_attention(q, k, v),
                              _sdpa(q, k, v, SDPBackend.FLASH_ATTENTION),
                              reference.causal_attention(q, k, v), q.dtype)
    return {"case": case, "kernel_max_abs": result.kernel.max_abs,
            "torch_max_abs": result.baseline.max_abs, "allowed_max_abs": result.allowed_max_abs,
            "passed": result.passed, "slice": "batch 0, heads 0-1"}


def _attention_memory_case(case):
    """Peak-memory rows for one S; an out-of-memory failure is recorded as a result."""
    import torch
    from torch.nn.attention import SDPBackend

    from kernelforge import causal_attention

    shape = (case["B"], case["H"], case["S"], case["D"])
    q, k, v = (torch.randn(shape, dtype=torch.float16, device="cuda") for _ in range(3))
    operations = {"kernelforge": lambda: causal_attention(q, k, v),
                  "sdpa_flash": lambda: _sdpa(q, k, v, SDPBackend.FLASH_ATTENTION),
                  "sdpa_math": lambda: _sdpa(q, k, v, SDPBackend.MATH)}
    rows = []
    for impl, operation in operations.items():
        try:
            operation()  # compile/tune outside the measured call
            rows.append({"case": case, "impl": impl,
                         "peak_mem_delta_bytes": _peak_memory_delta(operation)})
        except torch.OutOfMemoryError:
            rows.append({"case": case, "impl": impl, "peak_mem_delta_bytes": None,
                         "error": "OutOfMemoryError"})
        torch.cuda.empty_cache()
    return rows


def attention_memory():
    """Peak memory over the inputs vs S for KernelForge, SDPA-flash, and SDPA-math."""
    import torch

    rows = []
    for case in suites.attention_memory_cases():
        rows += _attention_memory_case(case)
        torch.cuda.empty_cache()
    return rows


def attention_bytes(case):
    """q, k, v read plus o written (informational; attention is judged in TFLOPS)."""
    return 4 * case["B"] * case["H"] * case["S"] * case["D"] * ELEM_SIZE[case["dtype"]]


def attention_flops(case):
    return metrics.attention_flops(case["B"], case["H"], case["S"], case["D"])


def softmax_bytes(case):
    return metrics.softmax_bytes(case["M"], case["N"], ELEM_SIZE[case["dtype"]])


def copy_bytes(case):
    return 2 * case["buffer_bytes"]


def _suite(suite, case):
    """(operations, accuracy inputs, byte accounting, FLOP accounting, accuracy fn) for a case."""
    if suite == "copy":
        return copy_operations(case), None, copy_bytes, None, None
    if suite == "softmax":
        operations, x = softmax_operations(case)
        return operations, (x,), softmax_bytes, None, softmax_accuracy
    if suite == "layer_norm":
        operations, inputs = layer_norm_operations(case)
        return operations, inputs, layer_norm_bytes, None, layer_norm_accuracy
    operations, inputs = attention_operations(case)
    return operations, inputs, attention_bytes, attention_flops, attention_accuracy


def _cases(suite):
    if suite == "copy":
        return suites.copy_cases()
    return suites.attention_sweep() if suite == "causal_attention" else suites.row_sweep()


def run_launch(suite, index, out_dir):
    """Measure every case of a suite once in this process and write its JSONL records."""
    import torch

    from kernelforge import is_triton_available

    if not is_triton_available():
        raise RuntimeError("GPU benchmarks require NVIDIA sm_80+ CUDA and PyTorch-bundled Triton")
    copy_gbps = copy_baseline(out_dir) if suite in RATIO_SUITES else None
    env, start = environment(), sample_telemetry()
    records, accuracy = [], []
    for case in _cases(suite):
        operations, inputs, byte_fn, flop_fn, accuracy_fn = _suite(suite, case)
        for impl, operation, call, best, rejected in operations:
            tuning = _first_call_seconds(operation)
            peak = _peak_memory_delta(operation)
            record = benchmark(operation, DoBenchTimer(), op=suite, impl=impl, case=case,
                               launch_index=index, env=env, byte_accounting=byte_fn,
                               flop_accounting=flop_fn, copy_gbps=copy_gbps,
                               peak_mem_delta_bytes=peak,
                               autotune=Autotune(best_config=best(), tuning_seconds=tuning,
                                                 invalid_configs=rejected()),
                               baseline_call=call)
            records.append(record)
            rate = (f"{record.tflops:.1f} TFLOPS" if record.tflops is not None
                    else f"{record.gbps:.1f} GB/s")
            print(f"{suite} {impl} {case} p50={record.ms_p50:.4f} ms {rate}", flush=True)
        if accuracy_fn is not None and index == 0:
            accuracy.append(accuracy_fn(case, *inputs))
        del operations, inputs
        torch.cuda.empty_cache()
    telemetry = Telemetry(start=start, end=sample_telemetry())
    directory = Path(out_dir) / suite
    directory.mkdir(parents=True, exist_ok=True)
    lines = [record.model_copy(update={"gpu_telemetry": telemetry}).model_dump_json(by_alias=True)
             for record in records]
    (directory / f"launch_{index}.jsonl").write_text("\n".join(lines) + "\n")
    if accuracy:
        (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    if suite == "causal_attention" and index == 0:
        (directory / "memory.json").write_text(json.dumps(attention_memory(), indent=2) + "\n")


def main(argv=None):
    """Entry point for one launch subprocess (used by run_suite)."""
    parser = argparse.ArgumentParser(prog="kernelforge.bench.gpu_runner")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--launch-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_launch(args.suite, args.launch_index, args.out)


if __name__ == "__main__":
    main()
