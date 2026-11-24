"""K7 profiling evidence without hardware counters: traces, compile metadata, variant experiments.

Run on the GPU box (nothing else on the GPU):
  python scripts/profile_kernels.py --out reports/raw/profiling
Writes one JSON file per experiment and prints paste-friendly tables. Timings use do_bench
(median of in-process repetitions); they are controlled comparisons, not pre-registered results.
"""
import argparse
import functools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_INPUT_BYTES = 512 * 1024**2
NCU_PROGRAM = "import torch; x = torch.ones(1 << 20, device='cuda'); x.add_(1); torch.cuda.synchronize()"


def bench(fn):
    """do_bench median in ms after one warm (compile/autotune) call."""
    import torch
    import triton.testing

    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, quantiles=[0.5])[0]


def rows_for(n, elem):
    return (MIN_INPUT_BYTES + n * elem - 1) // (n * elem)


def ncu_probe():
    """Run a trivial CUDA kernel under ncu and keep the verbatim outcome."""
    import shutil

    executable = shutil.which("ncu")
    if executable is None:
        return {"status": "ncu not found on PATH"}
    command = [executable, "--metrics", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
               sys.executable, "-c", NCU_PROGRAM]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    return {"command": " ".join(command[:3]) + " python -c <add kernel>",
            "returncode": result.returncode, "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:]}


def trace_kernels(label, fn):
    """torch.profiler Chrome trace of one call: GPU kernel names, durations, launch shape."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        trace.export_chrome_trace(handle.name)
        events = json.loads(Path(handle.name).read_text())["traceEvents"]
    kernels = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        kernels.append({"op": label, "name": event["name"][:100], "us": event.get("dur"),
                        "grid": args.get("grid"), "block": args.get("block"),
                        "registers_per_thread": args.get("registers per thread"),
                        "shared_memory": args.get("shared memory"),
                        "occupancy_pct": args.get("est. achieved occupancy %")})
    return kernels


def compiled_metadata():
    """Registers, spills, and shared memory of every compiled KernelForge Triton variant."""
    from kernelforge.ops import _attention_kernels, _layernorm_kernels, _softmax_kernels

    jit_functions = {
        "softmax_single": _softmax_kernels._single_softmax.fn,
        "softmax_looped": _softmax_kernels._looped_softmax.fn,
        "layer_norm_single": _layernorm_kernels._single_layer_norm.fn,
        "layer_norm_looped": _layernorm_kernels._looped_layer_norm.fn,
        "attention": _attention_kernels._causal_attention.fn}
    rows = []
    for name, jit in jit_functions.items():
        for device_cache in getattr(jit, "device_caches", {}).values():
            for compiled in device_cache[0].values():
                metadata = compiled.metadata
                rows.append({"kernel": name, "num_warps": metadata.num_warps,
                             "num_stages": metadata.num_stages,
                             "registers": getattr(compiled, "n_regs", None),
                             "spills": getattr(compiled, "n_spills", None),
                             "shared_bytes": metadata.shared,
                             "constants": {k: v for k, v in _constants(metadata).items()
                                           if k in ("BLOCK", "BLOCK_M", "BLOCK_N", "D", "N",
                                                    "STATS", "HAS_WEIGHT", "HAS_BIAS")}})
    return rows


def _constants(metadata):
    """Compile-time constants by argument name, across Triton metadata layouts."""
    constants = getattr(metadata, "constants", {}) or {}
    names = getattr(metadata, "arg_names", None)
    if names and constants and all(isinstance(k, tuple) for k in constants):
        return {names[k[0]]: v for k, v in constants.items() if len(k) == 1}
    return {str(k): v for k, v in constants.items()}


def selected_traces():
    """One trace per op at a pre-registered shape (KernelForge and the PyTorch baseline)."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from kernelforge import causal_attention, layer_norm, softmax

    x = torch.randn((rows_for(8192, 2), 8192), dtype=torch.float16, device="cuda")
    w = torch.randn(8192, dtype=torch.float16, device="cuda")
    b = torch.randn(8192, dtype=torch.float16, device="cuda")
    q, k, v = (torch.randn((4, 32, 4096, 128), dtype=torch.bfloat16, device="cuda")
               for _ in range(3))

    def flash():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    traces = []
    for label, fn in (("softmax N=8192 fp16 (kernelforge)", lambda: softmax(x)),
                      ("softmax N=8192 fp16 (torch)", lambda: torch.softmax(x, -1)),
                      ("layer_norm N=8192 fp16 (kernelforge)",
                       lambda: layer_norm(x, 8192, w, b)),
                      ("layer_norm N=8192 fp16 (torch)", lambda: F.layer_norm(x, (8192,), w, b)),
                      ("attention 4x32x4096x128 bf16 (kernelforge)",
                       lambda: causal_attention(q, k, v)),
                      ("attention 4x32x4096x128 bf16 (sdpa flash)", flash)):
        traces += trace_kernels(label, fn)
    return traces


def alignment_experiment():
    """Same N, natural row stride vs a 64-element-aligned row stride (a padded view)."""
    import torch

    from kernelforge import layer_norm, softmax

    rows = []
    for n in (4096, 4097, 31337):
        for dtype_name, dtype, elem in (("bf16", torch.bfloat16, 2), ("fp32", torch.float32, 4)):
            m = rows_for(n, elem)
            padded = (n + 63) // 64 * 64
            natural = torch.randn((m, n), dtype=dtype, device="cuda")
            aligned = torch.randn((m, padded), dtype=dtype, device="cuda")[:, :n]
            weight = torch.ones(n, dtype=dtype, device="cuda")
            for layout, x in (("natural", natural), ("aligned", aligned)):
                for op, fn in (("softmax", lambda x=x: softmax(x)),
                               ("layer_norm", lambda x=x, n=n, w=weight: layer_norm(x, n, w, w))):
                    p50 = bench(fn)
                    rows.append({"op": op, "N": n, "dtype": dtype_name, "layout": layout,
                                 "row_stride": x.stride(0), "ms_p50": p50,
                                 "gbps": 2 * m * n * elem / (p50 * 1e-3) / 1e9})
            del natural, aligned
            torch.cuda.empty_cache()
    return rows


def _layer_norm_single(x, y, warps):
    """Launch the single-block LayerNorm kernel with one forced num_warps (no affine, no stats)."""
    from kernelforge.ops._common import softmax_grid
    from kernelforge.ops._layernorm_kernels import _single_layer_norm

    m, n = x.shape
    _single_layer_norm.fn[softmax_grid(m)](
        x, y, x, x, y, y, m, n, x.stride(0), 1e-5, HAS_WEIGHT=False, HAS_BIAS=False, STATS=False,
        BLOCK=n, num_warps=warps)


def warps_experiment():
    """LayerNorm single path at fixed BLOCK with each num_warps forced (no autotuning)."""
    import torch

    rows = []
    for n in (8192, 32768):
        m = rows_for(n, 2)
        x = torch.randn((m, n), dtype=torch.float16, device="cuda")
        y = torch.empty_like(x)
        for warps in (1, 2, 4, 8, 16):
            if warps * 32 <= n:
                p50 = bench(functools.partial(_layer_norm_single, x, y, warps))
                rows.append({"N": n, "num_warps": warps, "ms_p50": p50,
                             "gbps": 2 * m * n * 2 / (p50 * 1e-3) / 1e9})
    return rows


def _attention_tile(q, k, v, out, config):
    """Launch the attention kernel with one forced (BLOCK_M, BLOCK_N, warps, stages)."""
    import triton

    from kernelforge.ops._attention_kernels import LOG2E, _causal_attention

    b, h, s, dim = q.shape
    bm, bn, warps, stages = config
    _causal_attention.fn[(triton.cdiv(s, bm), b * h)](
        q, k, v, out, out, dim ** -0.5 * LOG2E, h, s, *q.stride()[:3], *k.stride()[:3],
        *v.stride()[:3], *out.stride()[:3], D=dim, RETURN_LSE=False, BLOCK_M=bm, BLOCK_N=bn,
        num_warps=warps, num_stages=stages)


def attention_tiles_experiment():
    """Attention forced to each tile configuration at a selected shape (no autotuning)."""
    import itertools

    import torch

    rows = []
    for dim in (64, 128):
        b, h, s = 4, 32, 4096
        q, k, v = (torch.randn((b, h, s, dim), dtype=torch.bfloat16, device="cuda")
                   for _ in range(3))
        out = torch.empty_like(q)
        flops = 2 * b * h * s * s * dim
        for config in itertools.product((64, 128), (32, 64, 128), (4, 8), (2, 3)):
            if config[1] > config[0]:
                continue
            row = dict(zip(("BLOCK_M", "BLOCK_N", "num_warps", "num_stages"), config), D=dim)
            try:
                p50 = bench(functools.partial(_attention_tile, q, k, v, out, config))
                rows.append({**row, "ms_p50": p50, "tflops": flops / (p50 * 1e-3) / 1e12})
            except Exception as exc:  # noqa: BLE001 - an invalid tile is a recorded result
                rows.append({**row, "error": type(exc).__name__})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import triton

    env = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
           "triton": triton.__version__}
    steps = (("ncu_probe", ncu_probe), ("traces", selected_traces),
             ("alignment", alignment_experiment), ("warps", warps_experiment),
             ("attention_tiles", attention_tiles_experiment), ("compiled", compiled_metadata))
    for name, step in steps:
        result = step()
        (args.out / f"{name}.json").write_text(json.dumps({"env": env, "result": result},
                                                          indent=2, default=str) + "\n")
        print(f"== {name} ==")
        for row in result if isinstance(result, list) else [result]:
            print(json.dumps(row, default=str)[:300])


if __name__ == "__main__":
    main()
