"""Time an op's single-block and looped paths across N to justify its path threshold.

Run on the GPU box:
  python scripts/measure_path_threshold.py --op softmax --out reports/raw/softmax_threshold
  python scripts/measure_path_threshold.py --op layer_norm --out reports/raw/layer_norm_threshold
Each launch is a fresh process. Decision rules are fixed before measurement in DECISIONS.md;
`--tie-band` is the relative tolerance within which the single path counts as not slower.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

DECISION_WIDTHS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
INFO_WIDTHS = (4097, 6144, 8193, 12288)
DTYPES = {"fp16": 2, "bf16": 2, "fp32": 4}
MIN_INPUT_BYTES = 512 * 1024**2
LAUNCHES = 3
# launch_softmax picks the single path iff N <= limit, so these force one path.
FORCE_LIMIT = {"single": 2**31, "looped": 0}


def telemetry():
    """Sample clocks, power, temperature, and throttle reasons from nvidia-smi."""
    fields = "clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)
    return dict(zip(fields.split(","), (v.strip() for v in result.stdout.split(","))))


def _inputs(op, n, dtype_name):
    """A >= 512 MiB seeded input (plus same-dtype weight and bias for LayerNorm)."""
    import torch

    elem = DTYPES[dtype_name]
    m = (MIN_INPUT_BYTES + n * elem - 1) // (n * elem)
    dtype = getattr(torch, {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[dtype_name])
    generator = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn((m, n), generator=generator, dtype=dtype, device="cuda")
    if op == "softmax":
        return x, None, None
    return (x, torch.randn(n, generator=generator, dtype=dtype, device="cuda"),
            torch.randn(n, generator=generator, dtype=dtype, device="cuda"))


def _runner(op, x, weight, bias, y, path):
    """Callable that launches exactly one path, plus a function returning its winner."""
    from kernelforge.ops._common import softmax_grid

    grid = softmax_grid(x.shape[0])
    if op == "softmax":
        from kernelforge.ops import _softmax_kernels as kernels

        def best():
            return kernels.autotune_entry(x, FORCE_LIMIT[path])["best_config"]

        return lambda: kernels.launch_softmax(x, y, grid, FORCE_LIMIT[path]), best
    from kernelforge.ops import _layernorm_kernels as kernels

    def run():
        kernels.launch_layer_norm(x, y, weight, bias, None, None, 1e-5, grid, FORCE_LIMIT[path])

    return run, lambda: dict(kernels.LAST_LAUNCH)


def measure_case(op, n, dtype_name, path):
    """Autotune one path on a >=512 MiB input, then time it with do_bench."""
    import torch
    import triton.testing

    x, weight, bias = _inputs(op, n, dtype_name)
    y = torch.empty_like(x)
    run, best = _runner(op, x, weight, bias, y, path)
    torch.cuda.synchronize()
    run()  # autotunes on first call; excluded from timing
    torch.cuda.synchronize()
    winner = best()
    p20, p50, p80 = triton.testing.do_bench(run, quantiles=[0.2, 0.5, 0.8])
    m, elem = x.shape[0], DTYPES[dtype_name]
    moved = 2 * m * n * elem + (0 if op == "softmax" else 2 * n * elem)
    return {"N": n, "M": m, "dtype": dtype_name, "path": path, "ms_p20": p20, "ms_p50": p50,
            "ms_p80": p80, "gbps": moved / (p50 * 1e-3) / 1e9, "best_config": winner}


def run_launch(op, index, out_dir):
    """Measure every case once in this process and write one JSON file."""
    import torch
    import triton

    device = torch.cuda.get_device_name()
    record = {"schema": "kernelforge.path_threshold.v1", "op": op, "launch_index": index,
              "device": device, "torch": torch.__version__, "triton": triton.__version__,
              "telemetry_start": telemetry(), "cases": []}
    for n in DECISION_WIDTHS + INFO_WIDTHS:
        for dtype_name in DTYPES:
            for path in ("single", "looped"):
                try:
                    record["cases"].append(measure_case(op, n, dtype_name, path))
                    torch.cuda.empty_cache()
                # Any compile/launch/OOM failure is a result to record, not to hide or crash on.
                except Exception as exc:  # noqa: BLE001
                    record["cases"].append({"N": n, "dtype": dtype_name, "path": path,
                                            "error": f"{type(exc).__name__}: {exc}"[:500]})
                print(json.dumps(record["cases"][-1]), flush=True)
    record["telemetry_end"] = telemetry()
    (out_dir / f"launch_{index}.json").write_text(json.dumps(record, indent=2) + "\n")


def _cell(medians, spreads, key):
    """Format a median with its launch range, or mark the case as failed."""
    if key not in medians:
        return "error"
    low, high = spreads[key]
    return f"{medians[key]:.4f} ({low:.4f}-{high:.4f})"


def summarize(out_dir, tie_band):
    """Combine launches, apply the decision rule, and print a paste-friendly table."""
    launches = [json.loads((out_dir / f"launch_{i}.json").read_text()) for i in range(LAUNCHES)]
    medians, spreads = {}, {}
    for key in {(c["N"], c["dtype"], c["path"]) for c in launches[0]["cases"]}:
        cases = [c for launch in launches for c in launch["cases"]
                 if (c["N"], c["dtype"], c["path"]) == key and "error" not in c]
        if len(cases) == LAUNCHES:
            medians[key] = statistics.median(c["ms_p50"] for c in cases)
            spreads[key] = (min(c["ms_p50"] for c in cases), max(c["ms_p50"] for c in cases))
    lines = ["| N | dtype | single ms (range) | looped ms (range) | single/looped |",
             "|---|---|---|---|---|"]
    for n in DECISION_WIDTHS + INFO_WIDTHS:
        for dtype_name in DTYPES:
            s, lo = medians.get((n, dtype_name, "single")), medians.get((n, dtype_name, "looped"))
            ratio = f"{s / lo:.3f}" if s and lo else "n/a"
            lines.append(f"| {n} | {dtype_name} | {_cell(medians, spreads, (n, dtype_name, 'single'))}"
                         f" | {_cell(medians, spreads, (n, dtype_name, 'looped'))} | {ratio} |")
    limit = None
    for n in DECISION_WIDTHS:
        wins = all(medians.get((n, d, "single"), float("inf"))
                   <= (1 + tie_band) * medians.get((n, d, "looped"), float("-inf"))
                   for d in DTYPES)
        if not wins:
            break
        limit = n
    summary = {"medians_ms": {f"{n}/{d}/{p}": v for (n, d, p), v in sorted(medians.items())},
               "decision_limit": limit, "tie_band": tie_band, "devices": sorted({x["device"] for x in launches}),
               "telemetry": [{"start": x["telemetry_start"], "end": x["telemetry_end"]}
                             for x in launches]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nDecision limit (tie band {tie_band:.0%}): {limit}")
    for i, t in enumerate(summary["telemetry"]):
        print(f"launch {i} telemetry start={t['start']} end={t['end']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("softmax", "layer_norm"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tie-band", type=float, default=0.01)
    parser.add_argument("--launch-index", type=int, help="internal: run one launch in-process")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.launch_index is not None:
        run_launch(args.op, args.launch_index, args.out)
        return
    if not args.summarize_only:
        for index in range(LAUNCHES):
            subprocess.run([sys.executable, __file__, "--op", args.op, "--out", str(args.out),
                            "--launch-index", str(index)], check=True)
    summarize(args.out, args.tie_band)


if __name__ == "__main__":
    main()
