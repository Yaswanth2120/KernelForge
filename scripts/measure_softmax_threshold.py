"""Time the single-block and looped softmax paths across N to justify the path threshold.

Run on the GPU box:  python scripts/measure_softmax_threshold.py --out reports/raw/softmax_threshold
Each launch is a fresh process. Decision rule (fixed before measurement, see DECISIONS.md):
the limit is the largest power-of-two N whose single-path median-of-three-medians is <= the
looped path's for every dtype, at that N and every smaller power-of-two N.
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


def measure_case(n, dtype_name, path):
    """Autotune one path on a >=512 MiB input, then time it with do_bench."""
    import torch
    import triton.testing

    from kernelforge.ops._common import softmax_grid
    from kernelforge.ops._softmax_kernels import launch_softmax

    elem = DTYPES[dtype_name]
    m = (MIN_INPUT_BYTES + n * elem - 1) // (n * elem)
    dtype = getattr(torch, {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[dtype_name])
    generator = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn((m, n), generator=generator, dtype=dtype, device="cuda")
    y = torch.empty_like(x)
    grid = softmax_grid(m)

    def run():
        launch_softmax(x, y, grid, FORCE_LIMIT[path])

    torch.cuda.synchronize()
    run()  # autotunes on first call; excluded from timing
    torch.cuda.synchronize()
    p20, p50, p80 = triton.testing.do_bench(run, quantiles=[0.2, 0.5, 0.8])
    from kernelforge.ops import _softmax_kernels as kernels
    key = next(k for k in kernels.PRUNING_LOG if k[0] == path and k[1] == n and k[4] == m
               and k[2] == str(dtype))
    best = kernels.PRUNING_LOG[key]["best_config"]
    del x, y
    torch.cuda.empty_cache()
    return {"N": n, "M": m, "dtype": dtype_name, "path": path, "ms_p20": p20, "ms_p50": p50,
            "ms_p80": p80, "gbps": 2 * m * n * elem / (p50 * 1e-3) / 1e9, "best_config": best}


def run_launch(index, out_dir):
    """Measure every case once in this process and write one JSON file."""
    import torch
    import triton

    device = torch.cuda.get_device_name()
    record = {"schema": "kernelforge.softmax_threshold.v1", "launch_index": index,
              "device": device, "torch": torch.__version__, "triton": triton.__version__,
              "telemetry_start": telemetry(), "cases": []}
    for n in DECISION_WIDTHS + INFO_WIDTHS:
        for dtype_name in DTYPES:
            for path in ("single", "looped"):
                try:
                    record["cases"].append(measure_case(n, dtype_name, path))
                except Exception as exc:  # recorded as a result, not hidden
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


def summarize(out_dir):
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
                   <= medians.get((n, d, "looped"), float("-inf")) for d in DTYPES)
        if not wins:
            break
        limit = n
    summary = {"medians_ms": {f"{n}/{d}/{p}": v for (n, d, p), v in sorted(medians.items())},
               "decision_limit": limit, "devices": sorted({x["device"] for x in launches}),
               "telemetry": [{"start": x["telemetry_start"], "end": x["telemetry_end"]}
                             for x in launches]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nDecision limit by the pre-stated rule: {limit}")
    for i, t in enumerate(summary["telemetry"]):
        print(f"launch {i} telemetry start={t['start']} end={t['end']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/raw/softmax_threshold"))
    parser.add_argument("--launch-index", type=int, help="internal: run one launch in-process")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.launch_index is not None:
        run_launch(args.launch_index, args.out)
        return
    for index in range(LAUNCHES):
        subprocess.run([sys.executable, __file__, "--out", str(args.out),
                        "--launch-index", str(index)], check=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
