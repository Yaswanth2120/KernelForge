"""Generate reports/performance_report.md from committed raw JSON only (no hand-edited numbers).

Rules enforced here: a ratio is printed only from a three-launch aggregate (n is shown), and
criteria are evaluated only when every contributing record's device name contains "A100".
"""
import json
from pathlib import Path

from kernelforge.bench import suites
from kernelforge.bench.harness import group_launches, load_launch_records

DATASHEET_GBPS = 2039.0
RATIO_THRESHOLD = 0.80
DTYPE_ORDER = ("fp16", "bf16", "fp32")
ELEM_SIZE = {"fp16": 2, "bf16": 2, "fp32": 4}
KERNELFORGE_COLOR, TORCH_COLOR, REFERENCE_COLOR = "#2a78d6", "#eb6834", "#898781"
# Row-wise bandwidth-bound operations share shapes, criteria form, and report sections.
ROW_OPS = {
    "softmax": {"title": "Softmax", "criterion": 2, "torch": "torch.softmax",
                "call": "torch.softmax(x, dim=-1)", "bytes": "2*M*N*elem_size"},
    "layer_norm": {"title": "LayerNorm", "criterion": 3, "torch": "F.layer_norm",
                   "call": "F.layer_norm(x, (N,), weight, bias)",
                   "bytes": "2*M*N*elem_size + N*(weight_elem + bias_elem); weight and bias "
                            "use the input dtype; stats are not returned"},
}


def is_a100(device):
    """The only accepted evidence that a record came from an A100."""
    return "A100" in device


def _key(op, impl, case):
    return (op, impl, json.dumps(case, sort_keys=True))


def load_groups(raw_dir, suite):
    """Group a suite's launch records; missing suites yield no groups."""
    return group_launches(load_launch_records(Path(raw_dir) / suite))


def _devices(groups):
    return {record.env.device for _, group in groups.values() for record in group}


def environment_section(all_groups, copy_summary):
    """Header listing every distinct environment, and what ran on an A100."""
    envs = {}
    for groups in all_groups.values():
        for _, group in groups.values():
            env = group[0].env
            envs[json.dumps(env.model_dump(), sort_keys=True)] = env
    lines = ["## Environment", ""]
    if not envs:
        lines += ["No benchmark records found.", ""]
    else:
        lines += ["| device | cc | driver | CUDA | torch | triton | git SHA | date |",
                  "|---|---|---|---|---|---|---|---|"]
        for env in envs.values():
            lines.append(f"| {env.device} | {env.cc} | {env.driver} | {env.cuda} | {env.torch} | "
                         f"{env.triton} | {env.git_sha} | {env.date} |")
        lines.append("")
    for suite, groups in all_groups.items():
        devices = _devices(groups)
        if not devices:
            lines.append(f"- `{suite}`: not run.")
        elif all(is_a100(device) for device in devices):
            lines.append(f"- `{suite}`: run on an A100 ({', '.join(sorted(devices))}).")
        else:
            lines.append(f"- `{suite}`: NOT an A100 run ({', '.join(sorted(devices))}); "
                         "criteria are not evaluated for it.")
    if copy_summary is None:
        lines.append("- `copy-probe`: not run.")
    return lines + [""]


def copy_section(copy_summary):
    """BW_copy, each implementation's bandwidth, and its ratio to the datasheet figure."""
    lines = ["## Memory-copy baseline (BW_copy)", ""]
    if copy_summary is None:
        return lines + ["Not measured.", ""]
    lines += ["| implementation | call | GB/s | ms p50 (launch range) | n |", "|---|---|---|---|---|"]
    for impl, row in copy_summary["per_impl"].items():
        lines.append(f"| {impl} | `{row['baseline_call']}` | {row['gbps']:.1f} | "
                     f"{row['ms_p50']:.4f} ({row['ms_min']:.4f}-{row['ms_max']:.4f}) | {row['n']} |")
    bw = copy_summary["bw_copy_gbps"]
    lines += ["", (f"BW_copy = **{bw:.1f} GB/s** ({copy_summary['best_impl']}), "
              f"{bw / DATASHEET_GBPS:.3f} of the {DATASHEET_GBPS:.0f} GB/s datasheet figure. "
              f"Buffer: {copy_summary['case']['buffer_bytes']} bytes, counted as one read plus one "
              "write."), ""]
    return lines


def row_criteria(groups, op):
    """Ratio >= 0.80 table on the pre-registered cases; returns (lines, number of failures)."""
    info = ROW_OPS[op]
    lines = [f"## Pre-registered criterion {info['criterion']}: {info['title']} ratio >= 0.80", "",
             "| N | dtype | M | GB/s | ratio to BW_copy | threshold | result | n |",
             "|---|---|---|---|---|---|---|---|"]
    failures = 0
    for case in [{**case, "selected": True} for case in suites.selected_rows()]:
        aggregate, group = groups.get(_key(op, "kernelforge", case), (None, []))
        prefix = f"| {case['N']} | {case['dtype']} | {case['M']} |"
        if aggregate is None:
            lines.append(f"{prefix} - | - | 0.80 | not measured | {len(group)} |")
            continue
        if not all(is_a100(record.env.device) for record in group):
            lines.append(f"{prefix} {aggregate.gbps:.1f} | - | 0.80 | not evaluated: "
                         f"{group[0].env.device} is not an A100 | {aggregate.n} |")
            continue
        passed = aggregate.ratio_to_copy >= RATIO_THRESHOLD
        failures += not passed
        lines.append(f"{prefix} {aggregate.gbps:.1f} | {aggregate.ratio_to_copy:.3f} | 0.80 | "
                     f"{'PASS' if passed else 'FAIL'} | {aggregate.n} |")
    return lines + [""], failures


def row_sweep(groups, op):
    """Full sweep with the PyTorch baseline measured the same way."""
    info = ROW_OPS[op]
    lines = [f"## {info['title']} sweep", "",
             (f"Effective bandwidth = {info['bytes']}, divided by the median time (median of three "
              "launch medians; launch range in parentheses). Selected cases are marked *. "
              f"Baseline call: `{info['call']}`."), "",
             (f"| N | dtype | M | KernelForge GB/s | ms p50 (range) | {info['torch']} GB/s | "
              "KF/torch | KF ratio to BW_copy | n |"), "|---|---|---|---|---|---|---|---|---|"]
    for case in suites.row_sweep():
        ours, _ = groups.get(_key(op, "kernelforge", case), (None, []))
        theirs, _ = groups.get(_key(op, "torch", case), (None, []))
        name = f"{case['N']}{'*' if case['selected'] else ''}"
        if ours is None or theirs is None:
            lines.append(f"| {name} | {case['dtype']} | {case['M']} | not measured | | | | | |")
            continue
        ratio = "-" if ours.ratio_to_copy is None else f"{ours.ratio_to_copy:.3f}"
        lines.append(f"| {name} | {case['dtype']} | {case['M']} | {ours.gbps:.1f} | "
                     f"{ours.ms_p50:.4f} ({ours.ms_min:.4f}-{ours.ms_max:.4f}) | {theirs.gbps:.1f} | "
                     f"{ours.gbps / theirs.gbps:.3f} | {ratio} | {ours.n} |")
    return lines + [""]


def accuracy_section(raw_dir, op):
    """Kernel vs reference error next to PyTorch's own error, from launch 0 inputs."""
    path = Path(raw_dir) / op / "accuracy.json"
    lines = [f"## Accuracy ({ROW_OPS[op]['title']}, benchmark inputs)", ""]
    if not path.exists():
        return lines + ["Not measured.", ""]
    lines += ["| N | dtype | KernelForge max abs err | torch max abs err | allowed | result |",
              "|---|---|---|---|---|---|"]
    for row in json.loads(path.read_text()):
        case = row["case"]
        lines.append(f"| {case['N']} | {case['dtype']} | {row['kernel_max_abs']:.3e} | "
                     f"{row['torch_max_abs']:.3e} | {row['allowed_max_abs']:.3e} | "
                     f"{'PASS' if row['passed'] else 'FAIL'} |")
    return lines + [""]


def memory_section(groups, op):
    """Peak allocation beyond the inputs for one call (includes the output tensor)."""
    lines = [f"## Peak memory ({ROW_OPS[op]['title']})", "",
             ("Delta of `max_memory_allocated()` over the allocated inputs for one call; the "
              "output tensor is included."), "",
             "| N | dtype | input bytes | KernelForge delta | torch delta |", "|---|---|---|---|---|"]
    for case in suites.row_sweep():
        _, ours = groups.get(_key(op, "kernelforge", case), (None, []))
        _, theirs = groups.get(_key(op, "torch", case), (None, []))
        if not ours or not theirs:
            continue
        lines.append(f"| {case['N']} | {case['dtype']} | "
                     f"{case['M'] * case['N'] * ELEM_SIZE[case['dtype']]} | "
                     f"{max(r.peak_mem_delta_bytes for r in ours)} | "
                     f"{max(r.peak_mem_delta_bytes for r in theirs)} |")
    return lines + [""]


def autotune_section(groups, op):
    """Winning configuration per case and how many candidates were pruned or invalid."""
    lines = [f"## Autotune ({ROW_OPS[op]['title']})", "",
             "First-call seconds include compilation and autotuning and are excluded from timing.",
             "", ("| N | dtype | winner (launch 0) | same winner in all launches | rejected configs |"
                  " first-call s (max) |"), "|---|---|---|---|---|---|"]
    for case in suites.row_sweep():
        _, group = groups.get(_key(op, "kernelforge", case), (None, []))
        if not group:
            continue
        winners = [record.autotune.best_config for record in group]
        statuses = {}
        for config in group[0].autotune.invalid_configs:
            statuses[config.get("status", "?")] = statuses.get(config.get("status", "?"), 0) + 1
        seconds = [r.autotune.tuning_seconds for r in group if r.autotune.tuning_seconds is not None]
        seconds_cell = f"{max(seconds):.1f}" if seconds else "-"
        lines.append(f"| {case['N']} | {case['dtype']} | `{json.dumps(winners[0])}` | "
                     f"{'yes' if all(w == winners[0] for w in winners) else 'no'} | "
                     f"{json.dumps(statuses) if statuses else 'none'} | {seconds_cell} |")
    return lines + [""]


def where_we_lose(groups, op):
    """Cases slower than torch, and sweep cases below the 0.80 ratio, listed separately."""
    info = ROW_OPS[op]
    slower, below = [], []
    for case in suites.row_sweep():
        ours, _ = groups.get(_key(op, "kernelforge", case), (None, []))
        theirs, _ = groups.get(_key(op, "torch", case), (None, []))
        if ours is None or theirs is None:
            continue
        name = f"N={case['N']} {case['dtype']}"
        if ours.gbps < theirs.gbps:
            slower.append(f"- {name}: {ours.gbps:.1f} GB/s vs torch {theirs.gbps:.1f} GB/s (n=3).")
        if ours.ratio_to_copy is not None and ours.ratio_to_copy < RATIO_THRESHOLD:
            below.append(f"- {name}{' (selected)' if case['selected'] else ''}: ratio "
                         f"{ours.ratio_to_copy:.3f}, {ours.gbps:.1f} GB/s (n=3).")
    lines = [f"### {info['title']} slower than {info['torch']}", ""]
    lines += slower or ["None of the measured cases."]
    lines += ["", f"### {info['title']} below the 0.80 bandwidth ratio", ""]
    return lines + (below or ["None of the measured cases."]) + [""]


PEAK_TFLOPS = 312.0
ATTENTION_RATIO = 0.50


def _shape(case):
    return f"{case['B']}x{case['H']}x{case['S']}x{case['D']}"


def _attention_pair(groups, case):
    ours, ours_group = groups.get(_key("causal_attention", "kernelforge", case), (None, []))
    flash, flash_group = groups.get(_key("causal_attention", "sdpa_flash", case), (None, []))
    return ours, flash, ours_group + flash_group


def attention_criteria(groups):
    """Criterion 4: KernelForge TFLOPS >= 0.50 x SDPA-FlashAttention TFLOPS on selected cases."""
    lines = ["## Pre-registered criterion 4: attention TFLOPS >= 0.50 x SDPA-FlashAttention", "",
             ("| B x H x S x D | dtype | KernelForge TFLOPS | SDPA-flash TFLOPS | ratio | threshold |"
             " result | n |"), "|---|---|---|---|---|---|---|---|"]
    failures = 0
    for case in [{**case, "selected": True} for case in suites.selected_attention()]:
        ours, flash, group = _attention_pair(groups, case)
        prefix = f"| {_shape(case)} | {case['dtype']} |"
        if ours is None or flash is None:
            lines.append(f"{prefix} - | - | - | 0.50 | not measured | {len(group)} |")
            continue
        if not all(is_a100(record.env.device) for record in group):
            lines.append(f"{prefix} {ours.tflops:.1f} | {flash.tflops:.1f} | - | 0.50 | "
                         f"not evaluated: {group[0].env.device} is not an A100 | 3 |")
            continue
        ratio = ours.tflops / flash.tflops
        passed = ratio >= ATTENTION_RATIO
        failures += not passed
        lines.append(f"{prefix} {ours.tflops:.1f} | {flash.tflops:.1f} | {ratio:.3f} | 0.50 | "
                     f"{'PASS' if passed else 'FAIL'} | 3 |")
    return lines + [""], failures


def attention_sweep(groups):
    """TFLOPS (2*B*H*S^2*D) for KernelForge, SDPA-flash, and SDPA's default backend."""
    lines = ["## Causal attention sweep", "",
             ("TFLOPS = 2*B*H*S^2*D / median time (median of three launch medians; launch range in"
              " parentheses), with % of the 312 TFLOPS datasheet peak. Selected cases are marked *."),
             "", ("| B x H x S x D | dtype | KernelForge TFLOPS | ms p50 (range) | % of peak | "
             "SDPA-flash TFLOPS | KF/flash | SDPA default TFLOPS | n |"),
             "|---|---|---|---|---|---|---|---|---|"]
    defaults = set()
    for case in suites.attention_sweep():
        ours, flash, _ = _attention_pair(groups, case)
        default, default_group = groups.get(_key("causal_attention", "sdpa_default", case),
                                            (None, []))
        # Older records embedded the profiled kernel in the call; newer ones keep it in autotune.
        defaults.update(record.autotune.best_config.get("profiled_kernel", record.baseline_call)
                        for record in default_group)
        name = f"{_shape(case)}{'*' if case['selected'] else ''}"
        if ours is None or flash is None:
            lines.append(f"| {name} | {case['dtype']} | not measured | | | | | | |")
            continue
        default_cell = (f"{default.tflops:.1f}" if default is not None
                        else "not aggregated" if default_group else "-")
        lines.append(f"| {name} | {case['dtype']} | {ours.tflops:.1f} | {ours.ms_p50:.3f} "
                     f"({ours.ms_min:.3f}-{ours.ms_max:.3f}) | "
                     f"{100 * ours.tflops / PEAK_TFLOPS:.1f}% | {flash.tflops:.1f} | "
                     f"{ours.tflops / flash.tflops:.3f} | {default_cell} | 3 |")
    lines += ["", ("Default-backend kernels observed with torch.profiler (\"unknown\" = the "
                   "profiler captured no CUDA event; such launches cannot be aggregated with "
                   "launches that named the kernel):"), ""]
    lines += [f"- `{call}`" for call in sorted(defaults)] or ["- none measured"]
    return lines + [""]


def attention_accuracy_section(raw_dir):
    path = Path(raw_dir) / "causal_attention" / "accuracy.json"
    lines = ["## Accuracy (causal attention, benchmark inputs)", ""]
    if not path.exists():
        return lines + ["Not measured.", ""]
    lines += [("Measured on batch 0, heads 0-1 of each benchmark input (the fp32 reference "
              "materializes S x S). Baseline: SDPA forced to FlashAttention, same dtype."), "",
              ("| B x H x S x D | dtype | KernelForge max abs err | SDPA-flash max abs err | allowed "
              "| result |"), "|---|---|---|---|---|---|"]
    for row in json.loads(path.read_text()):
        lines.append(f"| {_shape(row['case'])} | {row['case']['dtype']} | "
                     f"{row['kernel_max_abs']:.3e} | {row['torch_max_abs']:.3e} | "
                     f"{row['allowed_max_abs']:.3e} | {'PASS' if row['passed'] else 'FAIL'} |")
    return lines + [""]


def attention_memory_section(raw_dir, plots_dir):
    """Peak memory vs S: O(S) for the fused kernels, O(S^2) for the math backend."""
    path = Path(raw_dir) / "causal_attention" / "memory.json"
    lines = ["## Attention memory scaling", ""]
    if not path.exists():
        return lines + ["Not measured.", ""]
    rows = json.loads(path.read_text())
    impls = ("kernelforge", "sdpa_flash", "sdpa_math")
    table = {}
    for row in rows:
        table.setdefault(row["case"]["S"], {})[row["impl"]] = row["peak_mem_delta_bytes"]
    lines += [("Peak `max_memory_allocated()` delta over q, k, v for one call; B=1, H=32, D=64, "
              "fp16; includes the output."), "",
              "| S | KernelForge MiB | SDPA-flash MiB | SDPA-math MiB |", "|---|---|---|---|"]
    for seq in sorted(table):
        cells = [_memory_cell(table[seq], impl) for impl in impls]
        lines.append(f"| {seq} | " + " | ".join(cells) + " |")
    plot = _plot_memory(table, plots_dir) if plots_dir else None
    if plot is not None:
        lines += ["", f"![Attention peak memory vs S](plots/{plot.name})"]
    return lines + [""]


def _memory_cell(row, impl):
    """MiB, OOM for a recorded out-of-memory result, or - when the impl was not measured."""
    if impl not in row:
        return "-"
    return "OOM" if row[impl] is None else f"{row[impl] / 2**20:.1f}"


def _plot_memory(table, plots_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axis = plt.subplots(figsize=(6, 3.8))
    for impl, color, label in (("kernelforge", KERNELFORGE_COLOR, "KernelForge"),
                               ("sdpa_flash", TORCH_COLOR, "SDPA flash"),
                               ("sdpa_math", "#1baf7a", "SDPA math")):
        points = [(seq, values[impl] / 2**20) for seq, values in sorted(table.items())
                  if values.get(impl)]
        if points:
            xs, ys = zip(*points)
            axis.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("S (sequence length)")
    axis.set_ylabel("peak memory delta (MiB)")
    axis.set_title("Attention peak memory vs S (B=1, H=32, D=64, fp16)", fontsize=11)
    axis.grid(True, color="#e5e4e0", linewidth=0.6)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    path = Path(plots_dir) / "attention_memory.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def attention_autotune_section(groups):
    lines = ["## Autotune (causal attention)", "",
             ("Space: BLOCK_M, BLOCK_N in {32, 64, 128} with BLOCK_N <= BLOCK_M, num_warps {4, 8}, "
             "num_stages {2, 3, 4}. Invalid = failed to compile or launch (e.g. shared memory)."), "",
             ("| B x H x S x D | dtype | winner (launch 0) | same winner in all launches | "
             "invalid configs | first-call s (max) |"), "|---|---|---|---|---|---|"]
    for case in suites.attention_sweep():
        _, group = groups.get(_key("causal_attention", "kernelforge", case), (None, []))
        if not group:
            continue
        winners = [record.autotune.best_config for record in group]
        invalid = [c for c in group[0].autotune.invalid_configs if c.get("status") == "invalid"]
        names = ", ".join(f"{c['block_m']}/{c['block_n']}/w{c['num_warps']}/s{c['num_stages']}"
                          for c in invalid) or "none"
        seconds = [r.autotune.tuning_seconds for r in group if r.autotune.tuning_seconds is not None]
        seconds_cell = f"{max(seconds):.1f}" if seconds else "-"
        lines.append(f"| {_shape(case)} | {case['dtype']} | `{json.dumps(winners[0])}` | "
                     f"{'yes' if all(w == winners[0] for w in winners) else 'no'} | {names} | "
                     f"{seconds_cell} |")
    return lines + [""]


def attention_losses(groups):
    slower, below = [], []
    for case in suites.attention_sweep():
        ours, flash, _ = _attention_pair(groups, case)
        if ours is None or flash is None:
            continue
        ratio = ours.tflops / flash.tflops
        label = f"- {_shape(case)} {case['dtype']}{' (selected)' if case['selected'] else ''}"
        if ratio < 1:
            slower.append(f"{label}: {ours.tflops:.1f} vs {flash.tflops:.1f} TFLOPS "
                          f"({ratio:.3f}, n=3).")
        if ratio < ATTENTION_RATIO:
            below.append(f"{label}: ratio {ratio:.3f}.")
    lines = ["### Attention slower than SDPA-FlashAttention", ""]
    lines += slower or ["None of the measured cases."]
    lines += ["", "### Attention below 0.50 x SDPA-FlashAttention", ""]
    return lines + (below or ["None of the measured cases."]) + [""]


def _load_profiling(raw_dir, name):
    path = Path(raw_dir) / "profiling" / f"{name}.json"
    return json.loads(path.read_text())["result"] if path.exists() else None


def profiling_section(raw_dir, copy_summary):
    """Numbers from scripts/profile_kernels.py; interpretation lives in docs/PROFILING.md."""
    lines = ["## Profiling findings", "", ("Narrative and analysis: docs/PROFILING.md. Experiment "
             "timings are single-process do_bench medians (controlled comparisons, not criteria)."),
             ""]
    ncu = _load_profiling(raw_dir, "ncu_probe")
    if ncu is None:
        return lines + ["Not measured.", ""]
    denied = "ERR_NVGPUCTRPERM" in ncu.get("stdout_tail", "") + ncu.get("stderr_tail", "")
    lines += [(f"- Nsight Compute: return code {ncu.get('returncode')}; "
              f"{'ERR_NVGPUCTRPERM (counter access denied)' if denied else 'see raw output'}. "
              "No hardware-counter metrics were collected."), ""]
    if copy_summary is not None:
        ridge = PEAK_TFLOPS * 1e12 / (copy_summary["bw_copy_gbps"] * 1e9)
        lines += [(f"- Roofline ridge point: {PEAK_TFLOPS:.0f} TFLOPS / BW_copy "
                  f"{copy_summary['bw_copy_gbps']:.0f} GB/s = {ridge:.0f} FLOP/byte. Softmax and "
                  "LayerNorm (< 3 FLOP/byte) are memory bound; causal attention (S/4 FLOP/byte, "
                  "512-2048 for the selected S) is compute bound."), ""]
    traces = _load_profiling(raw_dir, "traces") or []
    if traces:
        lines += ["### Kernel traces (torch.profiler)", "",
                  "| call | kernel | us | grid | block | regs/thread | shared B | est. occupancy % |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {t['op']} | `{t['name'][:60]}` | {t['us']:.0f} | {t['grid']} | {t['block']} | "
                  f"{t['registers_per_thread']} | {t['shared_memory']} | {t['occupancy_pct']} |"
                  for t in traces]
        lines.append("")
    alignment = _load_profiling(raw_dir, "alignment") or []
    if alignment:
        lines += ["### Experiment: natural vs 64-aligned row stride (same N)", "",
                  "| op | N | dtype | natural ms | aligned ms |", "|---|---|---|---|---|"]
        cells = {}
        for row in alignment:
            cells.setdefault((row["op"], row["N"], row["dtype"]), {})[row["layout"]] = row["ms_p50"]
        lines += [f"| {op} | {n} | {dtype} | {v['natural']:.3f} | {v['aligned']:.3f} |"
                  for (op, n, dtype), v in cells.items()]
        lines.append("")
    warps = _load_profiling(raw_dir, "warps") or []
    if warps:
        lines += ["### Experiment: LayerNorm single path, forced num_warps (fp16, BLOCK = N)", "",
                  "| N | num_warps | ms | GB/s |", "|---|---|---|---|"]
        lines += [f"| {w['N']} | {w['num_warps']} | {w['ms_p50']:.3f} | {w['gbps']:.0f} |"
                  for w in warps]
        lines.append("")
    tiles = _load_profiling(raw_dir, "attention_tiles") or []
    if tiles:
        lines += ["### Experiment: attention tiles forced (4x32x4096, bf16)", "",
                  "| D | BLOCK_M | BLOCK_N | warps | stages | TFLOPS |", "|---|---|---|---|---|---|"]
        lines += [f"| {t['D']} | {t['BLOCK_M']} | {t['BLOCK_N']} | {t['num_warps']} | "
                  f"{t['num_stages']} | {t['tflops']:.1f} |" if "tflops" in t else
                  f"| {t['D']} | {t['BLOCK_M']} | {t['BLOCK_N']} | {t['num_warps']} | "
                  f"{t['num_stages']} | {t.get('error')} |"
                  for t in sorted(tiles, key=lambda t: (t["D"], -t.get("tflops", 0)))]
        lines.append("")
    compiled = _load_profiling(raw_dir, "compiled") or []
    if compiled:
        lines += ["### Compiled variants (Triton metadata)", "",
                  "| kernel | variants | with spills | max spills | max shared B |",
                  "|---|---|---|---|---|"]
        kinds = {}
        for c in compiled:
            kinds.setdefault(c["kernel"], []).append(c)
        for kind, rows in kinds.items():
            spills = [c["spills"] or 0 for c in rows]
            lines.append(f"| {kind} | {len(rows)} | {sum(1 for x in spills if x)} | {max(spills)} | "
                         f"{max(c['shared_bytes'] or 0 for c in rows)} |")
        lines.append("")
    return lines


def telemetry_section(all_groups):
    """Start/end nvidia-smi samples per launch, flagging lowered clocks or throttling."""
    lines = ["## GPU telemetry per launch", "",
             ("Flagged when the end SM clock is below 97% of max or a non-idle throttle reason "
              "is active."), "", ("| suite | launch | start SM/mem MHz, W, C | end SM/mem MHz, W, C |"
                                  " end throttle reasons | flag |"), "|---|---|---|---|---|---|"]
    seen = False
    for suite, groups in all_groups.items():
        launches = {}
        for _, group in groups.values():
            for record in group:
                launches.setdefault(record.launch_index, record.gpu_telemetry)
        for index, telemetry in sorted(launches.items()):
            seen = True
            sample = telemetry.model_dump()
            flag = "CLOCK DROP" if gpu_clock_drop(sample) else "ok"
            lines.append(f"| {suite} | {index} | {_describe(sample['start'])} | "
                         f"{_describe(sample['end'])} | "
                         f"{sample['end'].get('clocks_throttle_reasons.active', '?')} | {flag} |")
    return (lines if seen else lines[:2] + ["No telemetry recorded."]) + [""]


def _describe(point):
    return (f"{point.get('clocks.sm', '?')}/{point.get('clocks.mem', '?')}, "
            f"{point.get('power.draw', '?')}, {point.get('temperature.gpu', '?')}")


def gpu_clock_drop(telemetry):
    """Shared clock-drop rule (defined with the runner that samples telemetry)."""
    from kernelforge.bench.gpu_runner import clock_drop

    return clock_drop(telemetry)


def plot_row_op(groups, copy_summary, plots_dir, op):
    """Small multiples (one panel per dtype) of GB/s vs power-of-two N; None without data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    info = ROW_OPS[op]
    widths = sorted(n for n in suites.SWEEP_WIDTHS if n & (n - 1) == 0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    plotted = False
    for axis, dtype in zip(axes, DTYPE_ORDER):
        for impl, color, label in (("kernelforge", KERNELFORGE_COLOR, "KernelForge"),
                                   ("torch", TORCH_COLOR, info["torch"])):
            points = []
            for case in suites.row_sweep():
                aggregate, _ = groups.get(_key(op, impl, case), (None, []))
                if case["dtype"] == dtype and case["N"] in widths and aggregate is not None:
                    points.append((case["N"], aggregate.gbps))
            if points:
                plotted = True
                xs, ys = zip(*sorted(points))
                axis.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
        if copy_summary is not None:
            axis.axhline(copy_summary["bw_copy_gbps"], color=REFERENCE_COLOR, linewidth=1,
                         linestyle="--", label="BW_copy")
        axis.set_xscale("log", base=2)
        axis.set_title(dtype, fontsize=11)
        axis.set_xlabel("N (columns)")
        axis.grid(True, color="#e5e4e0", linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("effective GB/s")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    if not plotted:
        plt.close(figure)
        return None
    figure.suptitle(f"{info['title']} effective bandwidth, median of 3 launches", fontsize=12)
    figure.tight_layout()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    path = Path(plots_dir) / f"{op}_bandwidth.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render(raw_dir, plots_dir=None):
    """Return (markdown, number of failed criteria) built only from files under raw_dir."""
    raw_dir = Path(raw_dir)
    copy_path = raw_dir / "copy_probe.json"
    copy_summary = json.loads(copy_path.read_text()) if copy_path.exists() else None
    groups = {op: load_groups(raw_dir, op) for op in ROW_OPS}
    attention = load_groups(raw_dir, "causal_attention")
    lines = ["# Performance report", "",
             "Generated by `kernelforge report` from `reports/raw/`; do not edit by hand.", ""]
    lines += environment_section({**groups, "causal_attention": attention}, copy_summary)
    failures = 0
    for op in ROW_OPS:
        criteria, failed = row_criteria(groups[op], op)
        lines += criteria
        failures += failed
    criteria, failed = attention_criteria(attention)
    lines += criteria
    failures += failed
    lines += copy_section(copy_summary)
    for op in ROW_OPS:
        lines += row_sweep(groups[op], op)
        plot = plot_row_op(groups[op], copy_summary, plots_dir, op) if plots_dir else None
        if plot is not None:
            lines += [f"![{ROW_OPS[op]['title']} effective bandwidth](plots/{plot.name})", ""]
        lines += accuracy_section(raw_dir, op) + memory_section(groups[op], op)
        lines += autotune_section(groups[op], op)
    lines += attention_sweep(attention) + attention_accuracy_section(raw_dir)
    lines += attention_memory_section(raw_dir, plots_dir) + attention_autotune_section(attention)
    lines += profiling_section(raw_dir, copy_summary)
    lines += ["## Where we lose and why", ""]
    for op in ROW_OPS:
        lines += where_we_lose(groups[op], op)
    lines += attention_losses(attention)
    lines += [("Evidence-backed explanations, and the cases still not understood, are in "
              "docs/PROFILING.md (section \"Where we lose\")."), ""]
    lines += telemetry_section({"copy": load_groups(raw_dir, "copy"), **groups,
                                "causal_attention": attention})
    lines += ["## Reproduction", "", "```sh", "bash scripts/gpu_session.sh setup",
              "bash scripts/gpu_session.sh bench all",
              "kernelforge report --raw reports/raw --out reports", "```", ""]
    return "\n".join(lines), failures


def write_report(raw_dir, out_dir):
    """Write performance_report.md (and plots) and return the number of failed criteria."""
    out_dir = Path(out_dir)
    markdown, failures = render(raw_dir, out_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "performance_report.md").write_text(markdown)
    return failures
