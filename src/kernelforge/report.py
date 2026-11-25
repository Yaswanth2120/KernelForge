"""Generate reports/performance_report.md from committed raw JSON only (no hand-edited numbers).

Rules enforced here: a ratio is printed only from a three-launch aggregate (n is shown), and
criteria are evaluated only when every contributing record's device name contains "A100".
"""
import json
from pathlib import Path

from kernelforge.bench import suites
from kernelforge.bench.harness import group_launches, load_launch_records

DATASHEET_GBPS = 2039.0
SOFTMAX_RATIO_THRESHOLD = 0.80
DTYPE_ORDER = ("fp16", "bf16", "fp32")
KERNELFORGE_COLOR, TORCH_COLOR, REFERENCE_COLOR = "#2a78d6", "#eb6834", "#898781"


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


def softmax_criteria(groups):
    """Criterion 2 table on the pre-registered cases; returns (lines, number of failures)."""
    lines = ["## Pre-registered criterion 2: softmax ratio >= 0.80", "",
             "| N | dtype | M | GB/s | ratio to BW_copy | threshold | result | n |",
             "|---|---|---|---|---|---|---|---|"]
    failures = 0
    selected = [{**case, "selected": True} for case in suites.selected_rows()]
    for case in selected:
        aggregate, group = groups.get(_key("softmax", "kernelforge", case), (None, []))
        prefix = f"| {case['N']} | {case['dtype']} | {case['M']} |"
        if aggregate is None:
            lines.append(f"{prefix} - | - | 0.80 | not measured | {len(group)} |")
            continue
        if not all(is_a100(record.env.device) for record in group):
            lines.append(f"{prefix} {aggregate.gbps:.1f} | - | 0.80 | not evaluated: "
                         f"{group[0].env.device} is not an A100 | {aggregate.n} |")
            continue
        passed = aggregate.ratio_to_copy >= SOFTMAX_RATIO_THRESHOLD
        failures += not passed
        lines.append(f"{prefix} {aggregate.gbps:.1f} | {aggregate.ratio_to_copy:.3f} | 0.80 | "
                     f"{'PASS' if passed else 'FAIL'} | {aggregate.n} |")
    return lines + [""], failures


def softmax_sweep(groups):
    """Full sweep with the torch.softmax baseline measured the same way."""
    lines = ["## Softmax sweep", "",
             ("Effective bandwidth = 2*M*N*elem_size / median time (median of three launch medians;"
             " launch range in parentheses). Selected cases are marked *. Baseline call: "
             "`torch.softmax(x, dim=-1)`."), "",
             ("| N | dtype | M | KernelForge GB/s | ms p50 (range) | torch GB/s | KF/torch | "
             "KF ratio to BW_copy | n |"), "|---|---|---|---|---|---|---|---|---|"]
    for case in suites.softmax_sweep():
        ours, _ = groups.get(_key("softmax", "kernelforge", case), (None, []))
        theirs, _ = groups.get(_key("softmax", "torch", case), (None, []))
        name = f"{case['N']}{'*' if case['selected'] else ''}"
        if ours is None or theirs is None:
            lines.append(f"| {name} | {case['dtype']} | {case['M']} | not measured | | | | | |")
            continue
        ratio = "-" if ours.ratio_to_copy is None else f"{ours.ratio_to_copy:.3f}"
        lines.append(f"| {name} | {case['dtype']} | {case['M']} | {ours.gbps:.1f} | "
                     f"{ours.ms_p50:.4f} ({ours.ms_min:.4f}-{ours.ms_max:.4f}) | {theirs.gbps:.1f} | "
                     f"{ours.gbps / theirs.gbps:.3f} | {ratio} | {ours.n} |")
    return lines + [""]


def accuracy_section(raw_dir):
    """Kernel vs reference error next to PyTorch's own error, from launch 0 inputs."""
    path = Path(raw_dir) / "softmax" / "accuracy.json"
    lines = ["## Accuracy (softmax, benchmark inputs)", ""]
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


def memory_section(groups):
    """Peak allocation beyond the inputs for one call (includes the output tensor)."""
    lines = ["## Peak memory (softmax)", "",
             ("Delta of `max_memory_allocated()` over the allocated inputs for one call; the "
             "output tensor is included."), "",
             "| N | dtype | input bytes | KernelForge delta | torch delta |", "|---|---|---|---|---|"]
    for case in suites.softmax_sweep():
        _ours, ours_group = groups.get(_key("softmax", "kernelforge", case), (None, []))
        _theirs, theirs_group = groups.get(_key("softmax", "torch", case), (None, []))
        if not ours_group or not theirs_group:
            continue
        size = case["M"] * case["N"] * {"fp16": 2, "bf16": 2, "fp32": 4}[case["dtype"]]
        lines.append(f"| {case['N']} | {case['dtype']} | {size} | "
                     f"{max(r.peak_mem_delta_bytes for r in ours_group)} | "
                     f"{max(r.peak_mem_delta_bytes for r in theirs_group)} |")
    return lines + [""]


def autotune_section(groups):
    """Winning configuration per case and how many candidates were pruned or invalid."""
    lines = ["## Autotune (softmax)", "",
             "First-call seconds include compilation and autotuning and are excluded from timing.",
             "", ("| N | dtype | winner (launch 0) | same winner in all launches | rejected configs |"
             " first-call s (max) |"), "|---|---|---|---|---|---|"]
    for case in suites.softmax_sweep():
        _, group = groups.get(_key("softmax", "kernelforge", case), (None, []))
        if not group:
            continue
        winners = [record.autotune.best_config for record in group]
        rejected = group[0].autotune.invalid_configs
        statuses = {}
        for config in rejected:
            statuses[config.get("status", "?")] = statuses.get(config.get("status", "?"), 0) + 1
        seconds = [r.autotune.tuning_seconds for r in group if r.autotune.tuning_seconds is not None]
        seconds_cell = f"{max(seconds):.1f}" if seconds else "-"
        lines.append(f"| {case['N']} | {case['dtype']} | `{json.dumps(winners[0])}` | "
                     f"{'yes' if all(w == winners[0] for w in winners) else 'no'} | "
                     f"{json.dumps(statuses) if statuses else 'none'} | {seconds_cell} |")
    return lines + [""]


def where_we_lose(groups):
    """Cases slower than torch, and sweep cases below the 0.80 ratio, listed separately."""
    slower, below = [], []
    for case in suites.softmax_sweep():
        ours, _ = groups.get(_key("softmax", "kernelforge", case), (None, []))
        theirs, _ = groups.get(_key("softmax", "torch", case), (None, []))
        if ours is None or theirs is None:
            continue
        name = f"N={case['N']} {case['dtype']}"
        if ours.gbps < theirs.gbps:
            slower.append(f"- {name}: {ours.gbps:.1f} GB/s vs torch {theirs.gbps:.1f} GB/s (n=3).")
        if ours.ratio_to_copy is not None and ours.ratio_to_copy < SOFTMAX_RATIO_THRESHOLD:
            below.append(f"- {name}{' (selected)' if case['selected'] else ''}: ratio "
                         f"{ours.ratio_to_copy:.3f}, {ours.gbps:.1f} GB/s (n=3).")
    lines = ["## Where we lose and why", "", "### Softmax slower than torch.softmax", ""]
    lines += slower or ["None of the measured cases."]
    lines += ["", "### Softmax below the 0.80 bandwidth ratio", ""]
    lines += below or ["None of the measured cases."]
    lines += ["", ("Explanations require profiling evidence; see docs/PROFILING.md (K7). Until "
              "then the cause is recorded as not yet understood."), ""]
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

            def describe(point):
                return (f"{point.get('clocks.sm', '?')}/{point.get('clocks.mem', '?')}, "
                        f"{point.get('power.draw', '?')}, {point.get('temperature.gpu', '?')}")

            flag = "CLOCK DROP" if gpu_clock_drop(sample) else "ok"
            lines.append(f"| {suite} | {index} | {describe(sample['start'])} | "
                         f"{describe(sample['end'])} | "
                         f"{sample['end'].get('clocks_throttle_reasons.active', '?')} | {flag} |")
    return (lines if seen else lines[:2] + ["No telemetry recorded."]) + [""]


def gpu_clock_drop(telemetry):
    """Shared clock-drop rule (defined with the runner that samples telemetry)."""
    from kernelforge.bench.gpu_runner import clock_drop

    return clock_drop(telemetry)


def plot_softmax(groups, copy_summary, plots_dir):
    """Small multiples (one panel per dtype) of GB/s vs power-of-two N; None without data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    widths = sorted(n for n in suites.SWEEP_WIDTHS if n & (n - 1) == 0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    plotted = False
    for axis, dtype in zip(axes, DTYPE_ORDER):
        for impl, color, label in (("kernelforge", KERNELFORGE_COLOR, "KernelForge"),
                                   ("torch", TORCH_COLOR, "torch.softmax")):
            points = []
            for case in suites.softmax_sweep():
                aggregate, _ = groups.get(_key("softmax", impl, case), (None, []))
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
    figure.suptitle("Softmax effective bandwidth, median of 3 launches", fontsize=12)
    figure.tight_layout()
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    path = Path(plots_dir) / "softmax_bandwidth.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render(raw_dir, plots_dir=None):
    """Return (markdown, number of failed criteria) built only from files under raw_dir."""
    raw_dir = Path(raw_dir)
    copy_path = raw_dir / "copy_probe.json"
    copy_summary = json.loads(copy_path.read_text()) if copy_path.exists() else None
    softmax_groups = load_groups(raw_dir, "softmax")
    lines = ["# Performance report", "",
             "Generated by `kernelforge report` from `reports/raw/`; do not edit by hand.", ""]
    lines += environment_section({"softmax": softmax_groups}, copy_summary)
    criteria, failures = softmax_criteria(softmax_groups)
    lines += criteria + copy_section(copy_summary) + softmax_sweep(softmax_groups)
    if plots_dir is not None:
        plot = plot_softmax(softmax_groups, copy_summary, plots_dir)
        if plot is not None:
            lines += [f"![Softmax effective bandwidth](plots/{plot.name})", ""]
    lines += accuracy_section(raw_dir) + memory_section(softmax_groups)
    lines += autotune_section(softmax_groups) + where_we_lose(softmax_groups)
    lines += telemetry_section({"copy": load_groups(raw_dir, "copy"), "softmax": softmax_groups})
    lines += ["## Not yet measured", "", ("LayerNorm (K4), causal attention (K6), and profiling "
              "(K7) sections are added when their measurements exist."), "",
              "## Reproduction", "", "```sh", "bash scripts/gpu_session.sh setup",
              "bash scripts/gpu_session.sh bench softmax", ("kernelforge report --raw reports/raw "
              "--out reports"), "```", ""]
    return "\n".join(lines), failures


def write_report(raw_dir, out_dir):
    """Write performance_report.md (and plots) and return the number of failed criteria."""
    out_dir = Path(out_dir)
    markdown, failures = render(raw_dir, out_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "performance_report.md").write_text(markdown)
    return failures
