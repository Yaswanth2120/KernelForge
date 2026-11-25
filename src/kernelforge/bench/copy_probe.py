"""Memory-copy baseline BW_copy: max over implementations of the median of three launches."""
import json
from pathlib import Path

from kernelforge.bench.harness import group_launches, load_launch_records


def summarize(raw_dir):
    """Aggregate the copy suite's launches and select the best implementation's bandwidth."""
    groups = group_launches(load_launch_records(Path(raw_dir) / "copy"))
    per_impl = {}
    for (_, impl, _), (aggregate, group) in groups.items():
        if aggregate is None:
            raise ValueError(f"copy impl {impl!r} has {len(group)} launches; exactly 3 required")
        per_impl[impl] = {"gbps": aggregate.gbps, "ms_p50": aggregate.ms_p50,
                          "ms_min": aggregate.ms_min, "ms_max": aggregate.ms_max, "n": aggregate.n,
                          "baseline_call": group[0].baseline_call}
    if not per_impl:
        raise ValueError(f"No copy records under {raw_dir}")
    best = max(per_impl, key=lambda impl: per_impl[impl]["gbps"])
    first = next(iter(groups.values()))[1][0]
    return {"schema": "kernelforge.copy_probe.v1", "bw_copy_gbps": per_impl[best]["gbps"],
            "best_impl": best, "per_impl": per_impl, "case": first.case,
            "env": first.env.model_dump()}


def run(out_path):
    """Run the three copy launches on the GPU, then write the BW_copy summary JSON."""
    from kernelforge.bench.gpu_runner import run_suite

    out_path = Path(out_path)
    run_suite("copy", out_path.parent)
    summary = summarize(out_path.parent)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
