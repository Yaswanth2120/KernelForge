"""Argparse wiring only; logic lives in env, suites, gpu_runner, copy_probe, and report.

Exit codes: 0 ok; 1 test/criterion failure; 2 usage error; 3 environment check failed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from kernelforge.bench.suites import check_prereg
from kernelforge.env import doctor, timing_source

# Test-file stem per op: tests/cpu/test_<stem>_interface.py and tests/gpu/test_<stem>.py.
OP_TEST_STEMS = {"softmax": "softmax", "layer_norm": "layernorm", "causal_attention": "attention"}


def _parser():
    parser = argparse.ArgumentParser(prog="kernelforge")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor")
    diagnostic.add_argument("--probe-ncu", action="store_true")
    diagnostic.add_argument("--timing-source", action="store_true")
    prereg = commands.add_parser("check-prereg")
    prereg.add_argument("--path", type=Path, default=Path("PREREG.md"))
    bench = commands.add_parser("bench")
    bench.add_argument("--suite", required=True)
    bench.add_argument("--launches", type=int, default=3)
    bench.add_argument("--out", type=Path, default=Path("reports/raw"))
    copy = commands.add_parser("copy-probe")
    copy.add_argument("--out", type=Path, default=Path("reports/raw/copy_probe.json"))
    tests = commands.add_parser("test")
    tests.add_argument("--gpu", action="store_true")
    tests.add_argument("--op", choices=sorted(OP_TEST_STEMS))
    report = commands.add_parser("report")
    report.add_argument("--raw", type=Path, default=Path("reports/raw"))
    report.add_argument("--out", type=Path, default=Path("reports"))
    return parser


def pytest_args(gpu, op):
    """The pytest command line for the CPU suite or the (non-slow) GPU suite, optionally one op."""
    if gpu:
        marker, path = "gpu and not slow", "tests/gpu"
        if op:
            path = f"tests/gpu/test_{OP_TEST_STEMS[op]}.py"
    else:
        marker, path = "not gpu and not slow", "tests/cpu"
        if op:
            path = f"tests/cpu/test_{OP_TEST_STEMS[op]}_interface.py"
    return [sys.executable, "-m", "pytest", "-m", marker, path]


def _test(args):
    # pytest exits 1 on failing tests and 2-5 on usage or collection problems.
    returncode = subprocess.run(pytest_args(args.gpu, args.op), check=False).returncode
    return returncode if returncode in (0, 1) else 2


def _gpu_ready():
    from kernelforge import is_triton_available

    if is_triton_available():
        return True
    print("Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton.")
    return False


def _bench(args):
    from kernelforge.bench.gpu_runner import (
        LAUNCHES,
        RATIO_SUITES,
        SUITES,
        copy_baseline,
        run_suite,
    )

    if args.suite not in SUITES or args.suite == "copy":
        print(f"Unknown benchmark suite {args.suite!r}; use one of {SUITES[1:]} "
              "(the copy baseline is `kernelforge copy-probe`).")
        return 2
    if args.launches != LAUNCHES:
        print("The registered protocol requires exactly three launches.")
        return 2
    if not _gpu_ready():
        return 3
    if args.suite in RATIO_SUITES:
        try:
            copy_baseline(args.out)
        except FileNotFoundError as exc:
            print(str(exc))
            return 3
    run_suite(args.suite, args.out, args.launches)
    return 0


def _copy_probe(args):
    from kernelforge.bench import copy_probe

    if not _gpu_ready():
        return 3
    summary = copy_probe.run(args.out)
    print(f"BW_copy = {summary['bw_copy_gbps']:.1f} GB/s ({summary['best_impl']}, n=3)")
    return 0


def _report(args):
    from kernelforge.report import write_report

    failures = write_report(args.raw, args.out)
    print(f"Wrote {args.out / 'performance_report.md'}; failed criteria cases: {failures}")
    return 1 if failures else 0


def main(argv=None):
    """Parse commands and delegate to CPU-safe application functions."""
    args = _parser().parse_args(argv)
    if args.command == "check-prereg":
        try:
            check_prereg(args.path)
        except (OSError, ValueError) as exc:
            print(str(exc))
            return 1
        print("Pre-registration and selected suites match.")
        return 0
    if args.command == "bench":
        return _bench(args)
    if args.command == "copy-probe":
        return _copy_probe(args)
    if args.command == "report":
        return _report(args)
    if args.command == "test":
        return _test(args)
    result = doctor(args.probe_ncu)
    print(json.dumps(result, indent=2))
    if args.timing_source:
        try:
            print(timing_source())
        except RuntimeError as exc:
            print(str(exc))
            return 3
    return 0 if result["gpu_available"] and result["torch_pin_matches"] else 3
