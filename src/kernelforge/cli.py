import argparse
import json
from pathlib import Path

from kernelforge.bench.suites import check_prereg
from kernelforge.env import doctor, timing_source


def main(argv=None):
    """Parse commands and delegate to CPU-safe application functions."""
    parser = argparse.ArgumentParser(prog="kernelforge")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor")
    diagnostic.add_argument("--probe-ncu", action="store_true")
    diagnostic.add_argument("--timing-source", action="store_true")
    prereg = commands.add_parser("check-prereg")
    prereg.add_argument("--path", type=Path, default=Path("PREREG.md"))
    args = parser.parse_args(argv)
    if args.command == "check-prereg":
        try:
            check_prereg(args.path)
        except (OSError, ValueError) as exc:
            print(str(exc))
            return 1
        print("Pre-registration and selected suites match.")
        return 0
    result = doctor(args.probe_ncu)
    print(json.dumps(result, indent=2))
    if args.timing_source:
        try:
            print(timing_source())
        except RuntimeError as exc:
            print(str(exc))
            return 3
    return 0 if result["gpu_available"] and result["torch_pin_matches"] else 3
