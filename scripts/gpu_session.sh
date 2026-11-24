#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  setup)
    python -m pip install 'torch==2.8.0'
    python -m pip install -e '.[dev,report]'
    mkdir -p reports/raw
    kernelforge doctor --probe-ncu --timing-source | tee reports/raw/doctor.txt
    ;;
  test)
    if ! find tests/gpu -name 'test_*.py' | read -r _; then
      echo 'No GPU tests available yet.' >&2; exit 1
    fi
    python -m pytest tests/gpu -m gpu -ra
    ;;
  bench)
    echo 'Benchmark runner is not implemented yet; no measurements produced.' >&2; exit 2
    ;;
  package) bash scripts/collect_results.sh ;;
  *) echo 'Usage: scripts/gpu_session.sh {setup|test|bench <suite>|package}' >&2; exit 2 ;;
esac
