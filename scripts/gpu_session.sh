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
    python - <<'PYTHON'
import sys
from kernelforge import is_triton_available
if not is_triton_available():
    print('GPU tests require NVIDIA sm_80+ CUDA and PyTorch-bundled Triton.', file=sys.stderr)
    raise SystemExit(3)
PYTHON
    shift
    mkdir -p reports/raw
    python -m pytest tests/gpu -m gpu -ra --tb=short --junitxml=reports/raw/gpu-tests.xml "$@"
    ;;
  bench)
    echo 'Benchmark runner is not implemented yet; no measurements produced.' >&2; exit 2
    ;;
  package) bash scripts/collect_results.sh ;;
  *) echo 'Usage: scripts/gpu_session.sh {setup|test|bench <suite>|package}' >&2; exit 2 ;;
esac
