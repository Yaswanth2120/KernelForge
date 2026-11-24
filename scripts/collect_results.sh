#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/raw
python -c 'import json; from kernelforge.env import doctor; print(json.dumps(doctor(), indent=2))' > reports/raw/environment.json
git rev-parse HEAD > reports/raw/git_sha.txt
tar -czf "kernelforge-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" reports docs/PROFILING.md PREREG.md
