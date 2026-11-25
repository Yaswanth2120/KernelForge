#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/raw
python -c 'import json; from kernelforge.env import doctor; print(json.dumps(doctor(), indent=2))' > reports/raw/environment.json
git rev-parse HEAD > reports/raw/git_sha.txt
# Record uncommitted source changes so the SHA alone never misdescribes what ran.
git status --porcelain --untracked-files=no > reports/raw/git_status.txt
git diff > reports/raw/uncommitted.diff
tar -czf "kernelforge-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" reports docs/PROFILING.md PREREG.md
