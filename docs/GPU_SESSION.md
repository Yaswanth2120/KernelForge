# GPU session

Use Linux x86_64, Python 3.10–3.12, an NVIDIA sm_80+ GPU, and a compatible driver.
PyTorch is pinned to **2.8.0**; Triton must come from that PyTorch installation.
Official package instructions: https://pytorch.org/get-started/previous-versions/
Use a fresh virtualenv, clone this repo, and run:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/gpu_session.sh setup
```

Setup installs dependencies and captures doctor output, an optional ncu probe, and the installed
`do_bench` source in `reports/raw/doctor.txt`. Review its warmup units, cache clearing,
and synchronization before benchmarking. Setup fails when CUDA or the version check fails.

On a Linux CPU environment with the same PyTorch-provided Triton, run
`python scripts/interpreter_probe.py` and save all dtype outcomes. This has not yet been verified.

`test` requires a supported NVIDIA runtime and runs the softmax GPU matrix. It accepts pytest
arguments after `test`, prints failing parametrizations with numerical error details, and saves
JUnit output plus autotune validity/winner diagnostics under `reports/raw/`.
GPU tests written, NOT run: `tests/gpu/test_softmax.py` (130 cases).

```sh
# Normal correctness checks on an inexpensive sm_80+ GPU:
TRITON_PRINT_AUTOTUNING=1 bash scripts/gpu_session.sh test -m 'gpu and not slow'
# Both int64 paths: more than 2^31 elements, >8 GiB tensors plus 2 GiB headroom:
bash scripts/gpu_session.sh test -m 'gpu and slow'
# Or run the full matrix in one command:
bash scripts/gpu_session.sh test
bash scripts/gpu_session.sh package
```

Paste the full short pytest summary and any failures back for review. A skipped int64 test is
pending validation, not a pass. Softmax remains unaccepted until the full GPU matrix passes.
The 8192-column path threshold is provisional and still needs measured justification after
reviewing the installed timer source; no crossover timing is claimed here.
`bench <suite>` is reserved and currently fails without writing measurements.
`package` archives results and environment information. Save the archive before destroying a rental.
Use a cheaper sm_80+ GPU for correctness; final measurements require A100 80GB SXM4,
three independent process launches, and telemetry. No GPU session has been verified yet.
