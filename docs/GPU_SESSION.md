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

`test` runs available GPU tests; it currently fails clearly because none exist yet.
`bench <suite>` is reserved and currently fails without writing measurements.
`package` archives results and environment information. Save the archive before destroying a rental.
Use a cheaper sm_80+ GPU for correctness; final measurements require A100 80GB SXM4,
three independent process launches, and telemetry. No GPU session has been verified yet.
