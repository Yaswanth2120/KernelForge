# GPU session

Use Linux x86_64, Python 3.10-3.12, an NVIDIA sm_80+ GPU, and a compatible driver.
PyTorch is pinned to **2.8.0**; Triton must come from that PyTorch installation (never
`pip install triton` separately). Package instructions:
<https://pytorch.org/get-started/previous-versions/>

Every command below was run on a vast.ai A100-SXM4-80GB container on 2026-11-24.

## 1. Setup

```sh
git clone https://github.com/Yaswanth2120/KernelForge.git kernelforge && cd kernelforge
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/gpu_session.sh setup
```

Setup installs dependencies and writes the doctor output and the installed `do_bench` source
to `reports/raw/doctor.txt`. It fails (exit 3) when CUDA or the version check fails.
**Stop other GPU workloads first.** On the rented box, a co-resident vLLM server held 72 GiB
until it was stopped, and any co-resident work invalidates timings.

## 2. Correctness

```sh
bash scripts/gpu_session.sh test -m 'gpu and not slow'   # all ops; ~30-40 min on an A100
bash scripts/gpu_session.sh test -m 'gpu and slow'       # >2^31-element int64 tests, ~25 GiB
bash scripts/gpu_session.sh test tests/gpu/test_attention.py   # or one op
```

Failures print the parametrization and the max-error numbers. A skipped int64 test is pending
validation, not a pass. Expect long first runs: large single-block variants spend minutes in
ptxas, and Triton caches the compiled kernels across runs.

## 3. Benchmarks (A100 for reported numbers)

```sh
bash scripts/gpu_session.sh bench all    # or: softmax | layer_norm | causal_attention
```

`bench` measures BW_copy once (three launches; `reports/raw/copy_probe.json`), then runs each
suite as three fresh processes, writes `reports/raw/<suite>/launch_<i>.jsonl`, and regenerates
`reports/performance_report.md`. Exit code 1 means a pre-registered criterion failed; the
criteria tables are printed either way. Approximate A100 times: copy 1 min, softmax 5 min,
LayerNorm 10 min, attention 35 min (autotuning dominates the first launch).

Path thresholds, if a kernel changes (three launches each):

```sh
python scripts/measure_path_threshold.py --op softmax --out reports/raw/softmax_threshold
python scripts/measure_path_threshold.py --op layer_norm --out reports/raw/layer_norm_threshold
```

## 4. Profiling and interpreter

```sh
python scripts/profile_kernels.py --out reports/raw/profiling      # traces, metadata, experiments
CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python scripts/interpreter_probe.py
```

On the rented container, `ncu` failed with `ERR_NVGPUCTRPERM` (see docs/PROFILING.md); the
interpreter probe passed for fp32/fp16/bf16 on the Linux CPU.

## 5. Package and leave

```sh
bash scripts/gpu_session.sh package
```

This archives `reports/`, `PREREG.md`, and `docs/PROFILING.md` with the environment, git SHA,
git status, and any uncommitted diff, writing the archive to the repo root. Copy results off the
machine before destroying the rental; stopped instances still bill for storage.
