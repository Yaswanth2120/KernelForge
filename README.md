# KernelForge

A small library of custom Triton GPU kernels (fused softmax, LayerNorm forward, and causal
attention forward), with correctness tests, benchmarks against PyTorch, and a pre-registered,
generated performance report measured on an NVIDIA A100 80GB SXM4.

## Results (A100-SXM4-80GB, torch 2.8.0, Triton 3.4.0, three independent launches)

| Pre-registered criterion | Cases | Result |
|---|---|---|
| 1. Correctness tests pass (CPU and GPU suites) | 219 CPU + 382 GPU | PASS (see the evidence under `reports/raw/k*_a100/`) |
| 2. Softmax effective bandwidth >= 0.80 x BW_copy (1757.9 GB/s) | 9 | 9/9 PASS (0.984-0.993) |
| 3. LayerNorm effective bandwidth >= 0.80 x BW_copy | 9 | 9/9 PASS (0.981-0.999) |
| 4. Causal attention TFLOPS >= 0.50 x SDPA-FlashAttention | 12 | 12/12 PASS (0.807-1.006) |
| 5. Numerical accuracy per the frozen tolerance rule | all tested cases | PASS |

The full tables, plots, accuracy, memory scaling, autotune choices, telemetry, and the "where we
lose" analysis are in the generated [performance report](reports/performance_report.md).

Where KernelForge loses (not pass/fail; all measured):

- LayerNorm at N >= 65536: 0.78-0.92x `F.layer_norm`. The looped path reads each row three
  times, a trade for accurate two-pass statistics.
- LayerNorm at irregular widths (1000, 4097, 31337), down to 0.17x `F.layer_norm`. The cause
  is **not understood**; row alignment was tested and ruled out ([profiling](docs/PROFILING.md)).
- Attention at head dim 128: 0.77-0.86x SDPA-FlashAttention.
- Softmax at N >= 65536: KernelForge is still faster than `torch.softmax`, but below 0.80 of
  BW_copy.

## Quickstart (CPU: tests, tooling, report generation)

Use Python 3.10-3.12. On macOS or Linux without a GPU:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
pytest -m "not gpu and not slow"
ruff check .
kernelforge check-prereg
kernelforge report --raw reports/raw --out reports   # regenerate the report from committed JSON
```

`import kernelforge` works without Triton or CUDA. The ops raise `GPUUnavailableError` there.

## GPU usage

On Linux with an NVIDIA sm_80+ GPU and the pinned PyTorch (Triton comes bundled with it):

```python
import torch
from kernelforge import causal_attention, layer_norm, softmax

x = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)
y = softmax(x)                                            # last dim only
z, mean, rstd = layer_norm(x, 8192, torch.ones(8192, device="cuda"), None, return_stats=True)
q, k, v = (torch.randn(2, 32, 4096, 128, device="cuda", dtype=torch.float16) for _ in range(3))
o = causal_attention(q, k, v)                             # (B, H, S, D), causal, no S x S matrix
```

Full GPU session (setup, tests, three-launch benchmarks, report, packaging):
[docs/GPU_SESSION.md](docs/GPU_SESSION.md).

## Documentation

- [PREREG.md](PREREG.md): frozen shapes, criteria, and timing protocol.
- [DECISIONS.md](DECISIONS.md): every ambiguous choice, every change, and every bug found on the GPU, dated.
- [docs/KERNELS.md](docs/KERNELS.md): per-kernel math, tiling diagrams, numerics, autotuning, and citations.
- [docs/PROFILING.md](docs/PROFILING.md): what could and could not be profiled, roofline-lite analysis, and the experiments.
- [EXPLAINED.md](EXPLAINED.md): a plain-language explanation of every function.
- [docs/VALIDATION.md](docs/VALIDATION.md): CPU references, the tolerance rule, and accounting.

## Status

- [x] CPU package, references, tolerance policy, accounting, harness, CLI, CI
- [x] Softmax: 130/130 GPU tests, threshold measured (65536), benchmark (criterion 2 pass)
- [x] LayerNorm: 168/168 GPU tests, threshold measured (32768), benchmark (criterion 3 pass)
- [x] Causal attention: 80/80 GPU tests, benchmark and memory scaling (criterion 4 pass)
- [x] Profiling with fallbacks (ncu blocked: `ERR_NVGPUCTRPERM`)
- [ ] Release tag `v0.1.0` (after the stacked PRs are merged into `main`)

## Limitations

- **Hardware scope.** Every number is from one rented A100-SXM4-80GB with torch 2.8.0 and
  Triton 3.4.0. Autotuned choices and performance do not transfer to other GPUs or versions.
- **Power and clock variance.** Attention launches ran at the 400 W software power cap (throttle
  reason 0x4); absolute TFLOPS are cap-limited, and the report flags it. Spreads across three
  launches are reported, not just medians.
- **Cache effects.** Bandwidth claims use inputs of at least 512 MiB (about 13x the L2 cache);
  small shapes in the sweep may partly measure cache.
- **No hardware counters.** Nsight Compute was blocked in the container, so the profiling
  conclusions rest on traces, compile metadata, and controlled experiments.
- **Accounting.** Effective bandwidth uses the frozen byte formulas. The looped paths physically
  move more bytes than counted (softmax two reads, LayerNorm three).
- **Not implemented** (non-goals): backward passes, autograd or `torch.compile` integration,
  GQA/MQA, variable length, dropout, KV cache, multi-GPU, non-NVIDIA backends.
- **Process note.** The spec assumed the coding agent could not run a GPU. In practice the
  GPU sessions were run over SSH on the owner's rented box. DECISIONS.md records this, plus
  every test fix and every post-hoc rule change, with original and changed values.
- **Effort.** The GPU spend is small; the cost is in iteration (edit, run, fail, fix) and in
  measurement.
