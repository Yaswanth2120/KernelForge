# KernelForge

Custom Triton GPU kernels with reproducible correctness tests and benchmark methodology.
The planned operations are fused softmax, LayerNorm, and causal attention forward.

## Quickstart

Use Python 3.10–3.12. On macOS:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
kernelforge check-prereg
kernelforge doctor
```

`doctor` prints diagnostics and exits 3 when a supported GPU environment is unavailable.
Imports work on CPU machines; operation calls raise `GPUUnavailableError` there.
Softmax has a Triton implementation that passed its 130-case GPU correctness matrix on an
NVIDIA A100-SXM4-80GB (2026-11-24; evidence in `reports/raw/k2_softmax_a100/`). LayerNorm and attention
remain explicit stubs that raise `NotImplementedError` on supported GPU calls.

## Status

- [x] Editable install and CPU checks verified
- [x] CPU-safe imports and unavailable-GPU errors verified
- [x] Pre-registered suites verified
- [x] CPU references, tolerance policy, accounting, and fake-timer harness verified
- [x] Softmax CPU interface and resource-pruning policy verified
- [x] Softmax GPU correctness matrix passed (130/130 incl. both >2^31-element int64 tests, A100, torch 2.8.0 / Triton 3.4.0)
- [x] Softmax single/looped path threshold measured on A100 (limit 65536; see DECISIONS.md)
- [x] Softmax A100 benchmark: copy baseline + sweep, three launches, report generated
- [x] LayerNorm GPU correctness matrix passed (168/168 incl. both >2^31-element int64 tests, A100, commit 9852793)
- [x] LayerNorm single/looped threshold measured on A100 (limit 32768)
- [ ] Causal attention GPU kernel validated
- [ ] LayerNorm and attention A100 measurements collected

## Results so far (A100-SXM4-80GB, three launches; full tables in the [report](reports/performance_report.md))

| criterion | cases | result |
|---|---|---|
| Softmax effective bandwidth >= 0.80 x BW_copy (1757.9 GB/s) | 9 pre-registered | 9/9 PASS (0.984-0.993) |

KernelForge softmax was faster than `torch.softmax` on all 39 sweep shapes. Sweep shapes with
N >= 65536 fall below the 0.80 ratio (not pass/fail); see the report's "Where we lose" section.
Claims are limited to the measured environment. Targets and measurement rules are frozen in [PREREG.md](PREREG.md).
CPU helpers and their contracts are described in [validation](docs/VALIDATION.md).
See [GPU setup](docs/GPU_SESSION.md), [profiling](docs/PROFILING.md), and [contributing](CONTRIBUTING.md).

## Limitations

GPU validation requires a separate NVIDIA session. Attention may miss its performance target;
a measured miss will be reported. Autotuning is hardware specific, rented hardware has clock
and thermal variance, and profiling counters may be unavailable. Large bandwidth cases use
at least 512 MiB inputs to reduce cache effects. Software versions must be recorded with results.
The work is iteration bound: planned final measurements need about 6–10 A100 hours,
with additional development sessions and substantially more debugging effort.
