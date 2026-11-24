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
Softmax has a Triton implementation awaiting NVIDIA validation. LayerNorm and attention
remain explicit stubs that raise `NotImplementedError` on supported GPU calls.

## Status

- [x] Editable install and CPU checks verified
- [x] CPU-safe imports and unavailable-GPU errors verified
- [x] Pre-registered suites verified
- [x] CPU references, tolerance policy, accounting, and fake-timer harness verified
- [x] Softmax CPU interface and resource-pruning policy verified
- [ ] Softmax GPU correctness accepted
- [ ] LayerNorm and attention GPU kernels validated
- [ ] A100 measurements collected

GPU tests written, NOT run: the 130-case softmax matrix in `tests/gpu/test_softmax.py`.
No kernel correctness or performance claim is made. No performance results are available. Targets and measurement rules are frozen in [PREREG.md](PREREG.md).
CPU helpers and their contracts are described in [validation](docs/VALIDATION.md).
See [GPU setup](docs/GPU_SESSION.md), [profiling](docs/PROFILING.md), and [contributing](CONTRIBUTING.md).

## Limitations

GPU validation requires a separate NVIDIA session. Attention may miss its performance target;
a measured miss will be reported. Autotuning is hardware specific, rented hardware has clock
and thermal variance, and profiling counters may be unavailable. Large bandwidth cases use
at least 512 MiB inputs to reduce cache effects. Software versions must be recorded with results.
The work is iteration bound: planned final measurements need about 6–10 A100 hours,
with additional development sessions and substantially more debugging effort.
