# Decisions

## 2026-11-23 — Project foundation

- Keep local workflow instructions outside version control at the owner's request. Public documents
  describe product behavior and validation rather than internal work stages.
- Create the repository private by default; visibility can be changed separately. Use feature branches,
  pull requests, conventional commits, CPU CI, and squash merges.
- Pin PyTorch 2.8.0 on CPU and GPU, with Python 3.10–3.12. GPU setup installs from PyPI,
  letting the Linux PyTorch dependency select its matching Triton; never install Triton separately.
  Official version instructions: https://pytorch.org/get-started/previous-versions/
- Select the minimum integer M giving at least 512 MiB for each registered N and dtype.
  Freeze the complete PREREG.md bytes by SHA-256 and independently validate suite constants.
- do_bench behavior: NOT YET DETERMINED from the installed GPU environment. No claims about
  warmup units, cache flushing, or synchronization are made. The owner must run
  `kernelforge doctor --timing-source` and provide the installed source; record exact behavior here
  before any timed suite. The source dump is saved by the setup script.
- Interpreter mode: NOT YET DETERMINED. Docker CLI exists, but its daemon is unavailable on
  the development Mac. The Linux CPU probe covers masked tails across fp32/fp16/bf16;
  success would establish only that probe's behavior, not attention intrinsic support.
  Interpreter CI tests will be added only after actual Linux evidence.
- `doctor --probe-ncu` runs a trivial PyTorch CUDA add under ncu on the owner's host.
  CPU doctor never runs kernels. Missing tools and offline environments produce structured diagnostics.
- Public operations are explicit stubs: missing GPU raises GPUUnavailableError; a usable GPU
  raises NotImplementedError until kernels are implemented. No fallback pretends to be a custom kernel.

- Owner requested no internal stage labels in GitHub. PREREG.md preserves all numerical criteria
  and protocols; its heading and timing instruction omit internal stage references only.

- GitHub rejected main-branch protection with HTTP 403: this private repository requires
  GitHub Pro or public visibility. Squash-only merging and automatic branch deletion are enabled;
  PR review and green CI are a documented workflow rather than server-enforced protection.
