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

## 2026-11-23 — Public repository policy

The owner clarified that KernelForge must be public. This supersedes the initial private
visibility choice. The repository is now public, and main-branch protection is enabled:
pull requests, current passing Python 3.10–3.12 checks, and administrator enforcement.
Force pushes and deletion of main are blocked. The previous plan limitation no longer applies.

## 2026-11-23 — CPU correctness and measurement foundation

- Reference functions return fp32 by default, with an explicit fp64 option for CPU self-tests.
  They promote the same original inputs before computing; they never regenerate inputs in higher
  precision. LayerNorm uses centered population variance and supports affine parameters independently,
  trailing multidimensional shapes, empty leading dimensions, and optional statistics.
  Returned statistics use the selected reference precision. Reference eps must be finite and positive.
- The attention reference materializes S×S scores and is intended only for small correctness cases.
  It uses an inclusive causal mask, supports scale=0, and optionally returns natural-log LSE.
  This reference is not a memory-efficient implementation or a custom GPU operation.
- Error comparisons convert outputs and reference to fp32. Maximum absolute error is accepted iff
  it is <= max(2.0 * PyTorch maximum absolute error, 2.0 * epsilon(dtype) * max(abs(reference))).
  Relative error ignores abs(reference) < 1e-8. Matching NaN locations and matching signed infinities
  are required for both the candidate and baseline; finite entries supply the numeric error metrics
  and reference magnitude. Any nonfinite mismatch fails regardless of the numeric threshold.
  Empty tensors have zero errors; all-zero references and exact baselines give a zero threshold.
  These constants implement the original policy; no performance or accuracy targets changed.
- Accounting uses decimal GB/s and TFLOPS, with times in milliseconds. Missing affine parameters
  contribute zero bytes. Attention retains the frozen 2*B*H*S²*D approximation, including its
  omission of the exact diagonal correction; the accounting is not silently changed to S*(S+1).
- Timer.measure returns p20/p50/p80 milliseconds. Raw-sample helpers use inclusive linear
  interpolation at positions (n-1)*q. A real timer may supply do_bench's quantiles directly;
  the installed source still needs review before any GPU benchmark.
- Pydantic validates schema identity, required environment/telemetry/autotune fields, ordered finite
  positive times, nonnegative integer counts, and derived rates. Additional fields for FLOPs,
  TFLOPS, measured copy bandwidth, baseline call text, and invalid configurations extend the
  original record without changing its schema identifier or required original keys.
  Use model_dump_json(by_alias=True) to emit the required JSON key `schema`.
  Unmeasured tuning duration is null rather than an invented zero.
- Aggregation requires exactly three records with indices {0,1,2} and matching operation,
  implementation, case, work counts, environment, copy baseline, and baseline call. It retains
  all source records, reports the median of process medians and their min/max, and computes
  rates from that median. Summary quantiles are medians of the corresponding per-launch
  quantiles, not pooled-sample quantiles. Distinct indices cannot prove distinct processes;
  actual process isolation remains the GPU runner's responsibility.
- Observed on macOS ARM64 / PyTorch 2.8.0: CPU SDPA FlashAttention with is_causal=True and
  scale=0 produced 384 NaNs for shape (1,1,7,64), while the forced math backend and explicit
  reference produced none and agreed exactly. The fp64 CPU self-test explicitly forces SDPA
  MATH, retains zero-scale coverage, and checks other scales too. This is an upstream/backend
  finding, not a custom kernel result. The math baseline is not used as a performance substitute
  for the required GPU FlashAttention baseline.

## 2026-11-23 — Softmax implementation pending NVIDIA validation

- Implement softmax only. GPU tests written, NOT run: tests/gpu/test_softmax.py (130 cases).
  CPU policy tests do not compile Triton or prove kernel correctness. Keep the implementation PR
  in draft until owner-run GPU evidence is available; do not advance to performance measurement.
- SOFTMAX_SINGLE_BLOCK_LIMIT is provisionally 8192 columns. This bounds the single-program row
  while keeping all selected bandwidth widths on that path; it is NOT a measured crossover.
  The required measured justification is unfinished. Any threshold update must be dated here,
  with its measured reason; no frozen shape, accuracy rule, or performance criterion was changed.
- Map logical rows through all leading shape/stride dimensions rather than reshaping the input,
  which could silently copy a leading-dimension transpose. Copy a strided final dimension with
  contiguous(), and always allocate a contiguous same-dtype output. An empty leading dimension
  returns without importing the kernel module. N=0/scalars and unsupported dtypes are rejected.
- Availability checks accept an optional device so a CUDA input is checked on its own GPU.
  The default availability interface and the other operations' behavior remain unchanged.
- Cast program IDs before all row/grid stride arithmetic, and cast large-row tile offsets to
  int64. Split rows into a second grid dimension above CUDA's x-grid limit and mask tail rows.
- Use fp32 max/exp/sum arithmetic. The looped path uses the published online normalizer
  recurrence, followed by an output pass, with a compensated scalar sum and disabled FP fusion.
  An all-negative-infinity prefix uses a temporary zero shift until a finite maximum arrives;
  final all-negative-infinity rows still produce NaNs. These numerical choices await GPU tests.
- Autotune warps {1,2,4,8,16}; derive single blocks from N and consider loop blocks {1024,2048,4096}.
  num_stages=1 is explicit: no measured loop-pipeline choice exists, and these kernels contain no
  dot operations. Include N/dtype/device/rows/layout in the tuning key to avoid sharing a winner
  across different layouts or GPUs. No winner or speed is claimed before actual execution.
- Prune configurations with excess idle threads; precompile and initialize handles for the rest.
  Catch only Triton's resource, compile-time assertion, and ptxas failures. Record valid/pruned/
  invalid configs plus exposed register/spill/shared metadata; unexpected compiler errors propagate.
  If all configurations fail, report all reasons instead of launching an invalid fallback.
  The private _init_handles API was reviewed in the pinned upstream 3.4.0 source, but not executed.
- The previous CPU test assumed softmax was a stub. That intended behavior changed: host-side
  softmax validation/copy/pruning tests replace that assumption; the remaining stub assertion now
  checks LayerNorm. No correctness test or tolerance was weakened to make a kernel pass.
- The owner-run GPU script accepts pytest flags, rejects unavailable GPU environments with code 3,
  saves JUnit output, and saves autotune diagnostics only when the GPU backend was actually loaded.
  Slow int64 cases need >8 GiB for input/output plus 2 GiB headroom; memory-based skips remain
  pending, not passes. The full matrix includes both single and looped int64 paths.
- The looped algorithm physically reads twice and writes once. Keep the registered 2*M*N*elem_size
  effective-bandwidth formula unchanged and disclose the difference from physical traffic in docs.

## 2026-11-24 — Softmax GPU validation run (A100) and tuple-indexing fix

- Provenance: Claude Code ran this session over SSH at the owner's request (owner-provided
  vast.ai container) instead of the owner pasting output. Device NVIDIA A100-SXM4-80GB (cc 8.0),
  driver 595.91.07, torch 2.8.0, Triton 3.4.0 (bundled), Python 3.12.3, Ubuntu 24.04. A
  co-resident vLLM server (owner's other project) was stopped before testing; GPU showed 0 MiB used.
- First run: 115 of 128 non-slow cases failed to compile. `_row_location` computed
  `axis = len(LEADING_SHAPE) - index - 1` from a `tl.static_range` index; in Triton 3.4 that
  arithmetic yields a tensor, which cannot index a constexpr tuple. Fix: iterate
  `tl.static_range(len(LEADING_SHAPE) - 1, -1, -1)` directly. Probed first on 2-D, 1-D and empty
  leading shapes. No test, tolerance, or criterion changed.
- After the fix: 130/130 passed in one `gpu_session.sh test` run (exit 0), including both
  >2^31-element int64 tests. Pruning recorded 662 valid / 138 pruned / 0 invalid configs; some
  single-block BLOCK=8192 and looped configs spill registers at num_warps=1 (kept; autotune chooses).
  The commit SHA in the results is 1d0a7da plus `uncommitted.diff` (this fix), both saved in
  `reports/raw/k2_softmax_a100/`. `collect_results.sh` does not record dirty trees; flagged, not changed.
- do_bench behavior (Triton 3.4.0 `triton/testing.py`, read from the installed source): one
  untimed call then synchronize; a 256 MiB int32 buffer is zeroed (`cache.zero_()`) before every
  timed call to flush L2, including the 5 estimation calls; warmup/rep are milliseconds converted
  to counts via `max(1, int(ms / estimate_ms))`; warmup calls do not flush; each timed call is
  bracketed by its own CUDA events and a single synchronize follows all repetitions; quantiles
  are computed over the per-call event times. The zeroing kernel is outside the event pair.
- SOFTMAX_SINGLE_BLOCK_LIMIT remains provisional (8192): no crossover timing was taken.

## 2026-11-24 — Softmax path-threshold measurement protocol (stated before measuring)

- Script: `scripts/measure_softmax_threshold.py`. Both paths are forced via `launch_softmax`'s
  limit argument. Decision widths N in {1024, 2048, 4096, 8192, 16384, 32768, 65536};
  informational widths {4097, 6144, 8193, 12288} are reported but do not decide. dtypes
  fp16/bf16/fp32; M chosen so the input is >= 512 MiB; autotune runs before timing;
  `do_bench` quantiles [0.2, 0.5, 0.8]; three fresh processes; nvidia-smi telemetry per launch.
- Rule: SOFTMAX_SINGLE_BLOCK_LIMIT = the largest decision N where the single path's
  median-of-three-medians p50 is <= the looped path's for every dtype, at that N and at every
  smaller decision N. Strict comparison, no tie band. If the rule gives a value other than 8192,
  the constant changes and the reason is recorded here; frozen PREREG shapes and criteria do not.
- Commit 2bc1e64 pushed this script with lint errors and without this entry (a shell chaining
  mistake); the follow-up commit fixes lint and adds this entry before any measurement ran.
