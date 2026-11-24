# Kernel design

## Softmax: GPU correctness verified, performance unmeasured

`tests/gpu/test_softmax.py` (130 cases) passed on an NVIDIA A100-SXM4-80GB on 2026-11-24
(torch 2.8.0, Triton 3.4.0; `reports/raw/k2_softmax_a100/`). Autotune winners from that run
were chosen on tiny correctness shapes and are not benchmark results. No measured crossover
or speed claim is available.
LayerNorm and causal attention remain unimplemented.

### Contract and layout

`softmax(x, dim=-1)` accepts fp16, bf16, or fp32 CUDA tensors and returns a contiguous tensor
of the same shape and dtype. Only the last dimension is normalized; either -1 or its positive
index is accepted. Scalar inputs and a zero-width final dimension are rejected. Empty leading
dimensions return an empty output without importing or launching the GPU kernel module.
This is forward-only code, without autograd integration.

The wrapper makes a contiguous copy when the final dimension is strided. With a unit final
stride, leading dimensions retain their strides, including row gaps, zero strides from expansion,
and transposed leading dimensions. A program unflattens its logical row ID into leading-axis
indices and takes their dot product with the input strides. The input tensor pointer already
includes its storage offset. Output rows are contiguous.

Program IDs are converted to int64 before grid flattening, row/stride products, and pointer
offsets. Large-path tile offsets also use int64. The host grid splits rows over CUDA's x/y grid
dimensions when the first dimension's limit would be exceeded; inactive rows have masked loads
and stores. Slow tests cover both row paths with more than 2^31 fp16 elements.

### Row-in-registers path

For N <= `SOFTMAX_SINGLE_BLOCK_LIMIT` (currently **8192, provisional**), one program holds a
row padded to the next power of two. Values are converted to fp32 before reductions.

```text
input logical rows                  one program per row
row 0: [x x x x x]       ->          [x x x x x -inf -inf -inf]
row 1: [x x x x x]       ->           max -> exp(x-max) -> sum
                                     -> divide -> masked output
output: contiguous rows, original dtype
```

For a row x, `m = max(x)` and `l = sum(exp(x - m))`; output is `exp(x - m) / l`.
Tail loads contribute negative infinity so they do not affect a finite row's maximum or sum.
The padded values are never stored outside the actual row.

### Looped path

Rows above the threshold use bounded tiles rather than one giant block. In the first pass,
let the running maximum and normalizer be m and l. For a tile T:

```text
m_new = max(m, max(T))
a     = exp(m - m_new)
l_new = a*l + sum(exp(T - m_new))
```

The code uses a compensated scalar sum to reduce accumulated rounding error across many
tiles. Its small correction is rescaled with the previous normalizer when the maximum changes.
FP fusion is disabled so the correction's subtraction sequence is retained. This numerical choice
still needs GPU evidence against the frozen tolerance rule.

```text
program for row r:
pass 1: [ tile 0 ] -> [ tile 1 ] -> ... -> [ final masked tile ]
          running maximum, normalizer, and rounding correction
pass 2: [ tile 0 ]    [ tile 1 ]    ...    [ final masked tile ]
          reread x -> exp(x - final_max) / final_sum -> output
```

An all-negative-infinity prefix needs a guard: if the merged maximum is still negative infinity,
the temporary exponent shift is zero. This avoids creating a NaN from `-inf - -inf` before a
later finite tile arrives. The stored maximum itself stays negative infinity. An entirely
negative-infinity row therefore still produces NaNs during final normalization, matching PyTorch.

NaN inputs enter the exponent sum and contaminate the row's normalizer. Positive-infinity
inputs also yield a NaN normalizer via `inf - inf`, as in PyTorch. Negative-infinity entries in
an otherwise finite row have exactly zero exponent. These cases are tested explicitly, but
remain unverified on NVIDIA hardware.

The looped path reads the input twice and writes once. Registered **effective** bandwidth still
uses the frozen `2*M*N*element_size` accounting; that idealized metric is not a claim that this
path physically transfers only two tensor-sized buffers. A final-dimension input copy adds
further traffic and allocation. The selected bandwidth cases currently use the single-block path.

### Autotuning and pruning

Both paths consider num_warps in {1, 2, 4, 8, 16}. The single-block path derives its block width
from N. The looped path considers {1024, 2048, 4096} columns per tile. num_stages is fixed at 1:
these reductions have no dot-driven load pipeline, and loop pipelining has not been measured.
Cache keys include N, dtype, device index, row count, leading shape, and leading strides.

Candidates with more threads than padded columns are pruned, except that one warp is allowed
for sub-warp rows. Remaining candidates are compiled and their handles initialized before
benchmarking; recognized resource, compile-time assertion, and ptxas errors remove that candidate.
Unexpected errors propagate. If none remain, a descriptive error lists the reasons, rather than
launching a failed configuration or calling a PyTorch fallback.

The pinned Triton implementation exposes `_init_handles()` and compiled register/spill/shared
memory information. This version-specific dependency is deliberately isolated in the GPU module.
Configuration status, exposed metadata, and the winning configuration are retained in
`PRUNING_LOG`. Owner-run pytest writes them to `reports/raw/softmax_autotune.json` with its exit
code; that file is configuration evidence, not a performance measurement. No actual valid or
winning configuration is known yet.

### Provenance and remaining work

The stable reduction structure is standard softmax math, illustrated by the
[Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
and its [pinned source](https://github.com/triton-lang/triton/blob/v3.4.0/python/tutorials/02-fused-softmax.py).
The large-row recurrence follows Milakov and Gimelshein's
[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
These structures are attributed algorithms, not a novelty claim. The implementation was written
for this project's stride mapping, bounded large-row path, edge cases, int64 indexing, and pruning
requirements; tutorial code was not pasted.

The pruning interface and resource exceptions were checked against the
[pinned autotuner source](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/runtime/autotuner.py)
and [compiled-kernel source](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/compiler.py).
Source review is not a substitute for compilation or execution on the owner's GPU.
The threshold must be justified by future measurements; neither path is accepted as correct or
fast until owner-provided GPU output supports it. Benchmarking, accuracy tables, and performance
criteria evaluation remain pending.
