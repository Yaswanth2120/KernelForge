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

For N <= `SOFTMAX_SINGLE_BLOCK_LIMIT` (**65536**, measured on the A100; see DECISIONS.md),
one program holds a row padded to the next power of two. At N <= 4096 both paths reach the same
~1.73 TB/s; from 8192 to 65536 the single path was 5-31% faster because the looped path reads
each row from HBM twice. Larger N was not measured, so the looped path keeps N > 65536. Values are converted to fp32 before reductions.

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

## LayerNorm forward

### Math and contract

For each row x of length N (the trailing `normalized_shape` dims, flattened):

```text
mean = sum(x) / N                      (row sum in fp64, rounded once to fp32)
var  = sum((x - mean)^2) / N           (population variance, centered: never E[x^2] - E[x]^2)
rstd = 1 / sqrt(var + eps)             (IEEE div_rn / sqrt_rn)
y    = (x - mean) * rstd * weight + bias   (weight, bias optional; fp32 or the input dtype)
```

Output has the input's dtype and shape; `return_stats=True` adds fp32 `mean` and `rstd` with the
leading shape. The wrapper flattens to (M, N): a view when the leading dims collapse to a single
row stride, otherwise a copy, and a non-unit-stride trailing block is copied. Row offsets use
int64 (`row * row_stride`), and rows above CUDA's x-grid limit spill into a second grid dimension.

### Single-block path (N <= 32768)

```text
program r:   x[r, 0:N] --load (masked, pad 0)--> registers [BLOCK = next_pow2(N)]
             sum (fp64) -> mean;  where(mask, x - mean, 0) -> centered
             sum(centered^2) -> var -> rstd
             centered * rstd * w + b --store (masked)--> y[r, 0:N]
```

One HBM read and one write per element. The masked padding is zero and is also zeroed after
centering, so padding never contributes to the variance.

### Looped path (N > 32768)

```text
program r:   pass 1: tiles of BLOCK columns      lane sums (fp64) += x      -> tree sum -> mean
             pass 2: same tiles, reread x        lane sums += (x - mean)^2  -> tree sum -> rstd
             pass 3: same tiles, reread x        write (x - mean) * rstd * w + b
```

Three reads and one write. Each lane accumulates only its own columns (N / BLOCK values), and a
single tree reduction finishes the row, so no running scalar is rounded at the whole row's scale.
An earlier design merged per-tile (count, mean, M2) with Chan et al.'s parallel-variance formula
in two reads; on the A100 it turned a +inf row's mean into NaN and lost ~2 ulp of rstd at
N = 262144, so it was replaced (DECISIONS.md, 2026-11-24).

### Numerical choices

- Triton lowers fp32 `/` and `sqrt` to fast approximate instructions by default. With them, a
  constant row 3,3,...,3 (N=31) got mean 93/31 != 3, and rstd amplified the tiny residue into a
  5e-5 output error. `tl.div_rn` / `tl.sqrt_rn` fixed it.
- The row sum is the only cancellation-prone statistic (e.g. 32769 normal values summing to
  ~0.005*N). Summing in fp64 matched PyTorch's accuracy on that case; centered squares do not
  cancel and stay in fp32. Stored statistics are fp32 as specified.
- NaN anywhere in a row makes the row NaN; +inf gives mean +inf (matching the fp32 reference;
  PyTorch reports NaN there) and a NaN output row, as in PyTorch.

### Threshold and autotuning

The single/looped switch is 32768, measured on the A100 with a rule fixed before measuring
(largest N where the single path is within 1% of, or faster than, the looped path for every
dtype at that N and below). At 65536 the single path is 10-23% slower: it keeps x and the
centered row live and spills registers. Autotune: num_warps {1, 2, 4, 8, 16} (more threads than
padded columns pruned), looped BLOCK {1024, 2048, 4096}; keyed on N, the weight/bias/stats
flags, and (automatically) tensor dtypes. Triton's autotuner skips configs that fail to compile
or launch.

### Known limits

Irregular widths (4097, 8193) were 1.6-2.5x slower than neighbouring powers of two on both
paths in fp16/bf16 during the threshold sweep; the cause is analyzed in docs/PROFILING.md.

### Provenance

The normalization is the standard LayerNorm definition (Ba, Kiros, Hinton,
[Layer Normalization](https://arxiv.org/abs/1607.06450)); the
[Triton layer-norm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
was read as a reference. The two-pass centered variance is the textbook stable method
(Chan, Golub, LeVeque, *Algorithms for computing the sample variance*, 1983, also the source of
the superseded merge formula). The code was written for this project's layouts, mixed-precision
affine, statistics output, looped path, and int64 indexing; tutorial code was not pasted.
