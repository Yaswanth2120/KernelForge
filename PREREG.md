## 3. Pre-registered criteria and protocol

### 3.1 Hardware and baselines
- Target GPU: NVIDIA A100 80GB SXM4 (compute capability 8.0, `sm_80`). Datasheet figures for context only: memory bandwidth 2,039 GB/s, BF16/FP16 Tensor Core peak 312 TFLOPS dense. **Pass/fail criteria use measured baselines, not datasheet numbers.**
- Results labeled "A100" must come from a run whose recorded device name contains `A100`; the report generator refuses to apply that label otherwise. Development results from other GPUs are never mixed into the report.
- Memory-copy baseline `BW_copy`: the maximum, over a Triton copy kernel and `torch.Tensor.copy_`, of achieved bandwidth on a ≥ 1 GiB buffer (bytes counted as 2 x size: one read plus one write), same timing method as the kernels, median of 3 independent process launches.
- Attention baseline: `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)` forced to the FlashAttention backend via `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)`. Also report the default-selected backend and the math backend (memory comparison).

### 3.2 Byte and FLOP accounting (state in the report)
- Softmax forward bytes: `2 * M * N * elem_size` (read x, write y).
- LayerNorm forward bytes: `2 * M * N * elem_size + N * (weight_elem + bias_elem)` (plus `2 * M * 4` if per-row mean/rstd are returned). Weight/bias terms are counted even though small.
- Causal attention forward FLOPs: `2 * B * H * S^2 * D` (the two matmuls counting only the unmasked lower triangle: `4 * B * H * S^2 * D` for full attention, halved). Report TFLOPS and also percent of the 312 TFLOPS datasheet peak.
- Effective bandwidth = bytes / median time. Ratio = effective bandwidth / `BW_copy`.

### 3.3 Selected shapes (the pass/fail set; the full sweep is reported regardless)
- **Softmax and LayerNorm:** `N` in {2048, 4096, 8192}; dtype in {fp16, bf16, fp32}; `M` chosen so the input tensor is ≥ 512 MiB (about 13x the A100's 40 MB L2, so results measure HBM, not cache).
- **Attention:** dtype in {fp16, bf16}; head dim `D` in {64, 128}; `H = 32`; `(B, S)` in {(8, 2048), (4, 4096), (2, 8192)}.

### 3.4 Success criteria (all must hold on all selected cases unless stated; report failures as results)
1. All correctness tests pass (CPU and GPU suites), including the edge-case matrix.
2. Softmax: ratio ≥ 0.80 on every selected case.
3. LayerNorm: ratio ≥ 0.80 on every selected case.
4. Attention: Triton TFLOPS ≥ 0.50 x SDPA-FlashAttention TFLOPS on every selected case.
5. Numerical accuracy: per section 6's rule on every tested case.
If a criterion is missed, the report states which cases, by how much, and the analysis; it does not hide or redefine the criterion.

### 3.5 Timing protocol
- `triton.testing.do_bench` with `quantiles=[0.2, 0.5, 0.8]`; report the median and the spread. Before benchmarking, read the installed version's implementation and record in `DECISIONS.md` exactly what it does about warmup units (milliseconds), L2 cache flushing between runs, and synchronization. Do not assume.
- Autotuning completes before timing; autotune time is excluded and reported separately.
- Fixed seeds; deterministic inputs; `torch.cuda.synchronize()` around anything not covered by `do_bench`.
- **Three independent process launches** per suite; final numbers are the median of the three medians, with the range reported. Record `nvidia-smi` SM clock, memory clock, power, and temperature at start and end of each launch; flag launches with clock drops.
- Peak memory per op via `torch.cuda.reset_peak_memory_stats()` and `max_memory_allocated()`, reported as the delta over the inputs.

