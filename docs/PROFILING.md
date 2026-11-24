# Profiling evidence (K7)

All numbers come from `reports/raw/profiling/` (produced by `scripts/profile_kernels.py` on the
A100-SXM4-80GB, torch 2.8.0, Triton 3.4.0) and from the benchmark records under `reports/raw/`.
Experiment timings are single-process `do_bench` medians: controlled comparisons, not criteria.

## What was and was not possible

| Tool | Outcome |
|---|---|
| Nsight Compute (`ncu` 2022.4 from Ubuntu apt) | **Failed: `ERR_NVGPUCTRPERM`.** The driver has `RmProfilingAdminOnly: 1` and the rented container lacks `CAP_SYS_ADMIN` (CapEff `0xa80405fb`, bit 21 clear). No DRAM throughput, occupancy, L2 hit rate, or stall-reason counters were collected. |
| `torch.profiler` | Worked: kernel names, durations, grid/block, registers per thread, shared memory, estimated occupancy (`traces.json`). Attention calls produced no CUDA kernel events in this capture (also seen intermittently in K6); attention evidence comes from the items below. |
| Triton compile metadata | Worked: registers, spills, shared memory for every compiled variant (`compiled.json`). |
| Variant experiments | Worked: row-stride alignment, forced `num_warps`, forced attention tiles, plus the single-vs-looped threshold sweeps already measured (`reports/raw/*_threshold/`). |

Section 8's fallback was therefore mandatory, and hardware-counter claims are absent.

## Roofline-lite classification

Ridge point = 312 TFLOPS (datasheet) / 1758 GB/s (measured BW_copy) = **177 FLOP/byte**.

| Kernel | FLOP/byte (approx.) | Class | Measured |
|---|---|---|---|
| Softmax (max, sub, exp, sum, div ~ 5 FLOP/element; 2 bytes moved per fp16 element each way) | ~1.25 (fp16), ~0.6 (fp32) | memory bound | 0.98-0.99 of BW_copy at selected N |
| LayerNorm (~8 FLOP/element) | ~2 (fp16), ~1 (fp32) | memory bound | 0.98-1.00 of BW_copy at selected N |
| Causal attention: 2*B*H*S^2*D FLOPs over 4*B*H*S*D elements | S/4 = 512-2048 at selected S | compute bound | 141-174 TFLOPS, 45-56% of datasheet peak, at the SW power cap |

The memory-bound kernels are at the measured copy ceiling for the selected shapes, so further
speedup there is not available without moving fewer bytes. Attention sits well right of the
ridge; its limit is tensor-core throughput and the 400 W power cap (K6 telemetry: throttle
reason 0x4 on every attention launch).

## Experiment 1: single vs looped paths (threshold sweeps)

Already measured with pre-stated rules (DECISIONS.md). Softmax: the looped path, which reads the
row twice, is 5-31% slower from N = 8192 to 65536, so the single path is used up to 65536.
LayerNorm: the looped path reads three times; the single path wins up to 32768 but is
10-23% slower at 65536, where it spills (Experiment 3). **Conclusion:** extra HBM passes cost
bandwidth in proportion to bytes; register spills cost more than a re-read once a row no longer
fits.

## Experiment 2: row-stride alignment (hypothesis refuted)

Hypothesis from K4: irregular widths are slow because unaligned rows prevent vectorized loads.
Test: same N with natural row stride vs a 64-element-aligned stride (a padded view).

| op | N | dtype | natural ms | aligned ms |
|---|---|---|---|---|
| layer_norm | 4097 | bf16 | 2.111 | 2.071 |
| layer_norm | 31337 | bf16 | 12.242 | 12.258 |
| layer_norm | 31337 | fp32 | 5.867 | 5.844 |
| softmax | 31337 | bf16 | 0.742 | 1.193 |

Aligning the rows does not help LayerNorm (within 2%), so **row alignment is not the cause**.
(Softmax got slower with the padded view at N = 31337 bf16; this was not investigated.)

## Experiment 3: forced `num_warps` on the LayerNorm single path (fp16, BLOCK = N)

| N | 1 warp | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8192 (ms) | 0.724 | 0.626 | 0.625 | 0.620 | 0.639 |
| 32768 (ms) | 5.707 | 6.344 | 0.729 | 0.686 | 0.637 |

At N = 32768, 1-2 warps are about 9x slower. Compile metadata for LayerNorm single-path variants
shows spills falling as warps increase (1 warp: 46-4974 spilled values; 16 warps: 234-268).
The capture did not record BLOCK constants, so spill counts cannot be matched to N exactly.
**Conclusion:** at large blocks the single path is limited by per-thread register pressure; more
warps spread the row over more registers. The autotuner picked 16 warps for N = 32768.

## Experiment 4: attention tile configuration (4x32x4096, bf16, forced)

Best and worst valid tiles:

| D | best (BM/BN/warps/stages) | TFLOPS | worst | TFLOPS |
|---|---|---|---|---|
| 64 | 128/64/4/2 | 159.1 | 64/32/8/2 | 49.2 |
| 128 | 128/64/8/3 | 153.4 | 128/128/4/3 | 2.8 |

A 3x spread at D = 64 and a ~55x collapse at D = 128. 28 of 76 compiled attention variants
spill; a plausible reading is that a 128 x 128 fp32 accumulator (64 KiB) over 4 warps needs
~128 registers per thread for the accumulator alone, but variants were not matched to tiles. **Conclusion:** tile choice dominates
attention performance, which is why autotuning is part of the kernel.

## Kernel traces at a selected shape (N = 8192, fp16)

| Call | Kernel us | Block | Regs/thread | Est. occupancy |
|---|---|---|---|---|
| KernelForge softmax | 606 | 256 threads | 48 | 63% |
| torch.softmax | 1828 | 1024 threads | 35 | 50% |
| KernelForge layer_norm | 609 | 256 threads | 64 | 50% |
| F.layer_norm | 981 | 32x4 threads | 31 | 100% |

At N = 8192 the KernelForge kernels run at the copy ceiling while PyTorch's do not; occupancy is
not what separates them (F.layer_norm has the highest estimated occupancy and is slower).
Without counters, the cause of PyTorch's gap is not established.

## Where we lose

| Case | Evidence | Status |
|---|---|---|
| Softmax N >= 65536 (0.62-0.74 of BW_copy) | N > 65536: looped path reads twice while the registered byte count assumes one read (physical traffic ~1.5x counted). N = 65536: single path; 30 of 110 softmax variants spill, but variants were not matched to BLOCK | Looped: explained by byte accounting. N = 65536: probable spills, not confirmed |
| LayerNorm N >= 65536 (0.78-0.92x F.layer_norm) | Looped path reads the row three times (two-pass centered statistics chosen for accuracy, DECISIONS.md) | Explained by design: an accuracy-for-bandwidth trade |
| LayerNorm irregular N (1000, 4097, 31337; down to 0.17x F.layer_norm) | Not alignment (Exp. 2). Same block and warps as N = 32768 yet 17x slower. Spill-heavy variants (Exp. 3). Leading hypothesis: N is a runtime argument, so Triton cannot prove 16-element mask alignment when N % 16 != 0 and emits per-element masked, spilled code. Softmax, which compiles N as a constant, does not show this. | **Not understood**: hypothesis untested (needs a constexpr-N LayerNorm variant) |
| Attention D = 128 (0.77-0.86x SDPA-flash) | Tile sweep: the best D = 128 tile (153 TFLOPS) is below the best D = 64 tile (159), large accumulators spill, and SDPA-flash gains more from D = 128 than KernelForge does | Not understood beyond tile/spill evidence; no counter data |
| Attention launches under the SW power cap | Telemetry throttle reason 0x4 at ~400 W | Environmental, affects both implementations |
