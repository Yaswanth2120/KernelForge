# CPU correctness and benchmark interfaces

`kernelforge.reference` provides `softmax`, `layer_norm`, and `causal_attention`.
Pass `dtype=torch.float64` for CPU self-tests; the default is fp32. The attention reference
materializes scores and should be used only for small correctness cases.

`compare_accuracy(candidate, torch_output, reference_output, input_dtype)` returns a structured
result with both error summaries, the maximum absolute error threshold, and a pass flag.
The threshold is `max(2 * torch_error, 2 * dtype_epsilon * max(abs(reference)))`.
Relative errors ignore reference magnitudes below 1e-8. NaN locations and signed infinities
must match exactly in both candidate and baseline; empty comparisons have zero error.

Metrics follow [PREREG.md](../PREREG.md). Rates use decimal units and millisecond timings.
LayerNorm counts only present affine parameters and includes fp32 statistics when requested.
Attention FLOPs use the registered lower-triangle approximation, without a diagonal correction.

The benchmark harness accepts a `Timer` with `measure(operation) -> Timing` and injected
byte/FLOP accounting functions. Setup, autotuning, synchronization, memory measurement,
and telemetry collection belong to the caller and future real GPU runner. The CPU harness
never executes CUDA or imports Triton. Synthetic timings exist only in tests.

Serialize each `Measurement` with `model_dump_json(by_alias=True)` and load it with
`Measurement.model_validate_json(text)`. Validation rejects invalid schema IDs, missing fields,
nonfinite/nonpositive timings, negative counts, unordered quantiles, and inconsistent rates.
Optional FLOPs and copy baselines must accompany their derived throughput and ratio values.

`aggregate_launches(records)` requires three compatible records indexed 0, 1, and 2. It preserves
all source records, reports the median of the three medians, and includes the min/max medians.
Per-launch p20/p80 values are combined by their medians; these are not pooled quantiles.
Launching three independent processes must be enforced by the GPU runner, not inferred from indices.
No real GPU timer, memory collection, or performance report is implemented yet.

The pinned CPU FlashAttention backend on macOS produced NaNs for zero attention scale;
fp64 correctness comparisons explicitly use SDPA's math backend. See DECISIONS.md for the
observed case. This does not change the required GPU FlashAttention performance baseline.
