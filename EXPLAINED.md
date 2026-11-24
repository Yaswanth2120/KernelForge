# Function guide

## `src/kernelforge/ops/_interpreter_probe.py:main`

Check masked indexing on CPU tensors and print the observed result. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/_interpreter_probe.py:copy_tail`

Supports the surrounding interpreter probe check. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/bench/suites.py:selected_rows`

Use the smallest row count meeting the registered input size. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/bench/suites.py:selected_attention`

Enumerate the frozen attention Cartesian product. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/bench/suites.py:check_prereg`

Reject edits to the registered protocol or executable suite definitions. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/cli.py:main`

Parse commands and delegate to CPU-safe application functions. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/env.py:command_output`

Capture a bounded external diagnostic without failing offline. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/env.py:doctor`

Collect environment facts without downloading or initializing kernels. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/env.py:timing_source`

Return the exact installed timer source for manual review on the GPU host. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/_common.py:is_triton_available`

Check the CUDA runtime and lazily import Triton. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/_common.py:require_gpu`

Reject calls without a supported runtime. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/attention.py:causal_attention`

Public interface reserved for the upcoming kernel implementation. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/layernorm.py:layer_norm`

Public interface reserved for the upcoming kernel implementation. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/softmax.py:softmax`

Public interface reserved for the upcoming kernel implementation. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `tests/cpu/test_foundation.py:test_import_without_triton`

Checks import without triton. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_unavailable_gpu`

Checks unavailable gpu. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_supported_gpu_stub`

Checks supported gpu stub. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_registered_suites`

Checks registered suites. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_protocol_drift`

Checks protocol drift. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_suite_drift`

Checks suite drift. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_offline_doctor`

Checks offline doctor. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_missing_command`

Checks missing command. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_cli_prereg`

Checks cli prereg. A failure raises an assertion or exception so regressions remain visible.

# CPU correctness and measurement functions

## `src/kernelforge/reference.py:_precision`

Promote the original input to the requested reference precision. The conversion preserves the already rounded input values so comparisons use identical data.

## `src/kernelforge/reference.py:softmax`

Compute stable last-dimension softmax in reference precision. Subtracting the maximum prevents overflow, while all-negative-infinity rows retain PyTorch’s NaN behavior.

## `src/kernelforge/reference.py:layer_norm`

Compute centered population variance over the trailing normalized dimensions. It applies optional weight and bias independently and can return mean and reciprocal standard deviation in the chosen reference precision.

## `src/kernelforge/reference.py:causal_attention`

Materialize causal scores for small correctness cases in reference precision. It includes the diagonal in the mask and optionally returns natural-log LSE; its quadratic storage limits it to small correctness cases.

## `src/kernelforge/tolerance.py:error_metrics`

Measure finite entries and require matching NaN locations and signed infinities. Small reference values are excluded only from relative error, while every finite entry contributes to absolute error.

## `src/kernelforge/tolerance.py:compare_accuracy`

Apply max(2 * PyTorch max error, 2 * dtype epsilon * reference magnitude). Both outputs must also match reference nonfinite locations; the returned object explains the threshold and pass/fail result.

## `src/kernelforge/metrics.py:_counts`

Reject negative or noninteger element and byte counts. Python integers keep large tensor accounting exact without introducing floating-point rounding.

## `src/kernelforge/metrics.py:softmax_bytes`

Count one input read and one output write. Leading dimensions are already flattened into the supplied row count.

## `src/kernelforge/metrics.py:layer_norm_bytes`

Include affine parameter reads and optional fp32 mean/rstd writes. Absent affine parameters contribute zero bytes, and returned statistics add eight bytes per row.

## `src/kernelforge/metrics.py:attention_flops`

Use the frozen lower-triangle approximation, 2 * B * H * S² * D. The expression intentionally preserves the pre-registered approximation rather than adding an exact diagonal correction.

## `src/kernelforge/metrics.py:_rate`

Convert work per millisecond into the requested decimal rate. A nonpositive or nonfinite time is rejected so invalid rates cannot enter reports.

## `src/kernelforge/metrics.py:gbps`

Convert bytes and milliseconds to decimal gigabytes per second. The millisecond and decimal-gigabyte conversions combine into division by one million.

## `src/kernelforge/metrics.py:tflops`

Convert FLOPs and milliseconds to teraFLOPs per second. The millisecond and teraFLOP conversions combine into division by one billion.

## `src/kernelforge/metrics.py:ratio_to_copy`

Divide effective bandwidth by the measured copy bandwidth. The baseline must be measured separately and positive; this helper supplies no hardware assumptions.

## `src/kernelforge/metrics.py:percent_of_peak`

Express throughput relative to the context-only dense Tensor Core peak. The default denominator is the 312 TFLOPS datasheet context value, not a pass/fail threshold.

## `src/kernelforge/bench/harness.py:quantile`

Compute an inclusive, linearly interpolated quantile of positive timings. Sorting and interpolation at position (n−1)q make the result deterministic, including for a single sample.

## `src/kernelforge/bench/harness.py:summarize`

Summarize raw samples with the registered p20, p50, and p80 quantiles. The returned timing model also checks positivity, finiteness, and ordering.

## `src/kernelforge/bench/harness.py:benchmark`

Delegate timing and accounting to injected functions and validate the result. The caller supplies tuning, memory, telemetry, and environment evidence; the harness only computes rates and builds the record.

## `src/kernelforge/bench/harness.py:_launch_values`

Validate launch identity and compute the median, range, and derived rates. It rejects incompatible environments or duplicate launch indices and computes rates from the median of medians.

## `src/kernelforge/bench/harness.py:aggregate_launches`

Require three matching launches and preserve their evidence in a validated summary. The retained launches are sorted by index, while process isolation itself remains the runner’s responsibility.

## `src/kernelforge/bench/harness.py:ordered`

Reject unordered or nonpositive timing quantiles. Returning the validated model lets Pydantic finish constructing a timing or measurement record.

## `src/kernelforge/bench/harness.py:consistent_rates`

Reject derived rates inconsistent with recorded work and median timing. Optional ratios and throughput require their corresponding baseline or work fields instead of accepting unsupported values.

## `src/kernelforge/bench/harness.py:consistent_summary`

Validate stored aggregates against all three underlying launch records. This prevents a deserialized summary from disagreeing with its retained source evidence.

## `src/kernelforge/bench/harness.py:measure`

Time a callable and return millisecond quantiles, excluding tuning. Real GPU adapters and fake CPU timers must obey the same small interface.

## `tests/cpu/test_reference.py:test_softmax_fp64`

This test checks softmax fp64. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_softmax_promotes_same_inputs`

This test checks softmax promotes same inputs. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_softmax_special_rows_and_strides`

This test checks softmax special rows and strides. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_layer_norm_fp64`

This test checks layer norm fp64. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_layer_norm_scales_and_mixed_affine`

This test checks layer norm scales and mixed affine. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_attention_fp64`

This test checks attention fp64. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_attention_promotes_strided_inputs`

This test checks attention promotes strided inputs. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_layer_norm_nan_and_singleton`

This test checks layer norm nan and singleton. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_reference.py:test_reference_argument_errors`

This test checks reference argument errors. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_torch_fake_passes_and_broken_fake_fails`

This test checks torch fake passes and broken fake fails. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_hand_computed_errors`

This test checks hand computed errors. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_zero_reference_and_empty`

This test checks zero reference and empty. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_nonfinite_policy`

This test checks nonfinite policy. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_bad_baseline_cannot_make_threshold_infinite`

This test checks bad baseline cannot make threshold infinite. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_policy_boundary`

This test checks policy boundary. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_tolerance.py:test_tolerance_arguments`

This test checks tolerance arguments. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_metrics.py:test_accounting_hand_calculations`

This test checks accounting hand calculations. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_metrics.py:test_invalid_counts`

This test checks invalid counts. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_metrics.py:test_invalid_rates`

This test checks invalid rates. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:measurement`

Build a synthetic measurement record with optional test-specific fields. It uses a clearly labeled CPU fixture environment so the values cannot be mistaken for real GPU evidence.

## `tests/cpu/test_harness.py:test_known_quantiles_and_timer_delegation`

This test checks known quantiles and timer delegation. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_three_launch_aggregation_and_range`

This test checks three launch aggregation and range. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_aggregation_with_work_and_copy_baseline`

This test checks aggregation with work and copy baseline. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_json_schema_roundtrip`

This test checks json schema roundtrip. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_invalid_records`

This test checks invalid records. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_requires_three_launches`

This test checks requires three launches. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_rejects_duplicate_launches`

This test checks rejects duplicate launches. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_rejects_incompatible_launches`

This test checks rejects incompatible launches. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_rejects_mixed_environments`

This test checks rejects mixed environments. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_invalid_sample_timings`

This test checks invalid sample timings. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_single_sample_and_endpoints`

This test checks single sample and endpoints. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:test_aggregate_json_cannot_forge_launch_count_or_rates`

This test checks aggregate json cannot forge launch count or rates. It uses explicit expected values or invalid inputs so a regression fails visibly without a GPU.

## `tests/cpu/test_harness.py:__init__`

Build a synthetic measurement record with optional test-specific fields. The fake timer retains synthetic samples and tracks invocations without touching CUDA.

## `tests/cpu/test_harness.py:measure`

Build a synthetic measurement record with optional test-specific fields. It calls the operation once to check delegation and summarizes only the supplied synthetic timings.
