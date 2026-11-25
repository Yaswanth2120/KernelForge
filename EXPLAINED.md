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

## `src/kernelforge/cli.py:pytest_args`

Builds the pytest command for `kernelforge test`: the CPU suite by default, or the non-slow GPU suite with `--gpu`. `--op` narrows the run to one op's test file. Slow int64 tests are left out because they need about 25 GiB of GPU memory.

## `src/kernelforge/cli.py:_test`

Runs that pytest command in a child process and converts its exit code to the CLI's convention: 0 means pass, 1 means a test failed, and 2 covers anything else pytest reports, such as collection errors.

## `src/kernelforge/env.py:command_output`

Capture a bounded external diagnostic without failing offline. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/env.py:doctor`

Collect environment facts without downloading or initializing kernels. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/env.py:timing_source`

Return the exact installed timer source for manual review on the GPU host. Its result is used by the calling interface or diagnostic; it makes no kernel correctness claim.

## `src/kernelforge/ops/_common.py:is_triton_available`

Check the CUDA runtime and lazily import Triton. It can check the input tensor’s device explicitly, and returns false when the supported runtime is unavailable.

## `src/kernelforge/ops/_common.py:require_gpu`

Reject calls without a supported runtime. The optional device is forwarded to the single availability check, while the default call preserves the CPU-safe error behavior.

## `src/kernelforge/ops/attention.py:causal_attention`

Computes softmax(q kᵀ · scale) v where query i only sees keys j <= i, for (B, H, S, D) fp16/bf16 tensors, without ever building the S x S score matrix. `return_lse=True` also returns each row's log-sum-exp in fp32. It never calls PyTorch attention.

## `src/kernelforge/ops/layernorm.py:layer_norm`

Normalizes the trailing `normalized_shape` dims of `x` to zero mean and unit variance, then applies optional `weight` and `bias`. Statistics are computed in fp32 (the row sum in fp64), the output keeps `x`'s dtype and shape, and `return_stats=True` also returns per-row fp32 `mean` and `rstd`. It never calls `torch.nn.functional.layer_norm`.

## `src/kernelforge/ops/softmax.py:softmax`

Normalize the final dimension with fp32 accumulation and same-dtype output. An empty leading dimension returns without loading Triton kernels; a nonempty CUDA input dispatches under its own device context.

## `tests/cpu/test_foundation.py:test_import_without_triton`

Checks import without triton. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_unavailable_gpu`

Checks unavailable gpu. A failure raises an assertion or exception so regressions remain visible.

## `tests/cpu/test_foundation.py:test_unimplemented_layer_norm_on_supported_gpu`

This test checks that LayerNorm remains an explicit stub on a mocked supported runtime. Softmax now has its own interface contract tests because its intended behavior changed.

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

## `tests/cpu/test_harness.py:test_missing_schema_identity_is_rejected`

This test rejects a serialized record whose schema identity is absent. It prevents the parser from silently assigning a known schema to unidentified data.

## `src/kernelforge/ops/_common.py:softmax_grid`

Split logical row IDs over CUDA's first two grid dimensions. The GPU row mapper widens IDs before flattening this grid, and masks any padded final rows.

## `src/kernelforge/ops/_common.py:prune_softmax_configs`

Keep structurally useful configurations that compile and fit device resources. It records reasons and exposed resource metadata, rejects an entirely invalid search space, and leaves unexpected compiler errors visible.

## `src/kernelforge/ops/softmax.py:_prepare_input`

Validate tensor metadata and copy only when the final dimension is strided. Unit-stride leading views are preserved; other final-dimension layouts are made contiguous before dispatch.

## `src/kernelforge/ops/_softmax_kernels.py:_row_location`

Map a two-dimensional grid ID to an int64 logical row and input offset. Unflattening the row through leading shapes and strides supports transposed or broadcast views without assuming a single uniform row stride.

## `src/kernelforge/ops/_softmax_kernels.py:_prune`

Compile candidates without executing them and retain resource-fit evidence. The resulting diagnostic records describe resource fit, not numerical correctness, and need a real owner-run GPU session.

## `src/kernelforge/ops/_softmax_kernels.py:_prune_single`

Prune the row-in-registers autotune candidates. Its block is derived from N, and its compiler callback checks each surviving warp count.

## `src/kernelforge/ops/_softmax_kernels.py:_prune_looped`

Prune the online-normalizer autotune candidates. Each candidate supplies its own bounded tile width and warp count before timing.

## `src/kernelforge/ops/_softmax_kernels.py:_single_softmax`

Keep one padded row in a program and perform a stable fp32 reduction. Masked negative-infinity padding contributes zero to a finite row’s denominator, and masked stores preserve the original width.

## `src/kernelforge/ops/_softmax_kernels.py:_looped_softmax`

Compute an online normalizer in bounded tiles, then reread and normalize. It guards all-negative-infinity prefixes and carries a compensated fp32 sum, then rereads the row to write probabilities with int64 offsets.

## `src/kernelforge/ops/_softmax_kernels.py:launch_softmax`

Launch a path and retain its winning configuration for later GPU artifacts. The tuning key includes layout, dtype, and device, while winner metadata is recorded only after the launch returns.

## `src/kernelforge/ops/_softmax_kernels.py:compile_config`

Load a candidate's compiled handles so resource failures precede tuning. The pinned private handle initializer checks resources and exposes register, spill, and shared-memory metadata without running the candidate kernel.

## `tests/cpu/test_softmax_interface.py:test_valid_metadata_preserves_input`

This test checks valid metadata preserves input. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_leading_strides_are_preserved`

This test checks leading strides are preserved. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_last_dimension_copy`

This test checks last dimension copy. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_invalid_metadata`

This test checks invalid metadata. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_sparse_layout_rejected`

This test checks sparse layout rejected. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_cpu_tensor_rejected_with_mocked_runtime`

This test checks cpu tensor rejected with mocked runtime. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_device_specific_availability`

This test checks device specific availability. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_grid_dimensions`

This test checks grid dimensions. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_grid_rejects_invalid_counts`

This test checks grid rejects invalid counts. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:fake_config`

Describe a candidate without importing Triton. The plain object lets CPU policy tests exercise candidate selection without loading a CUDA compiler.

## `tests/cpu/test_softmax_interface.py:test_resource_pruning_and_evidence`

This test checks resource pruning and evidence. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_no_valid_configuration_reports_failure`

This test checks no valid configuration reports failure. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_unexpected_compiler_error_propagates`

This test checks unexpected compiler error propagates. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:test_fixed_block_pruning`

This test checks fixed block pruning. It checks host-side behavior using CPU tensors or fake compilation results, without executing Triton.

## `tests/cpu/test_softmax_interface.py:compile_candidate`

Simulate one expected resource failure and one successful compilation. The test checks that only recognized resource failures are skipped and that usable candidate metadata is retained.

## `tests/cpu/test_softmax_interface.py:fail`

Represent a compiler bug that must not be hidden as resource pruning. The pruning policy must propagate this exception so a compiler bug cannot masquerade as a rejected resource configuration.

## `tests/gpu/test_softmax.py:supported_gpu`

Skip GPU tests on hosts without the required NVIDIA runtime. A skipped case remains pending and is never evidence of kernel correctness.

## `tests/gpu/test_softmax.py:random_input`

Create reproducible low-precision inputs directly on the GPU. A fixed CUDA generator seed makes the original typed input reproducible for both the custom operation and its PyTorch comparisons.

## `tests/gpu/test_softmax.py:check_result`

Check the frozen error rule, output contract, input preservation, and bitwise repeatability. Failures print candidate, PyTorch, and allowed errors; repeated outputs are compared as bytes so matching NaN values can still be checked for determinism.

## `tests/gpu/test_softmax.py:test_dtype_shape_matrix`

This test checks dtype shape matrix. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_layouts`

This test checks layouts. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_extreme_scales_and_constants`

This test checks extreme scales and constants. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_infinities_nan_and_empty_prefix`

This test checks infinities nan and empty prefix. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_looped_nan_locations`

This test checks looped nan locations. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_empty_leading_dimensions`

This test checks empty leading dimensions. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_cuda_argument_errors`

This test checks cuda argument errors. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_autotune_metadata`

This test checks autotune metadata. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/test_softmax.py:test_int64_offsets_over_two_billion_elements`

This test checks int64 offsets over two billion elements. It is written for the owner’s NVIDIA session and has not been run here.

## `tests/gpu/conftest.py:pytest_sessionfinish`

Save configuration diagnostics only if a real softmax backend was loaded. CPU-only runs leave no artifact, and GPU artifacts retain pytest’s exit code so failures cannot be presented as a passing session.

## `tests/cpu/test_softmax_interface.py:test_unusable_runtime_returns_false`

This test checks that invalid devices and missing Triton produce a false availability result. It uses mocked runtime functions and never opens a GPU context.

## `tests/cpu/test_softmax_interface.py:unavailable`

This helper raises a selected fake environment error for the availability test. It lets the test check both device rejection and import failure without needing CUDA or Triton installed.

## `src/kernelforge/ops/_copy_kernels.py:_copy`

Each program copies one contiguous tile of `BLOCK` elements, masking the tail so any length works. Offsets are int64 so buffers larger than 2^31 elements are safe. It exists only to measure how fast the GPU can move bytes (BW_copy).

## `src/kernelforge/ops/_copy_kernels.py:triton_copy`

Launches `_copy` with one program per tile and returns the autotuned winner. The benchmark compares it with `torch.Tensor.copy_` and keeps whichever is faster as the bandwidth ceiling.

## `src/kernelforge/ops/_softmax_kernels.py:_key_arguments`

Builds the scalar kernel arguments that double as the autotune key (shape, strides, dtype, device). Sharing this function keeps the launch and the later lookup of its autotune record from disagreeing.

## `src/kernelforge/ops/_softmax_kernels.py:autotune_entry`

Looks up the pruning log and winner for the most recent launch on a tensor shaped like `x`. The benchmark runner copies it into each result record.

## `src/kernelforge/bench/suites.py:rows_for`

Returns the smallest row count M that makes an M x N input at least 512 MiB, about 13x the A100's L2 cache, so timings reflect HBM rather than cache.

## `src/kernelforge/bench/suites.py:softmax_sweep`

Lists every benchmarked softmax shape (small, threshold, large, and irregular N for each dtype) and marks the pre-registered pass/fail cases as `selected`.

## `src/kernelforge/bench/suites.py:copy_cases`

Describes the single 1 GiB fp32 buffer used for the copy baseline; bytes are counted as one read plus one write.

## `src/kernelforge/bench/harness.py:case_key`

Identifies a measured configuration by operation, implementation, and case, ignoring which launch produced it, so the three launches can be grouped.

## `src/kernelforge/bench/harness.py:load_launch_records`

Reads every `launch_*.jsonl` file in a suite directory and validates each line against the result schema, so malformed or hand-edited records fail loudly.

## `src/kernelforge/bench/harness.py:group_launches`

Groups records per configuration and aggregates only groups with exactly three launches; other groups keep their records but no aggregate, so no ratio can be printed for them.

## `src/kernelforge/bench/gpu_runner.py:DoBenchTimer.measure`

Calls `triton.testing.do_bench` with quantiles 0.2/0.5/0.8 on an already-tuned callable and returns them as a `Timing`. do_bench flushes L2 before each timed call (see DECISIONS.md).

## `src/kernelforge/bench/gpu_runner.py:parse_telemetry`

Turns one `nvidia-smi` CSV line into a field dictionary, or an error entry if the line has the wrong shape.

## `src/kernelforge/bench/gpu_runner.py:sample_telemetry`

Queries SM clock, max SM clock, memory clock, power, temperature, and throttle reasons for GPU 0. It is called at the start and end of every launch.

## `src/kernelforge/bench/gpu_runner.py:clock_drop`

Flags a launch whose end sample shows the SM clock below 97% of its maximum or any throttle reason other than "idle". Missing telemetry is flagged too, because it cannot rule out a drop.

## `src/kernelforge/bench/gpu_runner.py:git_sha`

Returns the short commit SHA and appends `-dirty` when tracked source files differ from it, so a result never claims to come from code that did not run.

## `src/kernelforge/bench/gpu_runner.py:environment`

Captures device name, compute capability, torch/Triton/CUDA/driver versions, git SHA, and date for every result record.

## `src/kernelforge/bench/gpu_runner.py:launch_command`

Builds the command for one launch. Each of the three launches is a fresh Python process so compilation caches, allocator state, and clocks are independent.

## `src/kernelforge/bench/gpu_runner.py:run_suite`

Runs exactly three launches one after another and refuses any other count, because the registered protocol takes the median of three process medians.

## `src/kernelforge/bench/gpu_runner.py:copy_baseline`

Reads BW_copy from `copy_probe.json`. Benchmarks refuse to run without it, because every bandwidth ratio depends on it.

## `src/kernelforge/bench/gpu_runner.py:_first_call_seconds`

Times the first call, which compiles and autotunes. This time is recorded separately and excluded from benchmark timing.

## `src/kernelforge/bench/gpu_runner.py:_peak_memory_delta`

Resets PyTorch's peak-memory counter, runs one call, and reports the peak allocation above what was already allocated (the inputs). The output tensor is included in the delta.

## `src/kernelforge/bench/gpu_runner.py:copy_operations`

Builds the two copy implementations over the same buffer, together with their call strings and autotune metadata accessors.

## `src/kernelforge/bench/gpu_runner.py:softmax_operations`

Creates a seeded input for one case and returns KernelForge softmax and `torch.softmax` as callables over that same tensor.

## `src/kernelforge/bench/gpu_runner.py:softmax_accuracy`

Applies the frozen tolerance rule to the benchmark-sized input of launch 0, so the accuracy table covers the same shapes as the performance tables.

## `src/kernelforge/bench/gpu_runner.py:run_launch`

Measures every case once in the current process: tune, measure memory, time, then write all records with the launch's start/end telemetry to `launch_<i>.jsonl`.

## `src/kernelforge/bench/copy_probe.py:summarize`

Aggregates the three copy launches per implementation and takes the faster median as BW_copy. It refuses to summarize fewer than three launches.

## `src/kernelforge/bench/copy_probe.py:run`

Runs the copy suite's three launches on the GPU and writes the BW_copy summary JSON that benchmarks read.

## `src/kernelforge/report.py:render`

Builds the whole Markdown report from files under `reports/raw/` only, and returns how many pre-registered cases failed. Criteria are evaluated only for three-launch aggregates recorded on a device whose name contains "A100".

## `src/kernelforge/report.py:softmax_criteria`

Checks criterion 2 (ratio >= 0.80) on each pre-registered softmax case. Missing data prints "not measured" with its launch count, and non-A100 data prints "not evaluated".

## `src/kernelforge/report.py:where_we_lose`

Lists, separately, cases slower than PyTorch and sweep cases below the 0.80 ratio. It does not invent explanations; causes come from profiling evidence.

## `src/kernelforge/report.py:telemetry_section`

Tabulates each launch's start/end nvidia-smi samples and flags clock drops using the runner's rule.

## `src/kernelforge/report.py:plot_softmax`

Draws one panel per dtype of effective GB/s against N, with the KernelForge and torch lines and BW_copy as a dashed reference. It returns None if matplotlib is unavailable or there is no data.

## `src/kernelforge/cli.py:main`

Parses the command and delegates. Exit codes: 0 ok, 1 criterion failure, 2 usage error (for example `--launches` other than 3), 3 environment problem (no GPU, or BW_copy missing).

## `src/kernelforge/ops/layernorm.py:_normalized_size`

Checks that `normalized_shape` matches the input's trailing dimensions and returns their product N, the length of each row being normalized.

## `src/kernelforge/ops/layernorm.py:_check_affine`

Checks that weight or bias has shape `normalized_shape`, lives on the input's device, and has the input's dtype or fp32 (mixed precision). It returns a flat contiguous copy or view of length N.

## `src/kernelforge/ops/layernorm.py:_prepare`

Validates all arguments and flattens the input to an (M, N) matrix. `reshape` returns a view when the leading dims collapse to one row stride; otherwise it copies, and a strided trailing block is also copied so columns have unit stride.

## `src/kernelforge/ops/_layernorm_kernels.py:_prune_idle_warps`

Removes autotune candidates with more threads than padded columns, which would leave threads idle. Configs that run out of resources are skipped by Triton's autotuner itself.

## `src/kernelforge/ops/_layernorm_kernels.py:_row_id`

Turns the 2-D program grid into one int64 row number, so `row * row_stride` cannot overflow even with more than 2^31 elements.

## `src/kernelforge/ops/_layernorm_kernels.py:_row_mean`

Divides an fp64 row sum by N and rounds once to fp32. The sum is the only statistic that can cancel badly (many values summing to nearly zero); fp64 keeps it accurate, and constant rows stay exactly centered.

## `src/kernelforge/ops/_layernorm_kernels.py:_rstd`

Computes 1/sqrt(M2/N + eps) with IEEE round-to-nearest division and square root. Triton's default `/` and `sqrt` are fast approximations that were measured to be 2 ulp off.

## `src/kernelforge/ops/_layernorm_kernels.py:_store_outputs`

Multiplies the centered values by rstd, applies weight and bias in fp32 when present (the flags are compile-time, so absent ones cost nothing), and stores one tile of output in the input dtype.

## `src/kernelforge/ops/_layernorm_kernels.py:_single_layer_norm`

One program holds a whole row in registers: sum for the mean, subtract it, sum the squares of the centered values for the variance, then normalize. This is the two-pass centered method, so it never uses E[x^2] - E[x]^2.

## `src/kernelforge/ops/_layernorm_kernels.py:_looped_layer_norm`

For rows too long for one block, the program walks the row in tiles three times: once to sum for the mean, once to sum squared deviations, and once to write the output. Each lane keeps its own partial sums across tiles and one tree reduction combines them, which keeps rounding error small.

## `src/kernelforge/ops/_layernorm_kernels.py:launch_layer_norm`

Picks the single or looped path from N and the measured limit, passes dummy pointers for absent weight, bias, and stats (their code is compiled out), launches, and records the autotuned winner.

## `src/kernelforge/ops/attention.py:_prepare`

Validates shapes, dtypes, head dim (32/64/128), device, and the grid limit on B*H, and picks the default scale 1/sqrt(D). Batch, head, and sequence strides are passed through untouched; only a tensor whose last dim is strided is copied.

## `src/kernelforge/ops/_attention_kernels.py:_prune_blocks`

Keeps only autotune configs with BLOCK_N <= BLOCK_M, because the kernel splits keys at the start of the query block and that split must land on a key-block boundary. It also drops query blocks much larger than S, which would be mostly padding.

## `src/kernelforge/ops/_attention_kernels.py:_attend_blocks`

The online-softmax inner loop. For each key block it computes scores (already multiplied by scale·log2(e)), raises the running row maximum m if needed, rescales the old sum l and accumulator by 2^(m_old - m_new), and adds 2^(score - m_new) and its product with V. Only the diagonal stage applies the causal and sequence-end masks.

Why exp2: `exp(x) = 2^(x·log2 e)`, and the GPU has a fast native 2^x instruction. Folding log2(e) into the scale once means every exponential is a plain `exp2`, with no extra multiply per element.

Why online softmax works: if a later block raises the maximum from m to m', every earlier term exp(s - m) must become exp(s - m'), which is the same as multiplying the whole old sum (and accumulator) by exp(m - m'). So the row never needs to be seen all at once.

## `src/kernelforge/ops/_attention_kernels.py:_causal_attention`

One program handles BLOCK_M queries of one (batch, head). Keys before the query block are visible to every query in it, so that stage runs without masks. Keys inside the block's span get the j <= i mask. Keys after it are above the diagonal for every query in the block, so they are skipped entirely, which is about half the work. At the end it divides by l and stores the rows that exist.

## `src/kernelforge/ops/_attention_kernels.py:launch_causal_attention`

Launches one program per (query block, batch·head), records the autotuned winner, and, when a new tuning happened, which configs were valid or invalid (for example, too much shared memory at D=128).

## `src/kernelforge/ops/_attention_kernels.py:_is_failed`

Triton records a config that failed to compile or launch with infinite time; this turns that into an "invalid" status.

## `src/kernelforge/bench/harness.py:_aggregate_or_none`

Aggregates a group only if it has exactly three launches that agree on every identity field; otherwise it returns None, so the report prints "not aggregated" instead of a number or a crash.

## `src/kernelforge/bench/gpu_runner.py:_suite` and `_cases`

Map a suite name to its cases, implementations, byte/FLOP accounting, and accuracy check, so `run_launch` has one loop for every operation.

## `src/kernelforge/bench/gpu_runner.py:_sdpa`

Calls PyTorch's `scaled_dot_product_attention` with `is_causal=True`, optionally inside `sdpa_kernel(...)` to force a backend (FlashAttention for the criterion, math for memory scaling).

## `src/kernelforge/bench/gpu_runner.py:default_sdpa_kernel`

Uses `torch.profiler` to name the CUDA kernel PyTorch's default backend actually ran, instead of guessing which backend it picked. The name is stored as metadata, not as part of the measurement's identity, because the profiler occasionally captures nothing.

## `src/kernelforge/bench/gpu_runner.py:attention_operations`

Builds seeded q, k, v for one case and returns KernelForge, SDPA forced to FlashAttention, and SDPA's default backend as callables over the same tensors.

## `src/kernelforge/bench/gpu_runner.py:attention_accuracy`

Applies the frozen tolerance rule on batch 0, heads 0-1 of a benchmark input, because the fp32 reference materializes the full S x S matrix.

## `src/kernelforge/bench/gpu_runner.py:attention_memory` and `_attention_memory_case`

Measure peak extra memory for KernelForge, SDPA-flash, and SDPA-math as S grows, recording an out-of-memory failure as a result. Fused kernels grow linearly with S; the math backend grows with S².

## `src/kernelforge/bench/gpu_runner.py:attention_bytes` and `attention_flops`

FLOPs are the frozen 2·B·H·S²·D (lower triangle only). Bytes count q, k, v read and o written; they are informational because attention is judged in TFLOPS.

## `src/kernelforge/report.py:attention_criteria`

Criterion 4: for each pre-registered shape, KernelForge TFLOPS divided by SDPA-FlashAttention TFLOPS (both medians of three launches) must be at least 0.50, and only A100 records are evaluated.

## `src/kernelforge/report.py:attention_sweep`, `attention_accuracy_section`, `attention_memory_section`, `attention_autotune_section`, `attention_losses`

Render the attention tables: TFLOPS with % of peak and % of SDPA-flash, accuracy, memory vs S (with a log-log plot), winning tiles and invalid configs, and the cases below SDPA-flash.

## `src/kernelforge/report.py:profiling_section`

Prints the profiling numbers (ncu outcome, roofline ridge point, kernel traces, experiment tables, spill summary) straight from `reports/raw/profiling/`; the interpretation lives in docs/PROFILING.md.

## `scripts/measure_path_threshold.py`

Times an op's single-block and looped paths across N in three fresh processes and applies a rule fixed in DECISIONS.md before measuring, to choose where the op switches paths.

## `scripts/profile_kernels.py`

Collects profiling evidence without hardware counters: an ncu probe, torch.profiler traces at a selected shape, Triton compile metadata, and three controlled experiments (row alignment, forced warps, forced attention tiles).
