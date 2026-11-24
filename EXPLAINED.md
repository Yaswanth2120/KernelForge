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
