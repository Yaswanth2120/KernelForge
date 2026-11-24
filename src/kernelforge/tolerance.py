"""Error metrics and the frozen PyTorch-relative accuracy rule."""
import math
from dataclasses import dataclass

import torch

RELATIVE_FLOOR = 1e-8
ERROR_MULTIPLIER = 2.0
EPS_MULTIPLIER = 2.0


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    max_relative: float
    nonfinite_match: bool


@dataclass(frozen=True)
class AccuracyResult:
    kernel: ErrorMetrics
    baseline: ErrorMetrics
    allowed_max_abs: float
    passed: bool


def error_metrics(actual, reference, relative_floor=RELATIVE_FLOOR):
    """Measure finite entries and require matching NaN locations and signed infinities."""
    if actual.shape != reference.shape or actual.device != reference.device:
        raise ValueError("Tensors must have identical shapes and devices")
    if not math.isfinite(relative_floor) or relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    actual, reference = actual.float(), reference.float()
    finite = torch.isfinite(reference)
    match = torch.isfinite(actual) == finite
    match &= finite | (torch.isnan(actual) & torch.isnan(reference)) | (actual == reference)
    nonfinite_match = bool(match.all().item())
    if not nonfinite_match:
        return ErrorMetrics(float('inf'), float('inf'), float('inf'), False)
    errors = (actual[finite] - reference[finite]).abs()
    relative_mask = reference[finite].abs() >= relative_floor
    relative = errors[relative_mask] / reference[finite][relative_mask].abs()
    return ErrorMetrics(
        errors.max().item() if errors.numel() else 0.0,
        errors.mean().item() if errors.numel() else 0.0,
        relative.max().item() if relative.numel() else 0.0,
        True,
    )


def compare_accuracy(kernel, baseline, reference, dtype):
    """Apply max(2 * PyTorch max error, 2 * dtype epsilon * reference magnitude)."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Accuracy policy supports fp16, bf16, and fp32")
    kernel_error = error_metrics(kernel, reference)
    baseline_error = error_metrics(baseline, reference)
    finite_reference = reference.float()[torch.isfinite(reference.float())]
    magnitude = finite_reference.abs().max().item() if finite_reference.numel() else 0.0
    floor = EPS_MULTIPLIER * torch.finfo(dtype).eps * magnitude
    allowed = max(ERROR_MULTIPLIER * baseline_error.max_abs, floor)
    passed = (kernel_error.nonfinite_match and baseline_error.nonfinite_match
              and kernel_error.max_abs <= allowed)
    return AccuracyResult(kernel_error, baseline_error, allowed, passed)
