"""CPU-only timing orchestration and validated measurement records."""
import math
from collections.abc import Callable, Sequence
from statistics import median
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from kernelforge import metrics


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Timing(RecordModel):
    ms_p20: float = Field(gt=0)
    ms_p50: float = Field(gt=0)
    ms_p80: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        """Reject unordered or nonpositive timing quantiles."""
        if not self.ms_p20 <= self.ms_p50 <= self.ms_p80:
            raise ValueError("Timing quantiles must be ordered")
        return self


class Environment(RecordModel):
    device: str = Field(min_length=1)
    cc: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    triton: str = Field(min_length=1)
    cuda: str = Field(min_length=1)
    driver: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class Telemetry(RecordModel):
    start: dict[str, JsonValue] = Field(default_factory=dict)
    end: dict[str, JsonValue] = Field(default_factory=dict)


class Autotune(RecordModel):
    best_config: dict[str, JsonValue] = Field(default_factory=dict)
    tuning_seconds: float | None = Field(default=None, ge=0)
    invalid_configs: list[dict[str, JsonValue]] = Field(default_factory=list)


class Measurement(Timing):
    schema_id: Literal["kernelforge.bench.v1"] = Field(
        default="kernelforge.bench.v1", alias="schema")
    op: Literal["softmax", "layer_norm", "causal_attention", "copy"]
    impl: str = Field(min_length=1)
    case: dict[str, JsonValue]
    launch_index: int = Field(ge=0, strict=True)
    bytes: int = Field(ge=0, strict=True)
    flops: int | None = Field(default=None, ge=0, strict=True)
    gbps: float = Field(ge=0)
    tflops: float | None = Field(default=None, ge=0)
    copy_gbps: float | None = Field(default=None, gt=0)
    ratio_to_copy: float | None = Field(default=None, ge=0)
    peak_mem_delta_bytes: int = Field(ge=0, strict=True)
    env: Environment
    gpu_telemetry: Telemetry
    autotune: Autotune
    baseline_call: str | None = None

    @model_validator(mode="after")
    def consistent_rates(self):
        """Reject derived rates inconsistent with recorded work and median timing."""
        expected = {"gbps": metrics.gbps(self.bytes, self.ms_p50),
                    "tflops": None if self.flops is None else metrics.tflops(self.flops, self.ms_p50),
                    "ratio_to_copy": None if self.copy_gbps is None else
                    metrics.ratio_to_copy(self.gbps, self.copy_gbps)}
        for name, value in expected.items():
            actual = getattr(self, name)
            if (value is None) != (actual is None):
                raise ValueError(f"{name} requires its accounting/baseline field")
            if value is not None and not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{name} differs from the recorded accounting and timing")
        return self


class Aggregate(Timing):
    n: Literal[3] = 3
    ms_min: float = Field(gt=0)
    ms_max: float = Field(gt=0)
    gbps: float = Field(ge=0)
    tflops: float | None = Field(default=None, ge=0)
    ratio_to_copy: float | None = Field(default=None, ge=0)
    launches: list[Measurement] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def consistent_summary(self):
        """Validate stored aggregates against all three underlying launch records."""
        expected = _launch_values(self.launches)
        for field, value in expected.items():
            actual = getattr(self, field)
            if (actual is None) != (value is None):
                raise ValueError(f"Aggregate {field} is inconsistent")
            if value is not None and not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Aggregate {field} is inconsistent")
        return self


class Timer(Protocol):
    def measure(self, operation: Callable[[], object]) -> Timing:
        """Time a callable and return millisecond quantiles, excluding tuning."""
        ...


def quantile(samples: Sequence[float], fraction: float):
    """Compute an inclusive, linearly interpolated quantile of positive timings."""
    if not samples or not 0 <= fraction <= 1:
        raise ValueError("Expected samples and a fraction in [0, 1]")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise ValueError("Timing samples must be finite and positive")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: Sequence[float]):
    """Summarize raw samples with the registered p20, p50, and p80 quantiles."""
    return Timing(ms_p20=quantile(samples, 0.2), ms_p50=quantile(samples, 0.5),
                  ms_p80=quantile(samples, 0.8))


def benchmark(operation, timer: Timer, *, op, impl, case, launch_index, env,
              byte_accounting, flop_accounting=None, copy_gbps=None,
              peak_mem_delta_bytes=0, gpu_telemetry=None, autotune=None, baseline_call=None):
    """Delegate timing and accounting to injected functions and validate the result."""
    byte_count = byte_accounting(case)
    flop_count = flop_accounting(case) if flop_accounting is not None else None
    timing = timer.measure(operation)
    bandwidth = metrics.gbps(byte_count, timing.ms_p50)
    return Measurement(
        **timing.model_dump(), op=op, impl=impl, case=case, launch_index=launch_index,
        bytes=byte_count, flops=flop_count, gbps=bandwidth,
        tflops=None if flop_count is None else metrics.tflops(flop_count, timing.ms_p50),
        copy_gbps=copy_gbps,
        ratio_to_copy=None if copy_gbps is None else metrics.ratio_to_copy(bandwidth, copy_gbps),
        peak_mem_delta_bytes=peak_mem_delta_bytes, env=env,
        gpu_telemetry=Telemetry() if gpu_telemetry is None else gpu_telemetry,
        autotune=Autotune() if autotune is None else autotune, baseline_call=baseline_call)


def _launch_values(records: Sequence[Measurement]):
    """Validate launch identity and compute the median, range, and derived rates."""
    if len(records) != 3:
        raise ValueError("Exactly three independent launch records are required")
    launches = [Measurement.model_validate(record.model_dump(by_alias=True)) for record in records]
    if {record.launch_index for record in launches} != {0, 1, 2}:
        raise ValueError("Launch indices must be 0, 1, and 2")
    first = launches[0]
    for record in launches[1:]:
        for field in ("op", "impl", "case", "bytes", "flops", "env", "copy_gbps", "baseline_call"):
            if getattr(record, field) != getattr(first, field):
                raise ValueError(f"Cannot aggregate differing {field}")
    medians = [record.ms_p50 for record in launches]
    mid = median(medians)
    bandwidth = metrics.gbps(first.bytes, mid)
    return {
        "ms_p20": median(record.ms_p20 for record in launches), "ms_p50": mid,
        "ms_p80": median(record.ms_p80 for record in launches),
        "ms_min": min(medians), "ms_max": max(medians), "gbps": bandwidth,
        "tflops": None if first.flops is None else metrics.tflops(first.flops, mid),
        "ratio_to_copy": None if first.copy_gbps is None else
        metrics.ratio_to_copy(bandwidth, first.copy_gbps),
    }


def aggregate_launches(records: Sequence[Measurement]):
    """Require three matching launches and preserve their evidence in a validated summary."""
    return Aggregate(**_launch_values(records),
                     launches=sorted(records, key=lambda record: record.launch_index))
