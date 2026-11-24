"""Registered operation accounting; rates use decimal GB and TFLOP units."""
import math


def _counts(*values):
    """Reject negative or noninteger element and byte counts."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Counts must be nonnegative integers")


def softmax_bytes(m, n, elem_size):
    """Count one input read and one output write."""
    _counts(m, n, elem_size)
    return 2 * m * n * elem_size


def layer_norm_bytes(m, n, elem_size, weight_elem=0, bias_elem=0, return_stats=False):
    """Include affine parameter reads and optional fp32 mean/rstd writes."""
    _counts(m, n, elem_size, weight_elem, bias_elem)
    return 2 * m * n * elem_size + n * (weight_elem + bias_elem) + (8 * m if return_stats else 0)


def attention_flops(b, h, s, d):
    """Use the frozen lower-triangle approximation, 2 * B * H * S² * D."""
    _counts(b, h, s, d)
    return 2 * b * h * s**2 * d


def _rate(count, milliseconds, divisor):
    """Convert work per millisecond into the requested decimal rate."""
    _counts(count)
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError("Time must be finite and positive")
    return count / milliseconds / divisor


def gbps(byte_count, milliseconds):
    """Convert bytes and milliseconds to decimal gigabytes per second."""
    return _rate(byte_count, milliseconds, 1e6)


def tflops(flop_count, milliseconds):
    """Convert FLOPs and milliseconds to teraFLOPs per second."""
    return _rate(flop_count, milliseconds, 1e9)


def ratio_to_copy(bandwidth, copy_bandwidth):
    """Divide effective bandwidth by the measured copy bandwidth."""
    if (not math.isfinite(bandwidth) or bandwidth < 0
            or not math.isfinite(copy_bandwidth) or copy_bandwidth <= 0):
        raise ValueError("Bandwidth must be finite, with a positive copy baseline")
    return bandwidth / copy_bandwidth


def percent_of_peak(throughput, peak=312.0):
    """Express throughput relative to the context-only dense Tensor Core peak."""
    return 100 * ratio_to_copy(throughput, peak)
