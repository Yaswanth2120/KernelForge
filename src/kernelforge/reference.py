"""Small, materialized PyTorch references, never used by custom GPU operations."""
import math

import torch


def _precision(x, dtype):
    """Promote the original input to the requested reference precision."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Reference precision must be fp32 or fp64")
    return x.to(dtype)


def softmax(x, dim=-1, *, dtype=torch.float32):
    """Compute stable last-dimension softmax in reference precision."""
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("Expected a nonempty last dimension")
    if dim not in (-1, x.ndim - 1):
        raise NotImplementedError("Only the last dimension is supported")
    values = _precision(x, dtype)
    numerator = torch.exp(values - values.amax(dim=-1, keepdim=True))
    return numerator / numerator.sum(dim=-1, keepdim=True)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5,
               return_stats=False, *, dtype=torch.float32):
    """Compute centered population variance over the trailing normalized dimensions."""
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("normalized_shape must contain positive sizes")
    if len(shape) > x.ndim or tuple(x.shape[-len(shape):]) != shape:
        raise ValueError("normalized_shape must match the trailing input dimensions")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    for parameter in (weight, bias):
        if parameter is not None and tuple(parameter.shape) != shape:
            raise ValueError("Affine parameter shape must match normalized_shape")
    values = _precision(x, dtype)
    dims = tuple(range(x.ndim - len(shape), x.ndim))
    mean = values.mean(dim=dims, keepdim=True)
    centered = values - mean
    variance = centered.square().mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    output = centered * rstd
    if weight is not None:
        output = output * weight.to(dtype)
    if bias is not None:
        output = output + bias.to(dtype)
    if return_stats:
        leading = x.shape[:-len(shape)]
        return output, mean.reshape(leading), rstd.reshape(leading)
    return output


def causal_attention(q, k, v, sm_scale=None, return_lse=False, *, dtype=torch.float32):
    """Materialize causal scores for small correctness cases in reference precision."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical (B, H, S, D) shapes")
    if q.shape[-1] not in (32, 64, 128) or q.shape[-2] < 1:
        raise ValueError("Expected S >= 1 and D in {32, 64, 128}")
    scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    if not math.isfinite(scale):
        raise ValueError("sm_scale must be finite")
    query, key, value = (_precision(x, dtype) for x in (q, k, v))
    scores = (query @ key.transpose(-1, -2)) * scale
    positions = torch.arange(q.shape[-2], device=q.device)
    mask = positions[:, None] >= positions[None, :]
    scores = scores.masked_fill(~mask, -torch.inf)
    output = softmax(scores, dtype=dtype) @ value
    if return_lse:
        return output, torch.logsumexp(scores, dim=-1)
    return output
