"""CPU-safe LayerNorm interface; the GPU implementation is imported only for a CUDA call."""
import math

from kernelforge.errors import GPUUnavailableError

from ._common import require_gpu, softmax_grid

# Provisional: mirrors the measured softmax crossover (DECISIONS.md); not a LayerNorm measurement.
LAYER_NORM_SINGLE_BLOCK_LIMIT = 65536


def _normalized_size(x, normalized_shape):
    """Validate the trailing normalized dims and return their flattened size N."""
    shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("normalized_shape must contain positive integer sizes")
    if len(shape) > x.ndim or tuple(x.shape[x.ndim - len(shape):]) != shape:
        raise ValueError("normalized_shape must match the trailing input dimensions")
    return shape, math.prod(shape)


def _check_affine(parameter, name, shape, x):
    """Affine tensors match normalized_shape and use x's dtype or fp32 (mixed precision)."""
    import torch

    if parameter is None:
        return None
    if not isinstance(parameter, torch.Tensor) or tuple(parameter.shape) != shape:
        raise ValueError(f"{name} must be a tensor with shape normalized_shape")
    if parameter.dtype not in (x.dtype, torch.float32):
        raise TypeError(f"{name} must have the input dtype or fp32")
    if parameter.device != x.device:
        raise ValueError(f"{name} must be on the input's device")
    return parameter.reshape(-1).contiguous()


def _prepare(x, normalized_shape, weight, bias, eps):
    """Validate arguments and return (rows view or copy, N, shape, weight, bias)."""
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("layer_norm expects a torch.Tensor")
    if x.layout != torch.strided:
        raise NotImplementedError("layer_norm supports strided tensor layouts only")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("layer_norm supports fp16, bf16, and fp32")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    shape, n = _normalized_size(x, normalized_shape)
    weight = _check_affine(weight, "weight", shape, x)
    bias = _check_affine(bias, "bias", shape, x)
    leading = x.shape[:x.ndim - len(shape)]
    # Flatten the trailing block to N. reshape returns a view when the leading dims collapse to
    # one row stride and the trailing block is contiguous; otherwise it copies (documented).
    rows = x.reshape(math.prod(leading), n)
    if n > 1 and rows.stride(1) != 1:
        rows = rows.contiguous()
    return rows, leading, weight, bias


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, return_stats=False):
    """Normalize the trailing dims with fp32 statistics; output keeps x's dtype and shape."""
    device = x.device if getattr(x, "is_cuda", False) else None
    require_gpu(device)
    import torch

    rows, leading, weight, bias = _prepare(x, normalized_shape, weight, bias, eps)
    if not x.is_cuda:
        raise GPUUnavailableError("layer_norm requires an NVIDIA CUDA input tensor")
    output = torch.empty(rows.shape, dtype=x.dtype, device=x.device)
    mean = rstd = None
    if return_stats:
        mean = torch.empty(rows.shape[0], dtype=torch.float32, device=x.device)
        rstd = torch.empty_like(mean)
    if rows.shape[0] > 0:
        from ._layernorm_kernels import launch_layer_norm

        with torch.cuda.device(x.device):
            launch_layer_norm(rows, output, weight, bias, mean, rstd, float(eps),
                              softmax_grid(rows.shape[0]), LAYER_NORM_SINGLE_BLOCK_LIMIT)
    output = output.reshape(x.shape)
    if return_stats:
        return output, mean.reshape(leading), rstd.reshape(leading)
    return output
