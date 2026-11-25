"""CPU-safe causal attention interface; the GPU kernel is imported only for a CUDA call."""
import math

from kernelforge.errors import GPUUnavailableError

from ._common import require_gpu

HEAD_DIMS = (32, 64, 128)
MAX_GRID_Y = 65535  # CUDA's limit on the second grid dimension, which indexes batch * heads


def _prepare(q, k, v, sm_scale):
    """Validate (B, H, S, D) fp16/bf16 inputs; copy only a tensor whose last dim is strided."""
    import torch

    tensors = (q, k, v)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("causal_attention expects torch.Tensor q, k, v")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical (B, H, S, D) shapes")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("causal_attention supports fp16 or bf16, identical for q, k, v")
    if q.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"head dim D must be one of {HEAD_DIMS}")
    if q.shape[2] < 1:
        raise ValueError("sequence length S must be >= 1")
    if k.device != q.device or v.device != q.device:
        raise ValueError("q, k, v must be on the same device")
    if q.shape[0] * q.shape[1] > MAX_GRID_Y:
        raise ValueError(f"B * H must be <= {MAX_GRID_Y}")
    scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else float(sm_scale)
    if not math.isfinite(scale):
        raise ValueError("sm_scale must be finite")
    return tuple(t if t.stride(-1) == 1 else t.contiguous() for t in tensors), scale


def causal_attention(q, k, v, sm_scale=None, return_lse=False):
    """Causal self-attention softmax(q k^T * scale, j <= i) v without materializing S x S."""
    device = q.device if getattr(q, "is_cuda", False) else None
    require_gpu(device)
    import torch

    (q, k, v), scale = _prepare(q, k, v, sm_scale)
    if not q.is_cuda:
        raise GPUUnavailableError("causal_attention requires NVIDIA CUDA input tensors")
    batch, heads, seq, _ = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device) \
        if return_lse else None
    if batch * heads > 0:
        from ._attention_kernels import launch_causal_attention

        with torch.cuda.device(q.device):
            launch_causal_attention(q, k, v, out, lse, scale)
    return (out, lse) if return_lse else out
