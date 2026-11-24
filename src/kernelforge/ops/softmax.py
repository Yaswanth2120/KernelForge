"""CPU-safe interface; GPU implementation is imported only for a CUDA invocation."""
import operator

from kernelforge.errors import GPUUnavailableError

from ._common import require_gpu, softmax_grid

# Provisional until measured on the owner's GPU; not a performance claim.
SOFTMAX_SINGLE_BLOCK_LIMIT = 8192


def _prepare_input(x, dim):
    """Validate tensor metadata and copy only when the final dimension is strided."""
    import torch

    if not isinstance(x, torch.Tensor):
        raise TypeError("softmax expects a torch.Tensor")
    if x.layout != torch.strided:
        raise NotImplementedError("softmax supports strided tensor layouts only")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("softmax supports fp16, bf16, and fp32")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("softmax requires at least one dimension and N >= 1")
    dim = operator.index(dim)
    if dim not in (-1, x.ndim - 1):
        raise NotImplementedError("softmax supports only the last dimension")
    return x if x.stride(-1) == 1 else x.contiguous()


def softmax(x, dim=-1):
    """Normalize the final dimension with fp32 accumulation and same-dtype output."""
    device = x.device if getattr(x, "is_cuda", False) else None
    require_gpu(device)
    import torch

    x = _prepare_input(x, dim)
    if not x.is_cuda:
        raise GPUUnavailableError("softmax requires an NVIDIA CUDA input tensor")
    rows = x.numel() // x.shape[-1]
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows == 0:
        return output
    from ._softmax_kernels import launch_softmax

    with torch.cuda.device(x.device):
        launch_softmax(x, output, softmax_grid(rows), SOFTMAX_SINGLE_BLOCK_LIMIT)
    return output
