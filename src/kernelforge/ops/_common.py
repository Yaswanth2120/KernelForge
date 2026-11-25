import importlib

from kernelforge.errors import GPUUnavailableError


def is_triton_available(device=None):
    """Check the CUDA runtime and lazily import Triton."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            return False
        if torch.cuda.get_device_capability(device)[0] < 8:
            return False
        importlib.import_module("triton")
        return True
    except (ImportError, RuntimeError, OSError, ValueError):
        return False


def require_gpu(device=None):
    """Reject calls without a supported runtime."""
    available = is_triton_available() if device is None else is_triton_available(device)
    if not available:
        raise GPUUnavailableError("Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton.")


def softmax_grid(rows):
    """Split logical row IDs over CUDA's first two grid dimensions."""
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError("Expected a positive integer row count")
    width = min(rows, 2**31 - 1)
    height = (rows + width - 1) // width
    if height > 65535:
        raise ValueError("Row count exceeds CUDA grid limits")
    return width, height


def prune_softmax_configs(configs, fixed_block, compile_config, resource_errors):
    """Keep structurally useful configurations that compile and fit device resources."""
    valid, records = [], []
    for config in configs:
        block = config.kwargs.get("BLOCK", fixed_block)
        record = {"block": block, "num_warps": config.num_warps,
                  "num_stages": config.num_stages}
        if config.num_warps * 32 > max(32, block):
            records.append({**record, "status": "pruned", "reason": "more threads than padded columns"})
            continue
        try:
            compiled = compile_config(config)
        except resource_errors as exc:
            records.append({**record, "status": "invalid", "reason": str(exc)})
            continue
        valid.append(config)
        records.append({**record, "status": "valid", "registers": getattr(compiled, "n_regs", None),
                        "spills": getattr(compiled, "n_spills", None),
                        "shared_bytes": getattr(getattr(compiled, "metadata", None), "shared", None)})
    if not valid:
        raise RuntimeError(f"No valid softmax configurations: {records}")
    return valid, records
