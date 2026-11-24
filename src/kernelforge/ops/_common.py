import importlib

from kernelforge.errors import GPUUnavailableError


def is_triton_available():
    """Check the CUDA runtime and lazily import Triton."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            return False
        if torch.cuda.get_device_capability()[0] < 8:
            return False
        importlib.import_module("triton")
        return True
    except (ImportError, RuntimeError, OSError):
        return False


def require_gpu():
    """Reject calls without a supported runtime."""
    if not is_triton_available():
        raise GPUUnavailableError("Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton.")
