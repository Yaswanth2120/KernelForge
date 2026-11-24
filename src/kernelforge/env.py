import importlib.metadata
import platform
import shutil
import subprocess

from kernelforge.ops import is_triton_available

PINNED_TORCH = "2.8.0"


def command_output(command):
    """Capture a bounded external diagnostic without failing offline."""
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def doctor(probe_ncu=False):
    """Collect environment facts without downloading or initializing kernels."""
    versions = {}
    for package in ("torch", "triton", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    available = is_triton_available()
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "versions": versions, "gpu_available": available,
              "torch_pin_matches": (versions["torch"] or "").split("+")[0] == PINNED_TORCH,
              "nvidia_smi": command_output(["nvidia-smi"]),
              "interpreter": "not yet determined; run scripts/interpreter_probe.py on Linux",
              "ncu": {"status": "not run", "executable": shutil.which("ncu")}}
    if probe_ncu and available and shutil.which("ncu"):
        import sys

        result["ncu"] = command_output([
            "ncu", "--target-processes", "all", "--metrics", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            sys.executable, "-c",
            "import torch; x=torch.ones(1024, device='cuda'); x.add_(1); torch.cuda.synchronize()"])
    return result


def timing_source():
    """Return the exact installed timer source for manual review on the GPU host."""
    import inspect

    if not is_triton_available():
        raise RuntimeError("Installed CUDA/Triton environment required to inspect do_bench")
    from triton.testing import do_bench

    return inspect.getsource(do_bench)
