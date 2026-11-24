# Profiling evidence

No GPU profiling has been run. ncu availability and counter permissions are unknown.
The owner can run `kernelforge doctor --probe-ncu` to profile a trivial CUDA add;
retain return code, stdout, and stderr in the session artifacts.
An ERR_NVGPUCTRPERM result means counter access was denied; record it verbatim.
See https://developer.nvidia.com/ERR_NVGPUCTRPERM for host/container permissions.

Regardless of counter access, collect torch.profiler traces, exposed Triton compiled metadata,
analytic byte/FLOP accounting, and controlled launch-configuration experiments once kernels exist.
Record unavailable attributes explicitly rather than inventing occupancy/register measurements.
