"""Run manually on Linux with the PyTorch-bundled Triton, without a GPU."""
import os
import platform

os.environ["TRITON_INTERPRET"] = "1"


def main():
    """Check masked indexing on CPU tensors and print the observed result."""
    if platform.system() != "Linux":
        raise SystemExit("Probe requires Linux; result remains not yet determined.")
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def copy_tail(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < N, other=0)
        tl.store(y + offsets, values, offsets < N)

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        try:
            x = torch.arange(31, dtype=dtype)
            y = torch.empty_like(x)
            copy_tail[(1,)](x, y, 31, 32)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            print(f"{dtype}: masked CPU copy passed")
        except Exception as exc:  # noqa: BLE001 -- report unsupported interpreter intrinsics
            print(f"{dtype}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
