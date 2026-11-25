"""Owner-run CUDA test: the BW_copy Triton kernel must copy exactly before it is timed."""
import pytest
import torch

from kernelforge import is_triton_available


@pytest.fixture(autouse=True)
def supported_gpu():
    if not is_triton_available():
        pytest.skip('Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton')


@pytest.mark.gpu
@pytest.mark.parametrize('n', [1, 1023, 4097, 2**20 + 7])
def test_triton_copy_is_exact(n):
    from kernelforge.ops._copy_kernels import triton_copy

    x = torch.randn(n, device='cuda')
    y = torch.full_like(x, float('nan'))
    best = triton_copy(x, y)
    assert torch.equal(x, y)
    assert set(best) == {'block', 'num_warps', 'num_stages'}
