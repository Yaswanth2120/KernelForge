"""Owner-run CUDA tests. Passing CPU checks do not validate these kernels."""
import pytest
import torch

from kernelforge import is_triton_available, reference, softmax
from kernelforge.ops.softmax import SOFTMAX_SINGLE_BLOCK_LIMIT
from kernelforge.tolerance import compare_accuracy

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
WIDTHS = (1, 2, 3, 31, 127, 1023, 1025, 2048, 4096, 4097,
          SOFTMAX_SINGLE_BLOCK_LIMIT - 1, SOFTMAX_SINGLE_BLOCK_LIMIT,
          SOFTMAX_SINGLE_BLOCK_LIMIT + 1, 31337, 262144)


@pytest.fixture(autouse=True)
def supported_gpu():
    """Skip GPU tests on hosts without the required NVIDIA runtime."""
    if not is_triton_available():
        pytest.skip('Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton')


def random_input(shape, dtype):
    """Create reproducible low-precision inputs directly on the GPU."""
    generator = torch.Generator(device='cuda').manual_seed(73)
    return torch.randn(shape, generator=generator, dtype=dtype, device='cuda')


def check_result(x, dim=-1):
    """Check the frozen error rule, output contract, input preservation, and bitwise repeatability."""
    snapshot = x.clone()
    actual = softmax(x, dim)
    baseline = torch.softmax(x, dim=dim)
    expected = reference.softmax(x)
    accuracy = compare_accuracy(actual, baseline, expected, x.dtype)
    assert accuracy.passed, (
        f'shape={tuple(x.shape)}, dtype={x.dtype}, strides={x.stride()}, '
        f'kernel_max_abs={accuracy.kernel.max_abs}, torch_max_abs={accuracy.baseline.max_abs}, '
        f'allowed={accuracy.allowed_max_abs}, nonfinite_match={accuracy.kernel.nonfinite_match}')
    assert actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device
    assert actual.is_contiguous()
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0, equal_nan=True)
    repeated = softmax(x, dim)
    assert torch.equal(actual.view(torch.uint8), repeated.view(torch.uint8)), 'Nondeterministic output'
    return actual


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', WIDTHS)
def test_dtype_shape_matrix(dtype, n):
    check_result(random_input((3, n), dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, SOFTMAX_SINGLE_BLOCK_LIMIT + 1])
@pytest.mark.parametrize('layout', ['vector', 'multi', 'row_stride', 'leading_transpose',
                                   'last_stride', 'offset', 'broadcast'])
def test_layouts(dtype, n, layout):
    if layout == 'vector':
        x = random_input((n,), dtype)
    elif layout == 'multi':
        x = random_input((2, 3, n), dtype)
    elif layout == 'row_stride':
        x = random_input((12, n), dtype)[::3]
    elif layout == 'leading_transpose':
        x = random_input((2, 3, n), dtype).transpose(0, 1)
    elif layout == 'last_stride':
        x = random_input((3, 2 * n), dtype)[:, ::2]
    elif layout == 'offset':
        x = random_input((5, n + 2), dtype)[1:4, 1:n + 1]
    else:
        x = random_input((1, n), dtype).expand(4, n)
    check_result(x, x.ndim - 1)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, SOFTMAX_SINGLE_BLOCK_LIMIT + 1])
@pytest.mark.parametrize('scale', [0., 1e-10, 1e4])
def test_extreme_scales_and_constants(dtype, n, scale):
    x = torch.linspace(-1, 1, n, dtype=dtype, device='cuda').repeat(3, 1) * scale
    x[1] = scale
    check_result(x)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, SOFTMAX_SINGLE_BLOCK_LIMIT + 1])
def test_infinities_nan_and_empty_prefix(dtype, n):
    x = random_input((6, n), dtype)
    x[0, ::3] = -torch.inf
    x[1] = -torch.inf
    x[2, n // 2] = torch.nan
    x[3, 0] = torch.inf
    x[4] = torch.nan
    x[5] = -torch.inf
    x[5, -1] = 0
    result = check_result(x)
    assert torch.count_nonzero(result[0, ::3]) == 0
    assert torch.isnan(result[1:5]).all()
    assert result[5, -1] == 1
    assert torch.count_nonzero(result[5, :-1]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('position', [0, 4096, SOFTMAX_SINGLE_BLOCK_LIMIT])
def test_looped_nan_locations(dtype, position):
    x = random_input((1, SOFTMAX_SINGLE_BLOCK_LIMIT + 1), dtype)
    x[0, position] = torch.nan
    assert torch.isnan(check_result(x)).all()


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(0, 31), (2, 0, 8193)])
def test_empty_leading_dimensions(dtype, shape):
    check_result(torch.empty(shape, dtype=dtype, device='cuda'))


@pytest.mark.gpu
def test_cuda_argument_errors():
    with pytest.raises(NotImplementedError, match='last dimension'):
        softmax(torch.ones((2, 3), device='cuda'), dim=0)
    with pytest.raises(TypeError, match='fp16'):
        softmax(torch.ones(3, dtype=torch.float64, device='cuda'))
    with pytest.raises(ValueError, match='N >= 1'):
        softmax(torch.empty((2, 0), device='cuda'))


@pytest.mark.gpu
def test_autotune_metadata():
    from kernelforge.ops._softmax_kernels import PRUNING_LOG

    for n in (31, SOFTMAX_SINGLE_BLOCK_LIMIT + 1):
        check_result(random_input((3, n), torch.float32))
    assert {entry['path'] for entry in PRUNING_LOG.values()} == {'single', 'looped'}
    for entry in PRUNING_LOG.values():
        valid = [{name: item[name] for name in ('block', 'num_warps', 'num_stages')}
                 for item in entry['configs'] if item['status'] == 'valid']
        assert entry['best_config'] in valid
        assert all(item['status'] in ('valid', 'pruned', 'invalid') for item in entry['configs'])


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('n', [SOFTMAX_SINGLE_BLOCK_LIMIT, 32768])
def test_int64_offsets_over_two_billion_elements(n):
    torch.cuda.empty_cache()
    rows = 2**31 // n + 1
    tensor_bytes = rows * n * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 2 * tensor_bytes + 2 * 1024**3:
        pytest.skip('Needs >8 GiB for fp16 input/output plus 2 GiB headroom')
    x = torch.zeros((rows, n), dtype=torch.float16, device='cuda')
    assert x.numel() > 2**31
    indices = [0, 1, rows - 2, rows - 1]
    ramp = torch.linspace(-3, 3, n, dtype=x.dtype, device=x.device)
    x[indices] = torch.stack((ramp, ramp.flip(0), torch.zeros_like(ramp), ramp * 2))
    actual = softmax(x)
    sampled = actual[indices]
    expected = reference.softmax(x[indices])
    baseline = torch.softmax(x[indices], dim=-1)
    accuracy = compare_accuracy(sampled, baseline, expected, x.dtype)
    assert accuracy.passed, (
        f'int64 n={n}, kernel_max_abs={accuracy.kernel.max_abs}, '
        f'torch_max_abs={accuracy.baseline.max_abs}, allowed={accuracy.allowed_max_abs}')
