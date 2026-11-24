"""Owner-run CUDA tests for LayerNorm forward. Passing CPU checks do not validate the kernels."""
import math

import pytest
import torch
import torch.nn.functional as F

from kernelforge import is_triton_available, layer_norm, reference
from kernelforge.ops.layernorm import LAYER_NORM_SINGLE_BLOCK_LIMIT as LIMIT
from kernelforge.tolerance import compare_accuracy

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
WIDTHS = (1, 2, 3, 31, 127, 1023, 1025, 2048, 4096, 4097, 31337, LIMIT - 1, LIMIT, LIMIT + 1,
          262144)


@pytest.fixture(autouse=True)
def supported_gpu():
    if not is_triton_available():
        pytest.skip('Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton')


def random(shape, dtype, seed=11):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device='cuda')


def affine(shape, kind, dtype):
    """Weight/bias for an affine variant; `mixed` uses fp32 parameters with any input dtype."""
    param_dtype = torch.float32 if kind == 'mixed' else dtype
    weight = 1 + 0.5 * random(shape, param_dtype, seed=3)
    bias = random(shape, param_dtype, seed=5)
    return {'none': (None, None), 'weight': (weight, None), 'bias': (None, bias),
            'both': (weight, bias), 'mixed': (weight, bias)}[kind]


def same_dtype(parameter, dtype):
    """PyTorch's CUDA LayerNorm rejects fp32 affine with half input, so its baseline casts."""
    return None if parameter is None else parameter.to(dtype)


def check(x, shape, weight=None, bias=None, eps=1e-5, torch_stats=True):
    """Frozen error rule on output and stats, contract, input preservation, determinism.

    With torch_stats=False the stats are compared to the fp32 reference directly (same rule's
    floor, exact non-finite pattern) because PyTorch's own stats are not a valid baseline there.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    leading = x.shape[:x.ndim - len(shape)]
    snapshot = x.clone()
    out, mean, rstd = layer_norm(x, shape, weight, bias, eps, return_stats=True)
    baseline, base_mean, base_rstd = torch.ops.aten.native_layer_norm(
        x, shape, same_dtype(weight, x.dtype), same_dtype(bias, x.dtype), eps)
    expected, ref_mean, ref_rstd = reference.layer_norm(x, shape, weight, bias, eps,
                                                        return_stats=True)
    if not torch_stats:
        base_mean, base_rstd = ref_mean, ref_rstd
    for name, actual, base, ref, dtype in (
            ('output', out, baseline, expected, x.dtype),
            ('mean', mean, base_mean.reshape(leading).float(), ref_mean, torch.float32),
            ('rstd', rstd, base_rstd.reshape(leading).float(), ref_rstd, torch.float32)):
        result = compare_accuracy(actual, base, ref, dtype)
        assert result.passed, (
            f'{name}: shape={tuple(x.shape)} norm={shape} dtype={x.dtype} strides={x.stride()} '
            f'weight={None if weight is None else weight.dtype} '
            f'bias={None if bias is None else bias.dtype} kernel_max_abs={result.kernel.max_abs} '
            f'torch_max_abs={result.baseline.max_abs} allowed={result.allowed_max_abs} '
            f'nonfinite_match={result.kernel.nonfinite_match}')
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    assert mean.shape == leading and mean.dtype == torch.float32 and rstd.dtype == torch.float32
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0, equal_nan=True)
    repeated, _, _ = layer_norm(x, shape, weight, bias, eps, return_stats=True)
    assert torch.equal(out.view(torch.uint8), repeated.view(torch.uint8)), 'Nondeterministic output'
    # return_stats=False is a separately compiled/tuned variant: same rule, not bitwise equality.
    plain = layer_norm(x, shape, weight, bias, eps)
    result = compare_accuracy(plain, baseline, expected, x.dtype)
    assert result.passed, f'no-stats output: kernel_max_abs={result.kernel.max_abs}'
    return out, mean, rstd


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', WIDTHS)
def test_dtype_width_matrix(dtype, n):
    check(random((3, n), dtype), n, *affine((n,), 'both', dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, 4097, LIMIT + 1])
@pytest.mark.parametrize('kind', ['none', 'weight', 'bias', 'mixed'])
def test_affine_variants_and_mixed_precision(dtype, n, kind):
    check(random((4, n), dtype), n, *affine((n,), kind, dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('layout', ['vector', 'multi_dim_shape', 'row_stride', 'leading_transpose',
                                   'trailing_strided', 'offset', 'broadcast'])
@pytest.mark.parametrize('n', [33, LIMIT + 7])
def test_layouts(dtype, layout, n):
    if layout == 'vector':
        x, shape = random((n,), dtype), n
    elif layout == 'multi_dim_shape':
        x, shape = random((2, 3, 4, n), dtype), (4, n)
    elif layout == 'row_stride':
        x, shape = random((12, n), dtype)[::3], n
    elif layout == 'leading_transpose':
        x, shape = random((2, 3, n), dtype).transpose(0, 1), n
    elif layout == 'trailing_strided':
        x, shape = random((3, 2 * n), dtype)[:, ::2], n
    elif layout == 'offset':
        x, shape = random((5, n + 2), dtype)[1:4, 1:n + 1], n
    else:
        x, shape = random((1, n), dtype).expand(4, n), n
    full = (shape,) if isinstance(shape, int) else shape
    check(x, shape, *affine(full, 'both', dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('with_bias', [False, True])
@pytest.mark.parametrize('eps', [1e-5, 1e-2])
def test_single_column_rows(dtype, with_bias, eps):
    x = random((5, 1), dtype)
    bias = torch.full((1,), 0.75, dtype=dtype, device='cuda') if with_bias else None
    out, _, rstd = check(x, 1, None, bias, eps)
    assert torch.all(out == (0.75 if with_bias else 0))
    torch.testing.assert_close(rstd, torch.full_like(rstd, 1 / math.sqrt(eps)))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, LIMIT + 1])
@pytest.mark.parametrize('scale', [0., 1e-10, 1e4])
def test_constant_rows_and_extreme_scales(dtype, n, scale):
    x = torch.linspace(-1, 1, n, device='cuda').repeat(3, 1).mul(scale).add(scale).to(dtype)
    x[1] = 3.0
    out, _, _ = check(x, n, *affine((n,), 'both', dtype))
    assert torch.isfinite(out).all()


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('n', [31, LIMIT + 1])
def test_nan_and_inf_propagate(dtype, n):
    x = random((4, n), dtype)
    x[0, n // 2] = torch.nan
    x[1, 0] = torch.inf
    x[2] = torch.nan
    # PyTorch reports mean NaN for a +inf row (its Welford update computes inf - inf); the fp32
    # reference and this kernel give +inf, so stats are checked against the reference here.
    out, mean, _ = check(x, n, torch_stats=False)
    assert torch.isnan(out[:3]).all() and torch.isfinite(out[3]).all()
    assert torch.isnan(mean[0])


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape,norm', [((0, 31), 31), ((2, 0, 4, 33), (4, 33))])
def test_empty_leading_dimensions(dtype, shape, norm):
    x = torch.empty(shape, dtype=dtype, device='cuda')
    out, mean, rstd = layer_norm(x, norm, return_stats=True)
    assert out.shape == x.shape and mean.numel() == 0 and rstd.numel() == 0


@pytest.mark.gpu
def test_cuda_argument_errors():
    x = torch.ones(2, 3, device='cuda')
    with pytest.raises(ValueError, match='trailing'):
        layer_norm(x, 4)
    with pytest.raises(TypeError, match='fp16'):
        layer_norm(x.double(), 3)
    with pytest.raises(ValueError, match='device'):
        layer_norm(x, 3, torch.ones(3))
    with pytest.raises(ValueError, match='eps'):
        layer_norm(x, 3, eps=-1.0)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('n', [LIMIT, 2 * LIMIT])
def test_int64_offsets_over_two_billion_elements(n):
    torch.cuda.empty_cache()
    rows = 2**31 // n + 1
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 2 * rows * n * 2 + 2 * 1024**3:
        pytest.skip('Needs >8 GiB for fp16 input/output plus 2 GiB headroom')
    x = torch.zeros((rows, n), dtype=torch.float16, device='cuda')
    assert x.numel() > 2**31
    indices = [0, 1, rows - 2, rows - 1]
    ramp = torch.linspace(-3, 3, n, dtype=x.dtype, device=x.device)
    x[indices] = torch.stack((ramp, ramp.flip(0), torch.zeros_like(ramp), ramp * 2))
    out = layer_norm(x, n)[indices]
    result = compare_accuracy(out, F.layer_norm(x[indices], (n,)),
                              reference.layer_norm(x[indices], n), x.dtype)
    assert result.passed, (f'int64 n={n}, kernel_max_abs={result.kernel.max_abs}, '
                           f'torch_max_abs={result.baseline.max_abs}, '
                           f'allowed={result.allowed_max_abs}')
