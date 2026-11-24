"""CPU reference checks use seeded tensors and independent PyTorch operations."""
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernelforge import reference


@pytest.mark.parametrize('shape', [(1,), (3, 31), (2, 3, 127), (0, 7)])
def test_softmax_fp64(shape):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    torch.testing.assert_close(reference.softmax(x, dtype=torch.float64), torch.softmax(x, -1),
                               rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_softmax_promotes_same_inputs(dtype):
    x = torch.tensor([[1e4, -1e4, 0], [0, -torch.inf, 2]], dtype=dtype)
    y = reference.softmax(x)
    assert y.dtype == torch.float32
    assert y[1, 1] == 0
    torch.testing.assert_close(y, torch.softmax(x.float(), -1))


def test_softmax_special_rows_and_strides():
    x = torch.tensor([[1., 99., 2., 99.], [-torch.inf] * 4, [float('nan'), 0., 1., 0.]])[:, ::2]
    torch.testing.assert_close(reference.softmax(x), torch.softmax(x, -1), equal_nan=True)


@pytest.mark.parametrize('shape,normalized', [((2, 3, 7), (3, 7)), ((4, 1), 1),
                                            ((0, 3), 3), ((2, 31), 31)])
@pytest.mark.parametrize('affine', ['both', 'weight', 'bias', 'none'])
def test_layer_norm_fp64(shape, normalized, affine):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    parameter_shape = (normalized,) if isinstance(normalized, int) else normalized
    weight = torch.randn(parameter_shape, generator=generator, dtype=torch.float64)
    bias = torch.randn(parameter_shape, generator=generator, dtype=torch.float64)
    weight = weight if affine in ('both', 'weight') else None
    bias = bias if affine in ('both', 'bias') else None
    y, mean, rstd = reference.layer_norm(x, normalized, weight, bias, return_stats=True,
                                       dtype=torch.float64)
    torch.testing.assert_close(y, F.layer_norm(x, parameter_shape, weight, bias),
                               rtol=1e-12, atol=1e-12)
    dims = tuple(range(x.ndim - len(parameter_shape), x.ndim))
    torch.testing.assert_close(mean, x.mean(dims))
    if x.numel():
        torch.testing.assert_close(rstd, torch.rsqrt(x.var(dims, unbiased=False) + 1e-5))
    else:
        assert mean.shape == rstd.shape == (0,)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('scale', [1e-10, 1., 1e4])
def test_layer_norm_scales_and_mixed_affine(dtype, scale):
    x = torch.tensor([[0., 1., 2., 3.], [2., 2., 2., 2.]], dtype=dtype) * scale
    x = x.t().contiguous().t()  # Noncontiguous rows preserve the original values.
    weight, bias = torch.ones(4), torch.arange(4, dtype=torch.float32)
    output, mean, rstd = reference.layer_norm(x, 4, weight, bias, return_stats=True)
    torch.testing.assert_close(output, F.layer_norm(x.float(), (4,), weight, bias),
                               rtol=1e-5, atol=1e-6)
    assert mean.dtype == rstd.dtype == torch.float32
    torch.testing.assert_close(output[1], bias)
    torch.testing.assert_close(rstd[1], torch.tensor(1e-5).rsqrt())


@pytest.mark.parametrize('s,d', [(1, 32), (7, 64), (31, 128)])
@pytest.mark.parametrize('scale', [None, 0., 0.3])
def test_attention_fp64(s, d, scale):
    generator = torch.Generator().manual_seed(31)
    q, k, v = [torch.randn((2, 1, s, d), generator=generator, dtype=torch.float64)
               for _ in range(3)]
    output, lse = reference.causal_attention(q, k, v, scale, True, dtype=torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    scores = q @ k.transpose(-1, -2) * (d**-0.5 if scale is None else scale)
    scores.masked_fill_(torch.ones((s, s), dtype=torch.bool).triu(1), -torch.inf)
    torch.testing.assert_close(lse, torch.logsumexp(scores, -1))
    torch.testing.assert_close(output[..., 0, :], v[..., 0, :])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_attention_promotes_strided_inputs(dtype):
    generator = torch.Generator().manual_seed(3)
    q, k, v = [torch.randn((1, 7, 2, 32), generator=generator).to(dtype).transpose(1, 2)
               for _ in range(3)]
    actual = reference.causal_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_layer_norm_nan_and_singleton():
    x = torch.tensor([[float('nan'), 1.], [0., 1.]])
    torch.testing.assert_close(reference.layer_norm(x, 2), F.layer_norm(x, (2,)), equal_nan=True)
    y, mean, rstd = reference.layer_norm(torch.tensor([[4.]]), 1, bias=torch.tensor([7.]),
                                       return_stats=True)
    assert y.item() == 7 and mean.item() == 4
    torch.testing.assert_close(rstd, torch.tensor([1e-5]).rsqrt())


def test_reference_argument_errors():
    with pytest.raises(NotImplementedError):
        reference.softmax(torch.ones(2, 3), dim=0)
    with pytest.raises(ValueError):
        reference.softmax(torch.ones(3), dtype=torch.float16)
    with pytest.raises(ValueError):
        reference.softmax(torch.ones(0))
    for shape in [(), 0, (3, 2), 7]:
        with pytest.raises(ValueError):
            reference.layer_norm(torch.ones(2, 3), shape)
    with pytest.raises(ValueError):
        reference.layer_norm(torch.ones(2, 3), 3, weight=torch.ones(2))
    with pytest.raises(ValueError):
        reference.layer_norm(torch.ones(2, 3), 3, eps=-1)
    with pytest.raises(ValueError):
        reference.causal_attention(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        reference.causal_attention(*(torch.ones(1, 1, 2, 8) for _ in range(3)))
