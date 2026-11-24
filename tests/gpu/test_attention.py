"""Owner-run CUDA tests for causal attention forward. CPU checks do not validate the kernel."""
import math

import pytest
import torch
import torch.nn.functional as F

from kernelforge import causal_attention, is_triton_available, reference
from kernelforge.tolerance import compare_accuracy

DTYPES = (torch.float16, torch.bfloat16)
DIMS = (32, 64, 128)


@pytest.fixture(autouse=True)
def supported_gpu():
    if not is_triton_available():
        pytest.skip('Requires NVIDIA sm_80+ CUDA and PyTorch-bundled Triton')


def random(shape, dtype, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device='cuda')


def make_qkv(shape, dtype, seed=0):
    return tuple(random(shape, dtype, seed + i) for i in range(3))


def torch_lse(q, k, v, scale):
    """PyTorch's FlashAttention log-sum-exp (natural log), the same-dtype LSE baseline."""
    return torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, True, False, scale=scale)[1]


def check(q, k, v, sm_scale=None):
    """Frozen rule vs SDPA (output) and torch flash LSE, contract, and determinism."""
    scale = 1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    snapshot = [t.clone() for t in (q, k, v)]
    out, lse = causal_attention(q, k, v, sm_scale, return_lse=True)
    expected, ref_lse = reference.causal_attention(q, k, v, scale, return_lse=True)
    # PyTorch's flash op rejects a strided last dim; baselines get contiguous copies (same values).
    cq, ck, cv = (t.contiguous() for t in (q, k, v))
    baseline = F.scaled_dot_product_attention(cq, ck, cv, is_causal=True, scale=scale)
    for name, actual, base, ref, dtype in (
            ('output', out, baseline, expected, q.dtype),
            ('lse', lse, torch_lse(cq, ck, cv, scale).float(), ref_lse, torch.float32)):
        result = compare_accuracy(actual, base, ref, dtype)
        assert result.passed, (
            f'{name}: shape={tuple(q.shape)} dtype={q.dtype} scale={scale} '
            f'strides q={q.stride()} k={k.stride()} v={v.stride()} '
            f'kernel_max_abs={result.kernel.max_abs} torch_max_abs={result.baseline.max_abs} '
            f'allowed={result.allowed_max_abs} nonfinite_match={result.kernel.nonfinite_match}')
    assert out.shape == q.shape and out.dtype == q.dtype and out.is_contiguous()
    assert lse.shape == q.shape[:3] and lse.dtype == torch.float32
    for before, after in zip(snapshot, (q, k, v)):
        assert torch.equal(before, after)
    again, _ = causal_attention(q, k, v, sm_scale, return_lse=True)
    assert torch.equal(out.view(torch.uint8), again.view(torch.uint8)), 'Nondeterministic output'
    plain = causal_attention(q, k, v, sm_scale)
    assert compare_accuracy(plain, baseline, expected, q.dtype).passed
    return out, lse


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('seq', [1, 2, 31, 64, 127, 128, 129, 255, 1000])
def test_dtype_dim_seq_matrix(dtype, dim, seq):
    check(*make_qkv((2, 3, seq, dim), dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('dim', [64, 128])
def test_long_irregular_sequence(dtype, dim):
    check(*make_qkv((1, 2, 4097, dim), dtype))


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(1, 1, 1, 64), (1, 1, 130, 64), (4, 64, 64, 64),
                                   (16, 16, 33, 32)])
def test_batch_and_head_extremes(shape):
    check(*make_qkv(shape, torch.float16))


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('layout', ['bshd_transpose', 'mixed_layouts', 'sliced_heads',
                                   'last_dim_strided'])
def test_strided_views(dtype, layout):
    b, h, s, d = 2, 4, 193, 64
    if layout == 'bshd_transpose':
        q, k, v = (random((b, s, h, d), dtype, i).transpose(1, 2) for i in range(3))
    elif layout == 'mixed_layouts':
        q = random((b, h, s, d), dtype, 0)
        k = random((s, b, h, d), dtype, 1).permute(1, 2, 0, 3)
        v = random((b, s, h, d), dtype, 2).transpose(1, 2)
    elif layout == 'sliced_heads':
        q, k, v = (random((b, 2 * h, s + 5, d), dtype, i)[:, ::2, 3:s + 3] for i in range(3))
    else:
        q, k, v = (random((b, h, s, 2 * d), dtype, i)[..., ::2] for i in range(3))
    assert not all(t.is_contiguous() for t in (q, k, v))
    check(q, k, v)


@pytest.mark.gpu
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('scale', [0.05, 1.0, 3.0])
def test_custom_scales_and_large_logits(dtype, scale):
    q, k, v = make_qkv((1, 2, 300, 64), dtype)
    check(q * 4, k * 4, v, sm_scale=scale)


@pytest.mark.gpu
def test_first_row_attends_only_itself():
    q, k, v = make_qkv((1, 1, 70, 64), torch.float16)
    out = causal_attention(q, k, v)
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], rtol=0, atol=0)


@pytest.mark.gpu
def test_autotune_records_valid_and_invalid_configs():
    from kernelforge.ops._attention_kernels import LAST_LAUNCH

    check(*make_qkv((1, 2, 512, 128), torch.bfloat16))
    statuses = {config['status'] for config in LAST_LAUNCH['configs']}
    assert statuses <= {'valid', 'invalid'} and 'valid' in statuses
    best = LAST_LAUNCH['best_config']
    assert best['block_n'] <= best['block_m']


@pytest.mark.gpu
def test_cuda_argument_errors():
    q, k, v = make_qkv((1, 1, 8, 64), torch.float16)
    with pytest.raises(TypeError, match='fp16'):
        causal_attention(q.float(), k.float(), v.float())
    with pytest.raises(ValueError, match='head dim'):
        causal_attention(q[..., :48], k[..., :48], v[..., :48])
    with pytest.raises(ValueError, match='identical'):
        causal_attention(q, k[:, :, :4], v)


@pytest.mark.gpu
@pytest.mark.slow
def test_int64_offsets_over_two_billion_elements():
    # 1 * 1025 * 16384 * 128 = 2.15e9 elements per tensor > 2^31; 4.3 GB each in fp16.
    shape = (1, 1025, 16384, 128)
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 4 * math.prod(shape) * 2 + 4 * 1024**3:
        pytest.skip('Needs ~22 GiB for q, k, v, out plus reference headroom')
    q, k, v = (torch.empty(shape, dtype=torch.float16, device='cuda').normal_(generator=g)
               for g in (torch.Generator(device='cuda').manual_seed(i) for i in range(3)))
    out = causal_attention(q, k, v)
    for head in (0, shape[1] - 1):
        sl = slice(head, head + 1)
        expected = reference.causal_attention(q[:, sl], k[:, sl], v[:, sl])
        baseline = F.scaled_dot_product_attention(q[:, sl], k[:, sl], v[:, sl], is_causal=True)
        result = compare_accuracy(out[:, sl], baseline, expected, q.dtype)
        assert result.passed, f'head {head}: kernel_max_abs={result.kernel.max_abs}'
