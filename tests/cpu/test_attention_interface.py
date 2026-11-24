"""Host-side causal attention validation; these tests never execute Triton."""
import math

import pytest
import torch

from kernelforge import GPUUnavailableError, causal_attention
from kernelforge.ops.attention import _prepare


def qkv(shape=(2, 3, 17, 64), dtype=torch.float16):
    return tuple(torch.zeros(shape, dtype=dtype) for _ in range(3))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('dim', [32, 64, 128])
def test_valid_inputs_keep_strided_views(dtype, dim):
    q, k, v = qkv((2, 3, 17, dim), dtype)
    k = torch.zeros(2, 17, 3, dim, dtype=dtype).transpose(1, 2)
    (pq, pk, pv), scale = _prepare(q, k, v, None)
    assert pq is q and pk is k and pv is v
    assert scale == pytest.approx(1 / math.sqrt(dim))


def test_last_dim_stride_is_copied():
    q, _, v = qkv()
    k = torch.zeros(2, 3, 17, 128, dtype=torch.float16)[..., ::2]
    (_, pk, _), scale = _prepare(q, k, v, 0.5)
    assert pk.stride(-1) == 1 and pk.data_ptr() != k.data_ptr() and scale == 0.5


@pytest.mark.parametrize('change,error', [
    (lambda q, k, v: (q.float(), k.float(), v.float()), TypeError),
    (lambda q, k, v: (q, k.bfloat16(), v), TypeError),
    (lambda q, k, v: (q[..., :48], k[..., :48], v[..., :48]), ValueError),
    (lambda q, k, v: (q, k[:, :, :5], v), ValueError),
    (lambda q, k, v: (q[0], k[0], v[0]), ValueError),
    (lambda q, k, v: (q[:, :, :0], k[:, :, :0], v[:, :, :0]), ValueError),
    (lambda q, k, v: (None, k, v), TypeError),
])
def test_invalid_inputs(change, error):
    with pytest.raises(error):
        _prepare(*change(*qkv()), None)


def test_invalid_scale_and_grid_limit():
    with pytest.raises(ValueError, match='finite'):
        _prepare(*qkv(), float('inf'))
    big = torch.zeros(1, 1, 1, 64, dtype=torch.float16).expand(256, 257, 1, 64)
    with pytest.raises(ValueError, match='B \\* H'):
        _prepare(big, big, big, None)


def test_cpu_tensors_rejected_with_mocked_runtime(monkeypatch):
    monkeypatch.setattr('kernelforge.ops._common.is_triton_available', lambda: True)
    with pytest.raises(GPUUnavailableError, match='CUDA input'):
        causal_attention(*qkv())
