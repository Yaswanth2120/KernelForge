"""Host-side LayerNorm validation and layout policy; these tests never execute Triton."""
import pytest
import torch

from kernelforge import GPUUnavailableError, layer_norm
from kernelforge.ops.layernorm import _prepare


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_contiguous_rows_are_views(dtype):
    x = torch.empty(2, 3, 4, 5, dtype=dtype)
    rows, leading, weight, bias = _prepare(x, (4, 5), None, None, 1e-5)
    assert rows.shape == (6, 20) and rows.data_ptr() == x.data_ptr()
    assert leading == (2, 3) and weight is None and bias is None


def test_row_stride_and_broadcast_stay_views():
    x = torch.zeros(12, 31)[::3]
    rows, *_ = _prepare(x, 31, None, None, 1e-5)
    assert rows.stride() == (93, 1) and rows.data_ptr() == x.data_ptr()
    expanded = torch.zeros(1, 31).expand(4, 31)
    rows, *_ = _prepare(expanded, 31, None, None, 1e-5)
    assert rows.stride(0) == 0 and rows.data_ptr() == expanded.data_ptr()


def test_vector_input_is_one_row():
    rows, leading, *_ = _prepare(torch.zeros(7), 7, None, None, 1e-5)
    assert rows.shape == (1, 7) and leading == ()


@pytest.mark.parametrize('make', [lambda: torch.zeros(3, 62)[:, ::2],
                                  lambda: torch.zeros(3, 4, 5).transpose(1, 2),
                                  lambda: torch.zeros(2, 3, 8).transpose(0, 1)[:, :, :5]])
def test_noncollapsible_layouts_copy_to_unit_stride(make):
    x = make()
    rows, *_ = _prepare(x, x.shape[-1] if x.ndim == 2 else tuple(x.shape[1:]), None, None, 1e-5)
    assert rows.stride(1) == 1
    torch.testing.assert_close(rows, x.reshape(rows.shape))


def test_mixed_precision_affine_allowed():
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    _, _, weight, bias = _prepare(x, 8, torch.ones(8), torch.zeros(8, dtype=torch.bfloat16), 1e-5)
    assert weight.dtype == torch.float32 and bias.dtype == torch.bfloat16


@pytest.mark.parametrize('x,shape,weight,bias,eps,error', [
    (None, 3, None, None, 1e-5, TypeError),
    (torch.ones(2, 3, dtype=torch.float64), 3, None, None, 1e-5, TypeError),
    (torch.ones(2, 3), 4, None, None, 1e-5, ValueError),
    (torch.ones(2, 3), (2, 3, 1), None, None, 1e-5, ValueError),
    (torch.ones(2, 3), (), None, None, 1e-5, ValueError),
    (torch.ones(2, 3), 0, None, None, 1e-5, ValueError),
    (torch.ones(2, 3), 3, torch.ones(2), None, 1e-5, ValueError),
    (torch.ones(2, 3, dtype=torch.float16), 3, torch.ones(3, dtype=torch.bfloat16), None, 1e-5,
     TypeError),
    (torch.ones(2, 3), 3, None, None, 0.0, ValueError),
    (torch.ones(2, 3), 3, None, None, float('nan'), ValueError),
])
def test_invalid_arguments(x, shape, weight, bias, eps, error):
    with pytest.raises(error):
        _prepare(x, shape, weight, bias, eps)


def test_cpu_tensor_rejected_with_mocked_runtime(monkeypatch):
    monkeypatch.setattr('kernelforge.ops._common.is_triton_available', lambda: True)
    with pytest.raises(GPUUnavailableError, match='CUDA input'):
        layer_norm(torch.ones(2, 3), 3)
