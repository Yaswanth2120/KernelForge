"""Host-side validation and pruning policy; these tests never execute Triton."""
from types import SimpleNamespace

import pytest
import torch

from kernelforge import GPUUnavailableError, softmax
from kernelforge.ops._common import prune_softmax_configs, softmax_grid
from kernelforge.ops.softmax import SOFTMAX_SINGLE_BLOCK_LIMIT, _prepare_input


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('shape', [(31,), (2, 3, 127), (0, 7)])
def test_valid_metadata_preserves_input(dtype, shape):
    x = torch.empty(shape, dtype=dtype)
    assert _prepare_input(x, -1) is x
    assert _prepare_input(x, len(shape) - 1) is x


def test_leading_strides_are_preserved():
    x = torch.zeros(2, 3, 31).transpose(0, 1)
    assert not x.is_contiguous()
    assert _prepare_input(x, -1) is x
    expanded = torch.zeros(1, 31).expand(7, 31)
    assert _prepare_input(expanded, -1) is expanded


def test_last_dimension_copy():
    x = torch.arange(2 * 62, dtype=torch.float32).reshape(2, 62)[:, ::2]
    prepared = _prepare_input(x, -1)
    assert prepared.stride(-1) == 1 and prepared.is_contiguous()
    assert prepared.data_ptr() != x.data_ptr()
    torch.testing.assert_close(prepared, x)


@pytest.mark.parametrize('x,dim,error', [
    (None, -1, TypeError), (torch.ones(3, dtype=torch.int32), -1, TypeError),
    (torch.ones(3, dtype=torch.float64), -1, TypeError),
    (torch.ones(()), -1, ValueError), (torch.ones(2, 0), -1, ValueError),
    (torch.ones(2, 3), 0, NotImplementedError), (torch.ones(2, 3), -2, NotImplementedError),
    (torch.ones(3), 0.0, TypeError),
])
def test_invalid_metadata(x, dim, error):
    with pytest.raises(error):
        _prepare_input(x, dim)


def test_sparse_layout_rejected():
    with pytest.raises(NotImplementedError, match='strided'):
        _prepare_input(torch.eye(3).to_sparse(), -1)


def test_cpu_tensor_rejected_with_mocked_runtime(monkeypatch):
    monkeypatch.setattr('kernelforge.ops._common.is_triton_available', lambda: True)
    with pytest.raises(GPUUnavailableError, match='CUDA input'):
        softmax(torch.ones(3))


@pytest.mark.parametrize('capability,expected', [((7, 5), False), ((8, 0), True)])
def test_device_specific_availability(monkeypatch, capability, expected):
    from kernelforge.ops._common import is_triton_available

    checked = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: checked.append(device) or capability)
    monkeypatch.setattr('kernelforge.ops._common.importlib.import_module', lambda _: object())
    assert is_triton_available(torch.device('cuda:1')) is expected
    assert checked == [torch.device('cuda:1')]


@pytest.mark.parametrize('rows,grid', [(1, (1, 1)), (7, (7, 1)),
                                     (2**31 - 1, (2**31 - 1, 1)),
                                     (2**31, (2**31 - 1, 2))])
def test_grid_dimensions(rows, grid):
    assert softmax_grid(rows) == grid
    assert grid[0] * grid[1] >= rows


@pytest.mark.parametrize('rows', [0, -1, 1.5, True, (2**31 - 1) * 65535 + 1])
def test_grid_rejects_invalid_counts(rows):
    with pytest.raises(ValueError):
        softmax_grid(rows)


def fake_config(block, warps):
    """Describe a candidate without importing Triton."""
    return SimpleNamespace(kwargs={'BLOCK': block}, num_warps=warps, num_stages=1)


def test_resource_pruning_and_evidence():
    configs = [fake_config(32, 16), fake_config(1024, 1), fake_config(1024, 4)]
    attempted = []

    def compile_candidate(config):
        """Simulate one expected resource failure and one successful compilation."""
        attempted.append(config.num_warps)
        if config.num_warps == 1:
            raise MemoryError('fixture: exceeded register budget')
        return SimpleNamespace(n_regs=32, n_spills=0, metadata=SimpleNamespace(shared=256))

    valid, records = prune_softmax_configs(configs, None, compile_candidate, (MemoryError,))
    assert valid == [configs[-1]]
    assert attempted == [1, 4]
    assert [record['status'] for record in records] == ['pruned', 'invalid', 'valid']
    assert records[-1]['registers'] == 32 and records[-1]['shared_bytes'] == 256
    assert 'register budget' in records[1]['reason']


def test_no_valid_configuration_reports_failure():
    with pytest.raises(RuntimeError, match='No valid softmax'):
        prune_softmax_configs([fake_config(1, 16)], None, lambda _: None, (MemoryError,))


def test_unexpected_compiler_error_propagates():
    def fail(_):
        """Represent a compiler bug that must not be hidden as resource pruning."""
        raise ValueError('unexpected compiler failure')

    with pytest.raises(ValueError, match='unexpected'):
        prune_softmax_configs([fake_config(1024, 4)], None, fail, (MemoryError,))


def test_fixed_block_pruning():
    config = SimpleNamespace(kwargs={}, num_warps=1, num_stages=1)
    valid, records = prune_softmax_configs([config], 32, lambda _: object(), (MemoryError,))
    assert valid == [config] and records[0]['block'] == 32
    assert SOFTMAX_SINGLE_BLOCK_LIMIT >= 1


@pytest.mark.parametrize('failure', ['device', 'triton_import'])
def test_unusable_runtime_returns_false(monkeypatch, failure):
    from kernelforge.ops._common import is_triton_available

    def unavailable(*args):
        """Simulate an invalid CUDA device or unavailable Triton import."""
        if failure == 'device':
            raise ValueError('fixture: not a CUDA device')
        raise ImportError('fixture: no Triton')

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.version, 'hip', None)
    monkeypatch.setattr(torch.cuda, 'get_device_capability',
                        unavailable if failure == 'device' else lambda _: (8, 0))
    monkeypatch.setattr('kernelforge.ops._common.importlib.import_module', unavailable)
    assert not is_triton_available(torch.device('cuda:1'))
