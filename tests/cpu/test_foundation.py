import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernelforge import GPUUnavailableError, causal_attention, layer_norm, softmax
from kernelforge.bench import suites
from kernelforge.cli import main
from kernelforge.env import command_output, doctor


@pytest.mark.parametrize('module', ['kernelforge', 'kernelforge.ops', 'kernelforge.reference',
                                   'kernelforge.tolerance', 'kernelforge.metrics',
                                   'kernelforge.bench.harness', 'kernelforge.report',
                                   'kernelforge.bench.suites', 'kernelforge.bench.gpu_runner',
                                   'kernelforge.bench.copy_probe', 'kernelforge.cli'])
def test_import_without_triton(module):
    code = f"""
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] == 'triton':
        raise AssertionError('Triton imported eagerly')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import {module}
"""
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('operation,args', [(softmax, (None,)),
                                         (layer_norm, (None, 1)),
                                         (causal_attention, (None, None, None))])
def test_unavailable_gpu(monkeypatch, operation, args):
    monkeypatch.setattr('kernelforge.ops._common.is_triton_available', lambda: False)
    with pytest.raises(GPUUnavailableError, match='sm_80'):
        operation(*args)


def test_unimplemented_layer_norm_on_supported_gpu(monkeypatch):
    monkeypatch.setattr('kernelforge.ops._common.is_triton_available', lambda: True)
    with pytest.raises(NotImplementedError):
        layer_norm(None, 1)


def test_registered_suites():
    suites.check_prereg(Path('PREREG.md'))
    rows = suites.selected_rows()
    assert len(rows) == 9
    assert {(c['N'], c['dtype']) for c in rows} == {
        (n, dtype) for n in (2048, 4096, 8192) for dtype in ('fp16', 'bf16', 'fp32')}
    for case in rows:
        row_bytes = case['N'] * {'fp16': 2, 'bf16': 2, 'fp32': 4}[case['dtype']]
        assert case['M'] * row_bytes >= 512 * 1024**2
        assert (case['M'] - 1) * row_bytes < 512 * 1024**2
    assert suites.selected_attention() == [
        {'B': b, 'H': 32, 'S': s, 'D': d, 'dtype': dtype}
        for b, s in ((8, 2048), (4, 4096), (2, 8192))
        for d in (64, 128) for dtype in ('fp16', 'bf16')]


def test_protocol_drift(tmp_path):
    changed = tmp_path / 'PREREG.md'
    changed.write_text(Path('PREREG.md').read_text().replace('0.80', '0.70'))
    with pytest.raises(ValueError, match='changed'):
        suites.check_prereg(changed)


def test_suite_drift(monkeypatch):
    monkeypatch.setattr(suites, 'ROW_WIDTHS', (2048,))
    with pytest.raises(ValueError, match='Suite definitions'):
        suites.check_prereg(Path('PREREG.md'))


def test_offline_doctor(monkeypatch, capsys):
    monkeypatch.setattr('kernelforge.env.is_triton_available', lambda: False)
    monkeypatch.setattr('kernelforge.env.command_output', lambda _: {'error': 'offline'})
    assert doctor()['gpu_available'] is False
    assert main(['doctor']) == 3
    assert json.loads(capsys.readouterr().out)['gpu_available'] is False


def test_missing_command():
    assert command_output(['kernelforge-nonexistent-executable'])['returncode'] is None


def test_cli_prereg(capsys, tmp_path):
    assert main(['check-prereg']) == 0
    assert 'match' in capsys.readouterr().out
    assert main(['check-prereg', '--path', str(tmp_path / 'missing')]) == 1
