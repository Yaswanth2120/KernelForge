"""CPU tests for K3 benchmark plumbing. Fixture numbers are fake and never published."""
import importlib.util
import json
import sys

import pytest

from kernelforge import metrics, report
from kernelforge.bench import copy_probe, gpu_runner, suites
from kernelforge.bench.harness import Timing, benchmark
from kernelforge.cli import main

A100 = {'device': 'NVIDIA A100-SXM4-80GB', 'cc': '8.0', 'torch': '2.8.0', 'triton': '3.4.0',
        'cuda': '12.8', 'driver': '595.91.07', 'git_sha': 'fixture', 'date': '2026-11-24'}
OTHER = {**A100, 'device': 'NVIDIA GeForce RTX 4090', 'cc': '8.9'}


class FixedTimer:
    def __init__(self, ms):
        self.ms = ms

    def measure(self, operation):
        return Timing(ms_p20=self.ms * 0.99, ms_p50=self.ms, ms_p80=self.ms * 1.01)


def write_launches(directory, records_per_launch):
    directory.mkdir(parents=True, exist_ok=True)
    for index, records in enumerate(records_per_launch):
        lines = [record.model_dump_json(by_alias=True) for record in records]
        (directory / f'launch_{index}.jsonl').write_text('\n'.join(lines) + '\n')


def softmax_record(case, impl, ms, index, env, copy_gbps=1000.0):
    return benchmark(lambda: None, FixedTimer(ms), op='softmax', impl=impl, case=case,
                     launch_index=index, env=env, copy_gbps=copy_gbps,
                     byte_accounting=gpu_runner.softmax_bytes, baseline_call=f'{impl}(x)')


def copy_record(impl, ms, index):
    return benchmark(lambda: None, FixedTimer(ms), op='copy', impl=impl,
                     case=suites.copy_cases()[0], launch_index=index, env=A100,
                     byte_accounting=gpu_runner.copy_bytes, baseline_call=f'{impl}()')


def softmax_fixture(raw, env, kernel_ms, launches=3):
    """Every sweep case measured with a fixed time; selected cases share `kernel_ms`."""
    per_launch = []
    for index in range(launches):
        records = []
        for case in suites.softmax_sweep():
            records.append(softmax_record(case, 'kernelforge', kernel_ms + index * 1e-3, index, env))
            records.append(softmax_record(case, 'torch', kernel_ms * 1.1, index, env))
        per_launch.append(records)
    write_launches(raw / 'softmax', per_launch)
    (raw / 'copy_probe.json').write_text(json.dumps({
        'bw_copy_gbps': 1000.0, 'best_impl': 'torch_copy_', 'case': suites.copy_cases()[0],
        'per_impl': {'torch_copy_': {'gbps': 1000.0, 'ms_p50': 2.147, 'ms_min': 2.1,
                                     'ms_max': 2.2, 'n': 3, 'baseline_call': 'y.copy_(x)'}}}))


def test_sweep_contains_every_selected_case_at_registered_size():
    sweep = suites.softmax_sweep()
    selected = [{k: v for k, v in case.items() if k != 'selected'}
                for case in sweep if case['selected']]
    assert selected == suites.selected_rows()
    for case in sweep:
        assert case['M'] * case['N'] * {'fp16': 2, 'bf16': 2, 'fp32': 4}[case['dtype']] >= 512 * 2**20
    assert suites.copy_cases()[0]['buffer_bytes'] >= 2**30


def test_telemetry_parsing_and_clock_drop_flags():
    sample = gpu_runner.parse_telemetry('1410, 1410, 1593, 311.15, 36, 0x0000000000000000\n')
    assert sample['clocks.sm'] == '1410' and sample['power.draw'] == '311.15'
    assert 'error' in gpu_runner.parse_telemetry('garbage')
    assert not gpu_runner.clock_drop({'end': sample})
    assert not gpu_runner.clock_drop({'end': {**sample, 'clocks_throttle_reasons.active': '0x1'}})
    assert gpu_runner.clock_drop({'end': {**sample, 'clocks.sm': '1200'}})
    # 0x20 = SW thermal slowdown.
    assert gpu_runner.clock_drop({'end': {**sample, 'clocks_throttle_reasons.active': '0x20'}})
    assert gpu_runner.clock_drop({'end': {}})


def test_each_launch_is_a_fresh_process(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(gpu_runner.subprocess, 'run', lambda cmd, check: calls.append(cmd))
    gpu_runner.run_suite('softmax', tmp_path)
    assert [cmd[cmd.index('--launch-index') + 1] for cmd in calls] == ['0', '1', '2']
    assert all(cmd[:3] == [sys.executable, '-m', 'kernelforge.bench.gpu_runner'] for cmd in calls)
    with pytest.raises(ValueError, match='three'):
        gpu_runner.run_suite('softmax', tmp_path, launches=1)
    with pytest.raises(ValueError, match='Unknown'):
        gpu_runner.run_suite('nope', tmp_path)


def test_copy_probe_takes_max_over_implementations(tmp_path):
    # 2 GiB moved in 2.0 ms = 1073.74 GB/s beats 2.5 ms = 858.99 GB/s.
    write_launches(tmp_path / 'copy', [[copy_record('triton_copy', 2.0 + i * 0.01, i),
                                        copy_record('torch_copy_', 2.5, i)] for i in range(3)])
    summary = copy_probe.summarize(tmp_path)
    assert summary['best_impl'] == 'triton_copy'
    assert summary['bw_copy_gbps'] == pytest.approx(metrics.gbps(2 * 2**30, 2.01))
    assert summary['per_impl']['torch_copy_']['n'] == 3


def test_copy_probe_refuses_fewer_than_three_launches(tmp_path):
    write_launches(tmp_path / 'copy', [[copy_record('torch_copy_', 2.5, i)] for i in range(2)])
    with pytest.raises(ValueError, match='exactly 3'):
        copy_probe.summarize(tmp_path)


def test_report_evaluates_a100_criteria_and_counts_failures(tmp_path):
    # M*N*2*2 bytes for fp16 at 512 MiB input = 1.07e9 bytes; at 1.0 ms -> ~1074 GB/s, ratio 1.07.
    softmax_fixture(tmp_path, A100, kernel_ms=1.0)
    markdown, failures = report.render(tmp_path, tmp_path / 'plots')
    assert failures == 0
    assert markdown.count('| PASS | 3 |') == 9
    assert 'run on an A100' in markdown and 'BW_copy = **1000.0 GB/s**' in markdown
    plotted = (tmp_path / 'plots' / 'softmax_bandwidth.png').exists()
    assert plotted == (importlib.util.find_spec('matplotlib') is not None)
    assert ('![Softmax effective bandwidth]' in markdown) == plotted
    slow, _ = report.render(_fresh(tmp_path, 'slow', A100, kernel_ms=2.0))
    assert slow.count('| FAIL | 3 |') == 9


def _fresh(tmp_path, name, env, kernel_ms, launches=3):
    raw = tmp_path / name
    softmax_fixture(raw, env, kernel_ms, launches)
    return raw


def test_report_refuses_to_label_or_evaluate_non_a100(tmp_path):
    markdown, failures = report.render(_fresh(tmp_path, 'dev', OTHER, kernel_ms=2.0))
    assert failures == 0
    assert 'NOT an A100 run' in markdown and 'is not an A100' in markdown
    assert 'PASS' not in markdown.split('## Memory-copy')[0]


def test_report_never_prints_ratios_without_three_launches(tmp_path):
    markdown, failures = report.render(_fresh(tmp_path, 'two', A100, kernel_ms=1.0, launches=2))
    assert failures == 0
    assert markdown.count('not measured | 2 |') == 9
    sweep = markdown.split('## Softmax sweep')[1].split('## Accuracy')[0]
    assert 'not measured' in sweep and 'KF/torch' in sweep


def test_report_without_data_is_explicit(tmp_path):
    markdown, failures = report.render(tmp_path)
    assert failures == 0 and 'No benchmark records found.' in markdown
    assert '- `copy-probe`: not run.' in markdown


def test_cli_bench_usage_and_environment_errors(monkeypatch, tmp_path):
    assert main(['bench', '--suite', 'softmax', '--launches', '1', '--out', str(tmp_path)]) == 2
    assert main(['bench', '--suite', 'copy', '--out', str(tmp_path)]) == 2
    monkeypatch.setattr('kernelforge.is_triton_available', lambda: False)
    assert main(['bench', '--suite', 'softmax', '--out', str(tmp_path)]) == 3
    assert main(['copy-probe', '--out', str(tmp_path / 'copy_probe.json')]) == 3


def test_cli_report_exit_code_reflects_criteria(tmp_path):
    good = _fresh(tmp_path, 'good', A100, kernel_ms=1.0)
    assert main(['report', '--raw', str(good), '--out', str(tmp_path / 'out_good')]) == 0
    assert (tmp_path / 'out_good' / 'performance_report.md').exists()
    bad = _fresh(tmp_path, 'bad', A100, kernel_ms=2.0)
    assert main(['report', '--raw', str(bad), '--out', str(tmp_path / 'out_bad')]) == 1
