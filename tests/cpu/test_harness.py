"""Fake timings are test fixtures and must never be published as GPU measurements."""
import pytest
from pydantic import ValidationError

from kernelforge.bench.harness import (
    Environment,
    Measurement,
    aggregate_launches,
    benchmark,
    quantile,
    summarize,
)

ENV = {'device': 'CPU TEST FIXTURE', 'cc': 'none', 'torch': '2.8.0', 'triton': 'fixture',
       'cuda': 'fixture', 'driver': 'fixture', 'git_sha': 'fixture', 'date': '2026-11-23'}


class FakeTimer:
    def __init__(self, samples):
        self.samples = samples
        self.calls = 0

    def measure(self, operation):
        self.calls += 1
        operation()
        return summarize(self.samples)


def measurement(index=0, samples=(1., 2., 3., 4., 5.), **kwargs):
    return benchmark(lambda: None, FakeTimer(samples), op='softmax', impl='fixture',
                     case={'M': 3, 'N': 5, 'dtype': 'fp16'}, launch_index=index, env=ENV,
                     byte_accounting=lambda case: 2 * case['M'] * case['N'] * 2,
                     **kwargs)


def test_known_quantiles_and_timer_delegation():
    # Inclusive interpolation at .2 and .8 in [1,2,3,4,5] gives 1.8 and 4.2; median is 3.
    timer = FakeTimer([5., 2., 4., 1., 3.])
    calls = []
    result = benchmark(lambda: calls.append('run'), timer, op='softmax', impl='fixture',
                       case={}, launch_index=0, env=ENV, byte_accounting=lambda _: 6_000_000,
                       flop_accounting=lambda _: 3_000_000_000, copy_gbps=4)
    assert (result.ms_p20, result.ms_p50, result.ms_p80) == pytest.approx((1.8, 3., 4.2))
    # 6e6 bytes / 3 ms / 1e6 = 2 GB/s; 3e9 FLOPs / 3 ms / 1e9 = 1 TFLOPS.
    assert result.gbps == 2 and result.tflops == 1 and result.ratio_to_copy == 0.5
    assert calls == ['run'] and timer.calls == 1


def test_three_launch_aggregation_and_range():
    records = [measurement(2, [9.]), measurement(0, [1.]), measurement(1, [5.])]
    combined = aggregate_launches(records)
    assert combined.n == 3
    assert combined.ms_p50 == 5
    assert combined.ms_min == 1 and combined.ms_max == 9
    assert combined.gbps == 60 / 5 / 1e6
    assert [record.launch_index for record in combined.launches] == [0, 1, 2]
    assert combined.ratio_to_copy is None


def test_aggregation_with_work_and_copy_baseline():
    combined = aggregate_launches([measurement(i, [float(i + 1)], copy_gbps=10,
                                               flop_accounting=lambda _: 1_000_000_000)
                                  for i in range(3)])
    assert combined.ms_p50 == 2
    assert combined.tflops == 0.5
    assert combined.ratio_to_copy == combined.gbps / 10


def test_json_schema_roundtrip():
    record = measurement()
    data = record.model_dump_json(by_alias=True)
    assert '"schema":"kernelforge.bench.v1"' in data
    assert Measurement.model_validate_json(data) == record
    schema = Measurement.model_json_schema()
    assert schema['properties']['schema']['const'] == 'kernelforge.bench.v1'


@pytest.mark.parametrize('field,value', [
    ('ms_p20', 9), ('ms_p50', 0), ('ms_p80', float('inf')),
    ('bytes', -1), ('bytes', 1.2), ('bytes', True), ('gbps', 123),
    ('ratio_to_copy', 1), ('launch_index', -1), ('launch_index', True),
    ('peak_mem_delta_bytes', -1), ('schema', 'invented'), ('extra', 1),
    ('env', {'device': 'incomplete'}), ('autotune', {'tuning_seconds': -1}),
])
def test_invalid_records(field, value):
    data = measurement().model_dump(by_alias=True)
    data[field] = value
    with pytest.raises(ValidationError):
        Measurement.model_validate(data)


@pytest.mark.parametrize('count', [0, 1, 2, 4])
def test_requires_three_launches(count):
    with pytest.raises(ValueError, match='three'):
        aggregate_launches([measurement(i) for i in range(count)])


def test_rejects_duplicate_launches():
    with pytest.raises(ValueError, match='indices'):
        aggregate_launches([measurement(0), measurement(0), measurement(1)])


@pytest.mark.parametrize('field,value', [('case', {'M': 4}), ('impl', 'other'),
                                        ('bytes', 120), ('copy_gbps', 7),
                                        ('op', 'layer_norm'), ('baseline_call', 'torch.softmax(x)')])
def test_rejects_incompatible_launches(field, value):
    records = [measurement(i) for i in range(3)]
    data = records[-1].model_dump(by_alias=True)
    data[field] = value
    if field == 'bytes':
        data['gbps'] *= 2
    if field == 'copy_gbps':
        data['ratio_to_copy'] = data['gbps'] / value
    records[-1] = Measurement.model_validate(data)
    with pytest.raises(ValueError, match=f'differing {field}'):
        aggregate_launches(records)


def test_rejects_mixed_environments():
    records = [measurement(i) for i in range(3)]
    records[-1].env = Environment(**{**ENV, 'device': 'another device'})
    with pytest.raises(ValueError, match='differing env'):
        aggregate_launches(records)


@pytest.mark.parametrize('samples', [[], [0], [-1], [float('nan')], [float('inf')]])
def test_invalid_sample_timings(samples):
    with pytest.raises(ValueError):
        summarize(samples)


def test_single_sample_and_endpoints():
    assert summarize([3]).ms_p50 == 3
    assert quantile([4, 2], 0) == 2
    assert quantile([4, 2], 1) == 4
    with pytest.raises(ValueError):
        quantile([1], 1.1)


def test_aggregate_json_cannot_forge_launch_count_or_rates():
    from kernelforge.bench.harness import Aggregate

    result = aggregate_launches([measurement(i) for i in range(3)])
    data = result.model_dump(by_alias=True)
    assert Aggregate.model_validate_json(result.model_dump_json(by_alias=True)) == result
    data['launches'] = data['launches'][:2]
    with pytest.raises(ValidationError):
        Aggregate.model_validate(data)
    data = result.model_dump(by_alias=True)
    data['gbps'] *= 2
    with pytest.raises(ValidationError, match='inconsistent'):
        Aggregate.model_validate(data)


def test_missing_schema_identity_is_rejected():
    data = measurement().model_dump(by_alias=True)
    del data['schema']
    with pytest.raises(ValidationError, match='schema'):
        Measurement.model_validate(data)
