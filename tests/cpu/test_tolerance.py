import pytest
import torch

from kernelforge import reference
from kernelforge.tolerance import compare_accuracy, error_metrics


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_torch_fake_passes_and_broken_fake_fails(dtype):
    x = torch.randn((3, 31), generator=torch.Generator().manual_seed(41)).to(dtype)
    expected = reference.softmax(x)
    baseline = torch.softmax(x, -1)
    good = compare_accuracy(baseline.clone(), baseline, expected, dtype)
    bad = compare_accuracy(torch.zeros_like(baseline), baseline, expected, dtype)
    assert good.passed
    assert not bad.passed
    assert good.allowed_max_abs == max(2 * good.baseline.max_abs,
                                      2 * torch.finfo(dtype).eps * expected.abs().max().item())


def test_hand_computed_errors():
    # abs errors [1, 0.5, 1e-9]; only the first two references exceed the 1e-8 floor.
    result = error_metrics(torch.tensor([3., 1.5, 1e-9]), torch.tensor([2., 1., 0.]))
    assert result.max_abs == 1
    assert result.mean_abs == pytest.approx(0.5)
    assert result.max_relative == 0.5


def test_zero_reference_and_empty():
    zeros = torch.zeros(3)
    assert compare_accuracy(zeros, zeros, zeros, torch.float32).passed
    assert not compare_accuracy(zeros + 1e-12, zeros, zeros, torch.float32).passed
    empty = torch.empty(0)
    assert compare_accuracy(empty, empty, empty, torch.float16).passed


@pytest.mark.parametrize('actual', [[float('nan'), float('inf'), -float('inf')],
                                   [0., float('inf'), -float('inf')],
                                   [float('nan'), -float('inf'), -float('inf')]])
def test_nonfinite_policy(actual):
    reference = torch.tensor([float('nan'), float('inf'), -float('inf')])
    result = compare_accuracy(torch.tensor(actual), reference, reference, torch.float32)
    expected = torch.isnan(torch.tensor(actual)[0]).item() and actual[1] == float('inf')
    assert result.passed == expected


def test_bad_baseline_cannot_make_threshold_infinite():
    ref = torch.ones(2)
    result = compare_accuracy(ref, torch.tensor([float('nan'), 1.]), ref, torch.float32)
    assert not result.passed


def test_policy_boundary():
    ref = torch.tensor([1.])
    baseline = torch.tensor([1.125])
    assert compare_accuracy(torch.tensor([1.25]), baseline, ref, torch.float32).passed
    assert not compare_accuracy(torch.tensor([1.251]), baseline, ref, torch.float32).passed


def test_tolerance_arguments():
    with pytest.raises(ValueError):
        error_metrics(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        error_metrics(torch.ones(2), torch.ones(2), relative_floor=0)
    with pytest.raises(ValueError):
        compare_accuracy(torch.ones(2), torch.ones(2), torch.ones(2), torch.float64)
