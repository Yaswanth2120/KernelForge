import pytest

from kernelforge import metrics


def test_accounting_hand_calculations():
    # Softmax: 2 reads/writes * 3 rows * 5 columns * 2 bytes = 60.
    assert metrics.softmax_bytes(3, 5, 2) == 60
    # LayerNorm: 60 tensor bytes + 5*(4+4) affine bytes + 3*2*4 stats bytes = 124.
    assert metrics.layer_norm_bytes(3, 5, 2, 4, 4, True) == 124
    assert metrics.layer_norm_bytes(3, 5, 2) == 60
    assert metrics.layer_norm_bytes(3, 5, 2, bias_elem=4) == 80
    # Attention uses the registered approximation: 2*2*3*5²*64 = 19,200 FLOPs.
    assert metrics.attention_flops(2, 3, 5, 64) == 19200
    # 1e9 bytes / .002 seconds = 500 GB/s; 2e12 FLOPs / .004 seconds = 500 TFLOPS.
    assert metrics.gbps(1_000_000_000, 2) == 500
    assert metrics.tflops(2_000_000_000_000, 4) == 500
    assert metrics.ratio_to_copy(800, 1000) == 0.8
    assert metrics.percent_of_peak(156) == 50
    assert metrics.softmax_bytes(0, 31, 2) == 0


@pytest.mark.parametrize('bad', [-1, 1.5, True])
def test_invalid_counts(bad):
    with pytest.raises(ValueError):
        metrics.softmax_bytes(bad, 3, 2)
    with pytest.raises(ValueError):
        metrics.attention_flops(1, 1, bad, 32)


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_invalid_rates(bad):
    with pytest.raises(ValueError):
        metrics.gbps(100, bad)
    with pytest.raises(ValueError):
        metrics.ratio_to_copy(10, bad)
