import pytest
from expo_engine.statistics.power import (
    compute_binary_sample_size,
    compute_continuous_sample_size,
)


def test_continuous_sample_size_monotonicity():
    # As MDE increases, sample size must decrease
    res_small_mde = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=1.0)
    res_large_mde = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=2.0)
    assert res_large_mde.total_sample_size < res_small_mde.total_sample_size

    # As target power increases, sample size must increase
    res_p80 = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=1.0, power=0.80)
    res_p90 = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=1.0, power=0.90)
    assert res_p90.total_sample_size > res_p80.total_sample_size

    # As alpha decreases (stricter), sample size must increase
    res_a05 = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=1.0, alpha=0.05)
    res_a01 = compute_continuous_sample_size(baseline_variance=100.0, mde_absolute=1.0, alpha=0.01)
    assert res_a01.total_sample_size > res_a05.total_sample_size


def test_continuous_sample_size_cuped_projection():
    res = compute_continuous_sample_size(
        baseline_variance=100.0, mde_absolute=1.0, cuped_rho=0.60
    )
    assert res.cuped_projected_total is not None
    # 1 - 0.6^2 = 0.64 factor
    expected_projected = int(round(res.total_sample_size * 0.64))
    assert abs(res.cuped_projected_total - expected_projected) <= 2


def test_binary_sample_size_known_reference():
    # Baseline 10%, MDE +2% (target 12%), alpha=0.05, power=0.80
    res = compute_binary_sample_size(baseline_rate=0.10, mde_absolute=0.02, alpha=0.05, power=0.80)
    # Standard rule of thumb: ~3,700 - 3,900 per arm
    assert 3500 <= res.control_sample_size <= 4200
    assert res.treatment_sample_size == res.control_sample_size
    assert res.total_sample_size == res.control_sample_size * 2


def test_power_duration_calculation():
    res = compute_continuous_sample_size(
        baseline_variance=100.0, mde_absolute=2.0, daily_sample_rate=500.0
    )
    assert res.estimated_duration_days is not None
    assert res.estimated_duration_days > 0


def test_power_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="strictly positive"):
        compute_continuous_sample_size(baseline_variance=-1.0, mde_absolute=1.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_continuous_sample_size(baseline_variance=10.0, mde_absolute=1.0, alpha=1.5)
    with pytest.raises(ValueError, match="baseline_rate must be in"):
        compute_binary_sample_size(baseline_rate=1.5, mde_absolute=0.02)
