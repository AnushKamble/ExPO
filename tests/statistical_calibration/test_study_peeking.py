import pytest
from expo_engine.simulation.runners import run_peeking_simulation


@pytest.mark.calibration
def test_peeking_simulation_smoke():
    # Smoke run of Study A under null (delta=0)
    res_null = run_peeking_simulation(preset="smoke", seed=42, delta_true=0.0, alpha=0.05)

    # 1. Fixed horizon Welch Type I error should be roughly ~0.05
    fixed_rate = res_null["fixed_horizon"]["rejection_rate"]
    assert 0.01 <= fixed_rate <= 0.10

    # 2. Naive daily peeking should inflate Type I error above alpha (typically >= 15-20%)
    naive_rate = res_null["naive_daily_peeking"]["rejection_rate"]
    assert naive_rate > fixed_rate
    assert naive_rate >= 0.12

    # 3. Gaussian mixture sequential should control Type I error (<= alpha within tolerance)
    seq_rate = res_null["gaussian_mixture_sequential"]["rejection_rate"]
    assert seq_rate <= 0.08
