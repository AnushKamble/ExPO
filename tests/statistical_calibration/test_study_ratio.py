import pytest
from expo_engine.simulation.runners import run_ratio_simulation


@pytest.mark.calibration
def test_ratio_simulation_smoke():
    res = run_ratio_simulation(preset="smoke", seed=42, n_users=800)
    # Empirical coverage should be close to 95%
    cov = res["empirical_coverage"]
    assert 0.88 <= cov <= 1.0
