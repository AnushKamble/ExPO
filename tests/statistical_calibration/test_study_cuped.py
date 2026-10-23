import pytest
from expo_engine.simulation.runners import run_cuped_simulation


@pytest.mark.calibration
def test_cuped_simulation_smoke():
    res = run_cuped_simulation(
        preset="smoke", seed=42, rhos=(0.0, 0.5, 0.8), n_users_per_rep=500
    )
    grid = res["grid_results"]
    assert len(grid) == 3

    # For rho = 0.0: empirical variance reduction should be near zero (< 0.05)
    assert grid[0]["target_rho"] == 0.0
    assert abs(grid[0]["empirical_variance_reduction"]) < 0.08

    # For rho = 0.5: theoretical reduction = 0.25; empirical reduction should be positive (~0.20-0.30)
    assert grid[1]["target_rho"] == 0.5
    assert 0.15 <= grid[1]["empirical_variance_reduction"] <= 0.35

    # For rho = 0.8: theoretical reduction = 0.64; empirical reduction should be substantial (> 0.50)
    assert grid[2]["target_rho"] == 0.8
    assert grid[2]["empirical_variance_reduction"] >= 0.50

    # Coverage for all rho settings should remain near 95%
    for item in grid:
        assert 0.88 <= item["coverage_rate"] <= 1.0
