import pytest
from expo_engine.simulation.runners import run_srm_simulation


@pytest.mark.calibration
def test_srm_simulation_smoke():
    res = run_srm_simulation(
        preset="smoke",
        seed=42,
        sample_sizes=(1000, 5000),
        treatment_probs=(0.50, 0.55),
    )
    matrix = res["matrix_results"]
    assert len(matrix) == 4

    # Under true 0.50: detection rate should be roughly <= alpha_srm (0.001) or 0 in smoke
    null_small = [m for m in matrix if m["sample_size"] == 1000 and m["realized_treatment_prob"] == 0.50][0]
    assert null_small["detection_power"] <= 0.05

    # Under 0.55 with N=5000: detection power should be very high (> 0.70)
    alt_large = [m for m in matrix if m["sample_size"] == 5000 and m["realized_treatment_prob"] == 0.55][0]
    assert alt_large["detection_power"] >= 0.70
