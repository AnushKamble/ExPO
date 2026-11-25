import pytest
from expo_engine.simulation.runners import run_aa_simulation


@pytest.mark.calibration
def test_aa_simulation_smoke():
    res = run_aa_simulation(preset="smoke", seed=42, n_users_per_rep=500, alpha=0.05)

    fpr = res["observed_type_1_error"]
    # 200 reps: 95% Wilson CI covers alpha=0.05
    assert res["wilson_ci_lower"] <= 0.05 <= res["wilson_ci_upper"] or abs(fpr - 0.05) < 0.04

    # 10 deciles of p-values should all have observations
    decile_counts = res["p_value_decile_counts"]
    assert len(decile_counts) == 10
    assert all(c > 0 for c in decile_counts)
