import numpy as np
import pandas as pd
import pytest
from expo_engine.data.synthetic import generate_ratio_metric
from expo_engine.statistics.ratio import (
    compute_arm_ratio_and_variance,
    compute_ratio_metric,
)


def test_ratio_influence_function_equivalence():
    rng = np.random.default_rng(42)
    # 200 observations with arbitrary positive numerator and denominator
    y = rng.gamma(shape=2.0, scale=3.0, size=200)
    x = rng.poisson(lam=5.0, size=200) + 1.0

    r, delta_var, influence_var = compute_arm_ratio_and_variance(y, x)
    assert r > 0
    # Both variance formulas must match up to floating point precision
    assert delta_var == pytest.approx(influence_var, rel=1e-8)


def test_ratio_delta_method_known_fixture():
    # 3000 users with known true ratio difference = 0.02
    df, meta = generate_ratio_metric(
        n_users=3000, seed=123, click_prob_control=0.05, delta_click_prob=0.02
    )
    res = compute_ratio_metric(
        df,
        numerator_col="numerator",
        denominator_col="denominator",
        run_bootstrap=True,
        n_bootstraps=500,
        seed=123,
    )

    assert res.delta_ratio == pytest.approx(0.02, abs=0.01)
    assert res.is_statistically_significant is True
    assert res.ci_lower < 0.02 < res.ci_upper

    # Bootstrap CI should be close to delta method CI
    assert res.bootstrap_ci_lower is not None
    assert res.bootstrap_ci_upper is not None
    assert abs(res.bootstrap_ci_lower - res.ci_lower) < 0.005
    assert abs(res.bootstrap_ci_upper - res.ci_upper) < 0.005


def test_ratio_negative_denominator_error():
    df = pd.DataFrame({
        "variant": ["control", "treatment"],
        "num": [10, 20],
        "denom": [-5, 10],
    })
    with pytest.raises(ValueError, match="Negative values detected"):
        compute_ratio_metric(df, numerator_col="num", denominator_col="denom")


def test_ratio_zero_total_denominator_error():
    df = pd.DataFrame({
        "variant": ["control", "control", "treatment"],
        "num": [0, 0, 10],
        "denom": [0, 0, 10],
    })
    with pytest.raises(ValueError, match="strictly positive"):
        compute_ratio_metric(df, numerator_col="num", denominator_col="denom")


def test_ratio_small_denominator_warning():
    # Denominators are very small floats (< 1.0)
    df = pd.DataFrame({
        "variant": ["control"] * 50 + ["treatment"] * 50,
        "num": np.random.uniform(0, 0.5, 100),
        "denom": np.random.uniform(0.1, 0.4, 100),
    })
    res = compute_ratio_metric(df, numerator_col="num", denominator_col="denom")
    assert any("Average denominator is very small" in w for w in res.warnings)
