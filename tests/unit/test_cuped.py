import numpy as np
import pandas as pd
import pytest
from expo_engine.data.synthetic import generate_effect_normal, generate_null_normal
from expo_engine.statistics.cuped import compute_cuped


def test_cuped_constant_covariate():
    # Pre-covariate is identical constant for all users
    df = pd.DataFrame({
        "variant": ["control"] * 50 + ["treatment"] * 50,
        "revenue": np.concatenate([np.random.normal(10, 2, 50), np.random.normal(12, 2, 50)]),
        "pre_cov": [100.0] * 100,
    })
    res = compute_cuped(df, metric_col="revenue", pre_covariate_col="pre_cov")
    assert res.theta == 0.0
    assert res.theoretical_variance_reduction == 0.0
    assert any("near-zero variance" in w for w in res.warnings)


def test_cuped_high_correlation_variance_reduction():
    # 5000 users with rho = 0.8
    df, meta = generate_effect_normal(
        n_users=5000, seed=123, delta=2.0, rho=0.8, mu_y=50.0, sigma_y=10.0
    )
    res = compute_cuped(df, metric_col="metric_value", pre_covariate_col="pre_metric")

    # True rho is 0.8 -> theoretical reduction ~ 0.64
    assert res.sample_correlation == pytest.approx(0.8, abs=0.05)
    assert res.theoretical_variance_reduction == pytest.approx(0.64, abs=0.08)
    assert res.empirical_variance_reduction > 0.50

    # Adjusted standard error must be substantially smaller than unadjusted
    assert res.adjusted_se < res.unadjusted_welch.standard_error * 0.70

    # Effect estimate remains close to true delta = 2.0
    assert abs(res.adjusted_ate - 2.0) < 0.40


def test_cuped_zero_correlation():
    df, meta = generate_null_normal(n_users=3000, seed=42, rho=0.0)
    res = compute_cuped(df, metric_col="metric_value", pre_covariate_col="pre_metric")
    assert abs(res.sample_correlation) < 0.05
    assert res.empirical_variance_reduction < 0.05
    assert abs(res.adjusted_ate - 0.0) < 0.5


def test_cuped_missingness_policy():
    # Inject 20% missing covariates
    df, _ = generate_effect_normal(
        n_users=2000, seed=99, delta=1.5, missing_covariate_rate=0.20
    )
    res = compute_cuped(df, metric_col="metric_value", pre_covariate_col="pre_metric")

    assert res.unadjusted_all_valid_count == 2000
    assert 300 <= res.missing_covariate_count <= 500
    assert res.complete_cases_count == 2000 - res.missing_covariate_count
    assert any("Policy 'complete_case' applied" in w for w in res.warnings)
