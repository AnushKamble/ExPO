import numpy as np
import pandas as pd
from expo_engine.data.synthetic import (
    generate_null_normal,
    generate_effect_normal,
    generate_null_heavy_tailed,
    generate_conversion,
    generate_ratio_metric,
    generate_srm_injected,
)


def test_null_normal_determinism_and_properties():
    df1, meta1 = generate_null_normal(n_users=500, seed=123, rho=0.7)
    df2, meta2 = generate_null_normal(n_users=500, seed=123, rho=0.7)
    pd.testing.assert_frame_equal(df1, df2)
    assert meta1["true_ate"] == 0.0
    assert meta1["target_correlation"] == 0.7
    assert "user_id" in df1.columns
    assert "metric_value" in df1.columns
    assert "pre_metric" in df1.columns

    # Test missing covariate injection
    df_miss, _ = generate_null_normal(n_users=500, seed=123, missing_covariate_rate=0.20)
    missing_pct = df_miss["pre_metric"].isna().mean()
    assert 0.14 <= missing_pct <= 0.26


def test_effect_normal_ground_truth():
    delta_target = 2.5
    df, meta = generate_effect_normal(n_users=10000, seed=42, delta=delta_target, mu_y=100.0)
    assert meta["true_ate"] == delta_target

    # Check empirical mean difference is close to true delta
    means = df.groupby("variant")["metric_value"].mean()
    empirical_diff = means["treatment"] - means["control"]
    assert abs(empirical_diff - delta_target) < 0.5


def test_null_heavy_tailed():
    df, meta = generate_null_heavy_tailed(n_users=1000, seed=42, df_student=3.0)
    assert meta["normality_assumption_valid"] is False
    assert len(df) == 1000
    assert df["metric_value"].isna().sum() == 0


def test_conversion():
    df, meta = generate_conversion(n_users=5000, seed=42, p_control=0.10, delta=0.03)
    assert meta["true_ate_pp"] == 0.03
    rates = df.groupby("variant")["conversion"].mean()
    assert abs(rates["control"] - 0.10) < 0.02
    assert abs(rates["treatment"] - 0.13) < 0.02


def test_ratio_metric():
    df, meta = generate_ratio_metric(n_users=2000, seed=42, click_prob_control=0.08, delta_click_prob=0.02)
    assert "numerator" in df.columns
    assert "denominator" in df.columns
    assert (df["denominator"] > 0).all()
    assert (df["numerator"] <= df["denominator"]).all()
    assert meta["true_ratio_diff"] == 0.02


def test_srm_injected():
    df, meta = generate_srm_injected(n_users=5000, seed=42, realized_treatment_prob=0.55)
    assert meta["srm_corrupted"] is True
    share_treatment = (df["variant"] == "treatment").mean()
    assert abs(share_treatment - 0.55) < 0.03
