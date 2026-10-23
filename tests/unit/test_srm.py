import pandas as pd
import pytest
from expo_engine.checks.srm import check_srm


def test_srm_balanced_passes():
    # 500 control, 500 treatment
    df = pd.DataFrame({"variant": ["control"] * 500 + ["treatment"] * 500})
    res = check_srm(df, variant_col="variant", planned_ratio={"control": 0.5, "treatment": 0.5})
    assert res.status == "PASS"
    assert res.p_value > 0.05
    assert res.statistic == pytest.approx(0.0, abs=1e-5)


def test_srm_imbalanced_detected():
    # 600 control, 400 treatment (severe SRM for N=1000)
    df = pd.DataFrame({"variant": ["control"] * 600 + ["treatment"] * 400})
    res = check_srm(df, variant_col="variant", planned_ratio={"control": 0.5, "treatment": 0.5})
    assert res.status == "SRM_DETECTED"
    assert res.p_value < 0.001
    assert "SAMPLE RATIO MISMATCH DETECTED" in res.message


def test_srm_custom_allocation():
    # Planned 80% control, 20% treatment; observed exactly 800 and 200
    df = pd.DataFrame({"variant": ["control"] * 800 + ["treatment"] * 200})
    res = check_srm(df, variant_col="variant", planned_ratio={"control": 0.8, "treatment": 0.2})
    assert res.status == "PASS"
    assert res.p_value > 0.90


def test_srm_multi_arm():
    # 3 arms: A, B, C (1/3 each)
    df = pd.DataFrame({"variant": ["A"] * 333 + ["B"] * 333 + ["C"] * 334})
    res = check_srm(df, variant_col="variant", planned_ratio={"A": 1/3, "B": 1/3, "C": 1/3})
    assert res.status == "PASS"
    assert res.degrees_of_freedom == 2


def test_srm_small_expected_counts_exact_test():
    # Very small sample (N=16, expected 8 and 8 < 10)
    df = pd.DataFrame({"variant": ["control"] * 2 + ["treatment"] * 14})
    res = check_srm(df, variant_col="variant", planned_ratio={"control": 0.5, "treatment": 0.5})
    assert res.method == "exact_binomial"
    # Binomial p-value for 2 out of 16 with p=0.5 is 2 * sum(binom(16, 0..2)*0.5^16) = 0.00417
    assert res.p_value == pytest.approx(0.00417, abs=1e-3)
    assert res.status in ("PASS", "SRM_DETECTED", "WARNING")
