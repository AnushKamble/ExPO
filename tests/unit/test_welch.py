import numpy as np
import pytest
from scipy import stats
from expo_engine.statistics.welch import compute_welch_ttest


def test_welch_known_answer_fixture():
    # Known dataset
    control = np.array([12.0, 14.0, 15.0, 11.0, 13.0, 16.0, 10.0])
    treatment = np.array([17.0, 19.0, 20.0, 18.0, 22.0, 16.0, 21.0, 19.0])

    res = compute_welch_ttest(control, treatment, metric_name="test_fixture", alpha=0.05)

    # Compare with scipy
    scipy_res = stats.ttest_ind(treatment, control, equal_var=False)

    assert res.t_statistic == pytest.approx(scipy_res.statistic, rel=1e-5)
    assert res.p_value == pytest.approx(scipy_res.pvalue, rel=1e-5)
    assert res.degrees_of_freedom == pytest.approx(scipy_res.df, rel=1e-5)
    assert res.absolute_effect > 0
    assert res.is_statistically_significant is True


def test_welch_relative_lift_zero_mean_warning():
    # Control mean is 0.0
    control = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    treatment = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

    res = compute_welch_ttest(control, treatment, metric_name="zero_mean")
    assert not res.relative_lift_valid
    assert res.relative_lift is None
    assert any("near zero" in w for w in res.warnings)


def test_welch_directionality():
    control = np.array([10.0, 12.0, 11.0])
    treatment_high = np.array([15.0, 16.0, 14.0])
    treatment_low = np.array([5.0, 6.0, 4.0])

    res_high = compute_welch_ttest(control, treatment_high)
    assert res_high.absolute_effect > 0
    assert res_high.t_statistic > 0

    res_low = compute_welch_ttest(control, treatment_low)
    assert res_low.absolute_effect < 0
    assert res_low.t_statistic < 0


def test_welch_unequal_variance_and_sizes():
    rng = np.random.default_rng(42)
    control = rng.normal(loc=10.0, scale=2.0, size=50)
    treatment = rng.normal(loc=12.0, scale=8.0, size=120)

    res = compute_welch_ttest(control, treatment)
    scipy_res = stats.ttest_ind(treatment, control, equal_var=False)

    assert res.t_statistic == pytest.approx(scipy_res.statistic, rel=1e-5)
    assert res.p_value == pytest.approx(scipy_res.pvalue, rel=1e-5)
    assert res.degrees_of_freedom == pytest.approx(scipy_res.df, rel=1e-5)
    assert res.ci_lower < res.absolute_effect < res.ci_upper


def test_welch_invalid_alpha_raises():
    control = np.array([10.0, 12.0, 11.0])
    treatment = np.array([15.0, 16.0, 14.0])
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_welch_ttest(control, treatment, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_welch_ttest(control, treatment, alpha=1.0)

