"""
Welch's unequal-variance two-sample inference for continuous and binary experimentation metrics.
Implements exact Welch-Satterthwaite degrees of freedom and clear lift semantics.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from expo_engine.reporting.schemas import GroupSummary, WelchResult


def compute_group_summary(values: np.ndarray, variant: str) -> GroupSummary:
    """
    Computes descriptive and dispersion statistics for an experimental arm.
    """
    clean_vals = values[np.isfinite(values)]
    n = len(clean_vals)
    if n == 0:
        raise ValueError(f"No valid observations for variant '{variant}'.")

    mean_val = float(np.mean(clean_vals))
    var_val = float(np.var(clean_vals, ddof=1)) if n > 1 else 0.0
    std_val = float(np.std(clean_vals, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(clean_vals))
    max_val = float(np.max(clean_vals))
    med_val = float(np.median(clean_vals))
    q75, q25 = np.percentile(clean_vals, [75, 25])
    iqr_val = float(q75 - q25)

    return GroupSummary(
        variant=variant,
        sample_size=n,
        mean=mean_val,
        variance=var_val,
        std_dev=std_val,
        min_val=min_val,
        max_val=max_val,
        median=med_val,
        iqr=iqr_val,
    )


def compute_welch_ttest(
    control_vals: Union[Sequence[float], np.ndarray, pd.Series],
    treatment_vals: Union[Sequence[float], np.ndarray, pd.Series],
    metric_name: str = "metric",
    alpha: float = 0.05,
    run_mann_whitney: bool = True,
) -> WelchResult:
    """
    Performs Welch's unequal-variance two-sample t-test.
    Returns a validated WelchResult.
    """
    y_c = np.asarray(control_vals, dtype=float)
    y_t = np.asarray(treatment_vals, dtype=float)

    # Clean non-finite
    y_c = y_c[np.isfinite(y_c)]
    y_t = y_t[np.isfinite(y_t)]

    n_c = len(y_c)
    n_t = len(y_t)
    if n_c < 2 or n_t < 2:
        raise ValueError(f"Insufficient sample size for Welch t-test (n_c={n_c}, n_t={n_t}). Minimum is 2 per arm.")

    c_summary = compute_group_summary(y_c, "control")
    t_summary = compute_group_summary(y_t, "treatment")

    # ATE
    delta = t_summary.mean - c_summary.mean
    
    # Relative lift semantics
    warnings: List[str] = []
    if abs(c_summary.mean) < 1e-7:
        relative_lift = None
        relative_lift_valid = False
        warnings.append("Control mean is near zero (|mean| < 1e-7). Relative lift is mathematically unstable.")
    else:
        relative_lift = delta / c_summary.mean
        relative_lift_valid = True

    # Welch standard error
    s2_c = c_summary.variance
    s2_t = t_summary.variance
    var_c_mean = s2_c / n_c
    var_t_mean = s2_t / n_t
    se = float(np.sqrt(var_c_mean + var_t_mean))

    # Zero variance edge case
    if se == 0.0:
        warnings.append("Both control and treatment outcomes have zero variance (constant values).")
        t_stat = 0.0
        df = float(n_c + n_t - 2)
        p_val = 1.0 if delta == 0.0 else 0.0
        ci_lower = delta
        ci_upper = delta
    else:
        t_stat = float(delta / se)
        # Welch-Satterthwaite degrees of freedom
        denom = (var_c_mean**2 / (n_c - 1)) + (var_t_mean**2 / (n_t - 1))
        df = float((var_c_mean + var_t_mean)**2 / denom) if denom > 0 else float(n_c + n_t - 2)
        # Two-sided p-value
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))
        # Confidence interval
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        ci_lower = float(delta - t_crit * se)
        ci_upper = float(delta + t_crit * se)

    is_sig = bool(p_val < alpha)

    # Optional Mann-Whitney U sensitivity test
    mw_p: Optional[float] = None
    mw_note: Optional[str] = None
    if run_mann_whitney and n_c > 0 and n_t > 0:
        try:
            mw_res = stats.mannwhitneyu(y_t, y_c, alternative="two-sided")
            mw_p = float(mw_res.pvalue)
            mw_note = "Non-parametric rank sum test (evaluates stochastic dominance/distribution shift, NOT difference in means)."
        except Exception:
            mw_p = None

    return WelchResult(
        metric_name=metric_name,
        estimand="Average Treatment Effect (Delta = E[Y_T] - E[Y_C])",
        method="Welch's Unequal-Variance Two-Sample t-Test",
        control_summary=c_summary,
        treatment_summary=t_summary,
        absolute_effect=delta,
        relative_lift=relative_lift,
        relative_lift_valid=relative_lift_valid,
        standard_error=se,
        t_statistic=t_stat,
        degrees_of_freedom=df,
        p_value=p_val,
        alpha=alpha,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_statistically_significant=is_sig,
        warnings=warnings,
        mann_whitney_p_value=mw_p,
        mann_whitney_note=mw_note,
    )
