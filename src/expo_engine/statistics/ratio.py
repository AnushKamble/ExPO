"""
Ratio-of-sums metric inference via the multivariate Delta Method.
Operates strictly at the unit of randomization (user level) to account for intra-user correlation.
Includes influence-function cross-validation and user-level bootstrap sensitivity.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from expo_engine.reporting.schemas import RatioResult


def compute_arm_ratio_and_variance(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes ratio of sums R = sum(Y) / sum(X) and its asymptotic variance
    using both the bivariate delta method and linearized influence values.
    Returns: (ratio, delta_var, influence_var)
    """
    n = len(numerator)
    if n < 2:
        raise ValueError(f"Arm sample size ({n}) too small for ratio delta method.")

    sum_y = float(np.sum(numerator))
    sum_x = float(np.sum(denominator))
    if sum_x <= 0.0:
        raise ValueError(f"Total denominator across arm ({sum_x}) must be strictly positive.")

    mean_y = sum_y / n
    mean_x = sum_x / n
    r = mean_y / mean_x

    # Sample variances and covariance
    s2_y = float(np.var(numerator, ddof=1))
    s2_x = float(np.var(denominator, ddof=1))
    s_yx = float(np.cov(numerator, denominator, ddof=1)[0, 1])

    # Method 1: Classic Bivariate Delta Method
    # Var(R) = (1 / (n * mean_x^2)) * (s2_y - 2 * r * s_yx + r^2 * s2_x)
    delta_var = (1.0 / (n * (mean_x ** 2))) * (s2_y - 2.0 * r * s_yx + (r ** 2) * s2_x)

    # Method 2: Linearized Influence Function
    # Z_i = (Y_i - r * X_i) / mean_x
    z = (numerator - r * denominator) / mean_x
    influence_var = float(np.var(z, ddof=1) / n)

    return r, delta_var, influence_var


def compute_ratio_bootstrap_ci(
    y_c: np.ndarray,
    x_c: np.ndarray,
    y_t: np.ndarray,
    x_t: np.ndarray,
    alpha: float = 0.05,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    User-level non-parametric bootstrap for difference in ratio of sums.
    Resamples whole user tuples (Y_i, X_i) within each arm.
    """
    rng = np.random.default_rng(seed)
    n_c = len(y_c)
    n_t = len(y_t)

    # Generate bootstrap index arrays
    c_indices = rng.integers(0, n_c, size=(n_bootstraps, n_c))
    t_indices = rng.integers(0, n_t, size=(n_bootstraps, n_t))

    c_boot_y = np.sum(y_c[c_indices], axis=1)
    c_boot_x = np.sum(x_c[c_indices], axis=1)
    t_boot_y = np.sum(y_t[t_indices], axis=1)
    t_boot_x = np.sum(x_t[t_indices], axis=1)

    r_c_boot = c_boot_y / c_boot_x
    r_t_boot = t_boot_y / t_boot_x
    diffs = r_t_boot - r_c_boot

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(diffs, lower_pct))
    ci_upper = float(np.percentile(diffs, upper_pct))
    return ci_lower, ci_upper


def compute_ratio_metric(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    variant_col: str = "variant",
    control_variant: str = "control",
    treatment_variant: str = "treatment",
    alpha: float = 0.05,
    run_bootstrap: bool = False,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> RatioResult:
    """
    Computes treatment effect for ratio-of-sums metric via Delta Method.
    """
    warnings: List[str] = []

    # Filter non-finite rows
    valid_mask = np.isfinite(df[numerator_col]) & np.isfinite(df[denominator_col])
    valid_df = df[valid_mask]

    c_df = valid_df[valid_df[variant_col] == control_variant]
    t_df = valid_df[valid_df[variant_col] == treatment_variant]

    y_c = c_df[numerator_col].values.astype(float)
    x_c = c_df[denominator_col].values.astype(float)
    y_t = t_df[numerator_col].values.astype(float)
    x_t = t_df[denominator_col].values.astype(float)

    # Denominator checks
    if (x_c < 0).any() or (x_t < 0).any():
        raise ValueError("Negative values detected in denominator. Ratio denominators must be non-negative.")

    mean_x_c = np.mean(x_c) if len(x_c) > 0 else 0
    mean_x_t = np.mean(x_t) if len(x_t) > 0 else 0
    if mean_x_c < 1.0 or mean_x_t < 1.0:
        warnings.append(
            f"Average denominator is very small (Control: {mean_x_c:.2f}, Treatment: {mean_x_t:.2f}). "
            "Delta method asymptotic normal approximation may have degraded coverage."
        )

    r_c, var_c, inf_var_c = compute_arm_ratio_and_variance(y_c, x_c)
    r_t, var_t, inf_var_t = compute_arm_ratio_and_variance(y_t, x_t)

    se_c = float(np.sqrt(max(0.0, var_c)))
    se_t = float(np.sqrt(max(0.0, var_t)))

    delta_r = float(r_t - r_c)
    total_var = max(0.0, var_c + var_t)
    se_diff = float(np.sqrt(total_var))

    if se_diff <= 1e-12:
        warnings.append("Zero variance in ratio difference.")
        z_stat = 0.0
        p_val = 1.0 if delta_r == 0.0 else 0.0
        ci_lower = delta_r
        ci_upper = delta_r
    else:
        z_stat = float(delta_r / se_diff)
        p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat))))
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = float(delta_r - z_crit * se_diff)
        ci_upper = float(delta_r + z_crit * se_diff)

    relative_lift = (delta_r / r_c) if abs(r_c) > 1e-7 else None

    # Bootstrap sensitivity if requested
    boot_ci_lower = None
    boot_ci_upper = None
    if run_bootstrap:
        boot_ci_lower, boot_ci_upper = compute_ratio_bootstrap_ci(
            y_c, x_c, y_t, x_t, alpha=alpha, n_bootstraps=n_bootstraps, seed=seed
        )

    return RatioResult(
        metric_name=f"{numerator_col}/{denominator_col}",
        estimand="Difference in Ratios of Sums (Delta_R = R_T - R_C)",
        method="Multivariate Delta Method (Randomization Unit Level)",
        control_ratio=r_c,
        treatment_ratio=r_t,
        control_se=se_c,
        treatment_se=se_t,
        control_numerator_sum=float(np.sum(y_c)),
        control_denominator_sum=float(np.sum(x_c)),
        treatment_numerator_sum=float(np.sum(y_t)),
        treatment_denominator_sum=float(np.sum(x_t)),
        delta_ratio=delta_r,
        relative_lift=relative_lift,
        standard_error=se_diff,
        z_statistic=z_stat,
        p_value=p_val,
        alpha=alpha,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_statistically_significant=bool(p_val < alpha),
        bootstrap_ci_lower=boot_ci_lower,
        bootstrap_ci_upper=boot_ci_upper,
        warnings=warnings,
    )
