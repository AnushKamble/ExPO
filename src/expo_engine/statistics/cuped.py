"""
CUPED and ANCOVA Covariate-Adjustment Engine.
Provides optimal pre-experiment linear adjustment, heteroscedasticity-robust ANCOVA cross-checks,
and predeclared missing covariate policies.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from expo_engine.reporting.schemas import CUPEDResult, WelchResult
from expo_engine.statistics.welch import compute_welch_ttest


def compute_cuped(
    df: pd.DataFrame,
    metric_col: str,
    pre_covariate_col: str,
    variant_col: str = "variant",
    control_variant: str = "control",
    treatment_variant: str = "treatment",
    alpha: float = 0.05,
    missing_policy: str = "complete_case",
) -> CUPEDResult:
    """
    Computes CUPED-adjusted treatment effects and robust ANCOVA regression.
    """
    warnings: List[str] = []

    # 1. Unadjusted analysis on all valid outcome records
    valid_outcome_mask = np.isfinite(df[metric_col])
    all_valid_df = df[valid_outcome_mask]
    
    y_c_all = all_valid_df.loc[all_valid_df[variant_col] == control_variant, metric_col].values
    y_t_all = all_valid_df.loc[all_valid_df[variant_col] == treatment_variant, metric_col].values
    
    unadjusted_welch = compute_welch_ttest(
        y_c_all, y_t_all, metric_name=metric_col, alpha=alpha, run_mann_whitney=True
    )
    unadjusted_count = len(all_valid_df)

    # 2. Complete cases filtering for covariate adjustment
    complete_mask = valid_outcome_mask & np.isfinite(df[pre_covariate_col])
    complete_df = df[complete_mask].copy()
    
    n_complete = len(complete_df)
    n_missing_cov = unadjusted_count - n_complete
    
    if n_missing_cov > 0:
        pct_miss = (n_missing_cov / unadjusted_count) * 100
        warnings.append(
            f"Pre-covariate '{pre_covariate_col}' is missing for {n_missing_cov} of {unadjusted_count} rows ({pct_miss:.1f}%). "
            "Policy 'complete_case' applied: adjusted estimate computed on complete cases only; unadjusted retains all valid outcomes."
        )

    if n_complete < 10:
        raise ValueError(f"Insufficient complete cases ({n_complete}) to perform CUPED adjustment.")

    y = complete_df[metric_col].values.astype(float)
    x = complete_df[pre_covariate_col].values.astype(float)
    is_treatment = (complete_df[variant_col] == treatment_variant).astype(int).values

    # Check covariate variance
    x_var = float(np.var(x, ddof=1))
    if x_var <= 1e-12:
        warnings.append(f"Pre-covariate '{pre_covariate_col}' has near-zero variance. CUPED adjustment skipped.")
        return CUPEDResult(
            metric_name=metric_col,
            covariate_name=pre_covariate_col,
            unadjusted_welch=unadjusted_welch,
            adjusted_ate=unadjusted_welch.absolute_effect,
            adjusted_se=unadjusted_welch.standard_error,
            adjusted_ci_lower=unadjusted_welch.ci_lower,
            adjusted_ci_upper=unadjusted_welch.ci_upper,
            adjusted_p_value=unadjusted_welch.p_value,
            is_statistically_significant=unadjusted_welch.is_statistically_significant,
            theta=0.0,
            sample_correlation=0.0,
            theoretical_variance_reduction=0.0,
            empirical_variance_reduction=0.0,
            complete_cases_count=n_complete,
            missing_covariate_count=n_missing_cov,
            unadjusted_all_valid_count=unadjusted_count,
            ancova_robust_se=unadjusted_welch.standard_error,
            ancova_p_value=unadjusted_welch.p_value,
            warnings=warnings,
        )

    # 3. Compute optimal theta and sample correlation
    cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
    theta = cov_yx / x_var
    
    y_var = float(np.var(y, ddof=1))
    rho = cov_yx / np.sqrt(x_var * y_var) if y_var > 1e-12 else 0.0
    theoretical_reduction = float(rho ** 2)

    # 4. CUPED transform: Y_adj = Y - theta * (X - mean(X))
    x_mean = float(np.mean(x))
    y_adj = y - theta * (x - x_mean)
    
    complete_df["y_adj"] = y_adj
    y_adj_c = complete_df.loc[complete_df[variant_col] == control_variant, "y_adj"].values
    y_adj_t = complete_df.loc[complete_df[variant_col] == treatment_variant, "y_adj"].values

    # Welch test on adjusted outcomes
    welch_adj = compute_welch_ttest(
        y_adj_c, y_adj_t, metric_name=f"{metric_col}_cuped", alpha=alpha, run_mann_whitney=False
    )

    # Empirical variance reduction
    var_adj = float(np.var(y_adj, ddof=1))
    empirical_reduction = max(0.0, float(1.0 - (var_adj / y_var))) if y_var > 1e-12 else 0.0

    # 5. Robust ANCOVA regression cross-check: Y = beta_0 + tau * T + beta_1 * (X - mean(X)) + eps
    X_reg = np.column_stack([np.ones(n_complete), is_treatment, x - x_mean])
    ols_model = sm.OLS(y, X_reg)
    ols_res = ols_model.fit(cov_type="HC1")  # heteroscedasticity-robust SE
    
    ancova_ate = float(ols_res.params[1])
    ancova_robust_se = float(ols_res.bse[1])
    ancova_p_value = float(ols_res.pvalues[1])

    # Adjusted confidence interval from robust ANCOVA (or CUPED Welch)
    # We use ANCOVA robust SE as primary or cross-check
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_complete - 3))
    adj_ci_lower = float(ancova_ate - t_crit * ancova_robust_se)
    adj_ci_upper = float(ancova_ate + t_crit * ancova_robust_se)

    return CUPEDResult(
        metric_name=metric_col,
        covariate_name=pre_covariate_col,
        unadjusted_welch=unadjusted_welch,
        adjusted_ate=ancova_ate,
        adjusted_se=ancova_robust_se,
        adjusted_ci_lower=adj_ci_lower,
        adjusted_ci_upper=adj_ci_upper,
        adjusted_p_value=ancova_p_value,
        is_statistically_significant=bool(ancova_p_value < alpha),
        theta=theta,
        sample_correlation=rho,
        theoretical_variance_reduction=theoretical_reduction,
        empirical_variance_reduction=empirical_reduction,
        complete_cases_count=n_complete,
        missing_covariate_count=n_missing_cov,
        unadjusted_all_valid_count=unadjusted_count,
        ancova_robust_se=ancova_robust_se,
        ancova_p_value=ancova_p_value,
        warnings=warnings,
    )
