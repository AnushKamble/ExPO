"""
Sample Ratio Mismatch (SRM) checks and traffic integrity verification.
Implements Pearson chi-square goodness-of-fit and exact binomial tests for small counts.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class SRMResult(BaseModel):
    status: str = Field(..., description="'PASS', 'SRM_DETECTED', or 'WARNING'")
    p_value: float = Field(..., description="p-value from goodness-of-fit test")
    statistic: float = Field(..., description="Chi-square test statistic or binomial count")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom (K - 1)")
    threshold: float = Field(0.001, description="Threshold for flagging SRM alert")
    observed_counts: Dict[str, int] = Field(default_factory=dict)
    expected_counts: Dict[str, float] = Field(default_factory=dict)
    observed_shares: Dict[str, float] = Field(default_factory=dict)
    expected_shares: Dict[str, float] = Field(default_factory=dict)
    method: str = Field(..., description="Test used: 'chi_square' or 'exact_binomial'")
    level: str = Field("assignment", description="'assignment' or 'exposure'")
    message: str = Field(..., description="Human-readable audit message")


def check_srm(
    df: pd.DataFrame,
    variant_col: str,
    planned_ratio: Dict[str, float],
    threshold: float = 0.001,
    level: str = "assignment",
) -> SRMResult:
    """
    Evaluates Sample Ratio Mismatch against planned allocation proportions.
    Uses exact binomial test when expected count per arm is under 10 (for 2 arms),
    and Pearson's chi-square test for large samples.
    """
    variants = list(planned_ratio.keys())
    counts = df[variant_col].value_counts().to_dict()
    total_n = len(df)

    if total_n == 0:
        raise ValueError("Cannot perform SRM test on an empty dataframe.")

    obs_counts = {v: int(counts.get(v, 0)) for v in variants}
    exp_counts = {v: float(total_n * planned_ratio[v]) for v in variants}
    obs_shares = {v: float(obs_counts[v] / total_n) for v in variants}
    exp_shares = {v: float(planned_ratio[v]) for v in variants}

    k = len(variants)
    df_stat = k - 1
    min_exp = min(exp_counts.values())

    # Small sample path: exact binomial for two arms
    if k == 2 and min_exp < 10.0:
        method = "exact_binomial"
        v0, v1 = variants[0], variants[1]
        k_obs = obs_counts[v0]
        p_exp = planned_ratio[v0]
        # Two-sided binomial test
        res = stats.binomtest(k=k_obs, n=total_n, p=p_exp, alternative="two-sided")
        p_val = float(res.pvalue)
        stat = float(k_obs)
    else:
        method = "chi_square"
        o = np.array([obs_counts[v] for v in variants], dtype=float)
        e = np.array([exp_counts[v] for v in variants], dtype=float)

        # Pearson chi-square statistic: sum((O - E)^2 / E)
        chi2_stat = float(np.sum((o - e) ** 2 / e))
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df=df_stat))
        stat = chi2_stat

    # Determine status and message
    is_srm = (p_val < threshold)
    if is_srm:
        status = "SRM_DETECTED"
        message = (
            f"SAMPLE RATIO MISMATCH DETECTED ({level} level, p={p_val:.2e} < {threshold}). "
            f"Observed allocation significantly deviates from planned ratio. "
            "Causal interpretation of treatment effects is unsafe."
        )
    elif min_exp < 10.0:
        status = "WARNING"
        message = (
            f"Sample size is very small (minimum expected count = {min_exp:.1f} < 10). "
            f"SRM test p={p_val:.4f} did not reject, but test power is limited."
        )
    else:
        status = "PASS"
        message = (
            f"No Sample Ratio Mismatch detected ({level} level, p={p_val:.4f} >= {threshold}). "
            "Observed traffic matches planned allocation within expected random variation."
        )

    return SRMResult(
        status=status,
        p_value=p_val,
        statistic=stat,
        degrees_of_freedom=df_stat,
        threshold=threshold,
        observed_counts=obs_counts,
        expected_counts=exp_counts,
        observed_shares=obs_shares,
        expected_shares=exp_shares,
        method=method,
        level=level,
        message=message,
    )
