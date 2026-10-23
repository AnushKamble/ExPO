"""
Experiment design, power planning, sample size, and MDE calculators.
Supports continuous and binary metrics with flexible allocation ratios and CUPED projections.
"""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class PowerPlanningResult(BaseModel):
    metric_kind: str
    target_power: float
    alpha: float
    mde_absolute: float
    mde_relative: Optional[float] = None
    control_sample_size: int
    treatment_sample_size: int
    total_sample_size: int
    allocation_ratio: float
    cuped_projected_total: Optional[int] = None
    cuped_correlation_assumed: Optional[float] = None
    cuped_note: Optional[str] = None
    daily_sample_rate: Optional[float] = None
    estimated_duration_days: Optional[int] = None
    method_notes: str


def compute_continuous_sample_size(
    baseline_variance: float,
    mde_absolute: float,
    baseline_mean: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,  # kappa = n_T / n_C
    cuped_rho: Optional[float] = None,
    daily_sample_rate: Optional[float] = None,
) -> PowerPlanningResult:
    """
    Computes required sample size for continuous outcomes under normal approximation.
    """
    if baseline_variance <= 0:
        raise ValueError(f"baseline_variance must be strictly positive, got {baseline_variance}")
    if mde_absolute <= 0:
        raise ValueError(f"mde_absolute must be strictly positive, got {mde_absolute}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")
    if allocation_ratio <= 0:
        raise ValueError(f"allocation_ratio must be strictly positive, got {allocation_ratio}")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    # Standard formula for unequal allocation:
    # n_C = (1 + 1/kappa) * sigma^2 * (z_alpha + z_beta)^2 / delta^2
    n_c_raw = (1.0 + (1.0 / allocation_ratio)) * baseline_variance * ((z_alpha + z_beta) ** 2) / (mde_absolute ** 2)
    n_c = int(np.ceil(n_c_raw))
    n_t = int(np.ceil(n_c * allocation_ratio))
    n_total = n_c + n_t

    rel_mde = (mde_absolute / baseline_mean) if (baseline_mean and abs(baseline_mean) > 1e-7) else None

    # CUPED projection
    cuped_total = None
    cuped_note = None
    if cuped_rho is not None:
        if not (-1.0 <= cuped_rho <= 1.0):
            raise ValueError(f"cuped_rho must be in [-1, 1], got {cuped_rho}")
        variance_factor = max(0.0, 1.0 - (cuped_rho ** 2))
        cuped_total = int(np.ceil(n_total * variance_factor))
        cuped_note = (
            f"Projected sample size assuming pre-experiment correlation rho={cuped_rho:.2f} "
            f"(theoretical variance factor 1 - rho^2 = {variance_factor:.2f}). "
            "This is a planning projection, not an achieved guarantee."
        )

    # Estimated duration
    duration_days = None
    if daily_sample_rate is not None and daily_sample_rate > 0:
        duration_days = int(np.ceil(n_total / daily_sample_rate))

    notes = (
        f"Continuous two-sample normal approximation (alpha={alpha}, power={power}, "
        f"allocation ratio={allocation_ratio}). Assumes independent observations and equal variance."
    )

    return PowerPlanningResult(
        metric_kind="continuous_mean",
        target_power=power,
        alpha=alpha,
        mde_absolute=mde_absolute,
        mde_relative=rel_mde,
        control_sample_size=n_c,
        treatment_sample_size=n_t,
        total_sample_size=n_total,
        allocation_ratio=allocation_ratio,
        cuped_projected_total=cuped_total,
        cuped_correlation_assumed=cuped_rho,
        cuped_note=cuped_note,
        daily_sample_rate=daily_sample_rate,
        estimated_duration_days=duration_days,
        method_notes=notes,
    )


def compute_binary_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    daily_sample_rate: Optional[float] = None,
) -> PowerPlanningResult:
    """
    Computes required sample size for binary proportions under standard normal approximation.
    """
    if not (0.0 < baseline_rate < 1.0):
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    if mde_absolute <= 0:
        raise ValueError(f"mde_absolute must be strictly positive, got {mde_absolute}")
    p1 = baseline_rate + mde_absolute
    if p1 >= 1.0 or p1 <= 0.0:
        raise ValueError(f"Target rate p1 = baseline ({baseline_rate}) + mde ({mde_absolute}) = {p1} is outside (0, 1)")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)

    p0 = baseline_rate
    p_bar = (p0 + p1) / 2.0

    # Equal allocation standard formula
    # n = [z_alpha * sqrt(2*p_bar*(1-p_bar)) + z_beta * sqrt(p0*(1-p0) + p1*(1-p1))]^2 / (p1 - p0)^2
    term1 = z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
    term2 = z_beta * np.sqrt(p0 * (1.0 - p0) + p1 * (1.0 - p1))
    n_per_arm_raw = ((term1 + term2) ** 2) / (mde_absolute ** 2)
    
    # Adjust for allocation ratio if unequal
    if allocation_ratio == 1.0:
        n_c = int(np.ceil(n_per_arm_raw))
        n_t = n_c
    else:
        # Generalized variance weighting
        n_c = int(np.ceil(n_per_arm_raw * (1.0 + 1.0 / allocation_ratio) / 2.0))
        n_t = int(np.ceil(n_c * allocation_ratio))

    n_total = n_c + n_t
    rel_mde = mde_absolute / baseline_rate

    duration_days = None
    if daily_sample_rate is not None and daily_sample_rate > 0:
        duration_days = int(np.ceil(n_total / daily_sample_rate))

    notes = (
        f"Binary conversion normal approximation (baseline={baseline_rate:.2%}, target={p1:.2%}, "
        f"alpha={alpha}, power={power}). Assumes independent Bernoulli trials."
    )

    return PowerPlanningResult(
        metric_kind="binary_mean",
        target_power=power,
        alpha=alpha,
        mde_absolute=mde_absolute,
        mde_relative=rel_mde,
        control_sample_size=n_c,
        treatment_sample_size=n_t,
        total_sample_size=n_total,
        allocation_ratio=allocation_ratio,
        cuped_projected_total=None,
        cuped_correlation_assumed=None,
        cuped_note="CUPED projection not applicable to pure binary conversion without continuous pre-covariate.",
        daily_sample_rate=daily_sample_rate,
        estimated_duration_days=duration_days,
        method_notes=notes,
    )
