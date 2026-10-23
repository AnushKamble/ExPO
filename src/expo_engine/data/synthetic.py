"""
Synthetic experiment data generators with recorded ground truth parameters.
Uses numpy.random.Generator with explicit seeds for strict reproducibility.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def generate_null_normal(
    n_users: int = 10000,
    seed: int = 42,
    mu_y: float = 50.0,
    sigma_y: float = 10.0,
    mu_x: float = 45.0,
    sigma_x: float = 8.0,
    rho: float = 0.6,
    missing_covariate_rate: float = 0.0,
    experiment_id: str = "exp_null_normal",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a Gaussian A/B test with true treatment effect Delta = 0,
    and a correlated pre-treatment covariate X for CUPED evaluation.
    """
    rng = np.random.default_rng(seed)
    
    # Generate pre-treatment covariate X
    x = rng.normal(loc=mu_x, scale=sigma_x, size=n_users)
    
    # Generate potential outcome Y(0) correlated with X
    z = rng.normal(loc=0.0, scale=1.0, size=n_users)
    y0 = mu_y + rho * (sigma_y / sigma_x) * (x - mu_x) + np.sqrt(max(0.0, 1.0 - rho**2)) * sigma_y * z
    
    # Assignment: 50/50 balanced
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    # True treatment effect is exactly 0
    delta_true = 0.0
    y = y0.copy()
    
    # Apply missingness to pre-covariate if requested
    pre_covariate = x.copy()
    if missing_covariate_rate > 0:
        mask = rng.random(size=n_users) < missing_covariate_rate
        pre_covariate = np.where(mask, np.nan, pre_covariate)
        
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "metric_value": y,
        "pre_metric": pre_covariate,
    })
    
    metadata = {
        "scenario": "null_normal",
        "n_users": n_users,
        "seed": seed,
        "true_ate": delta_true,
        "true_relative_lift": 0.0,
        "target_correlation": rho,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "known_difference_variance": 2.0 * (sigma_y**2),  # for sequential testing
    }
    return df, metadata


def generate_effect_normal(
    n_users: int = 10000,
    seed: int = 42,
    delta: float = 1.5,
    mu_y: float = 50.0,
    sigma_y: float = 10.0,
    mu_x: float = 45.0,
    sigma_x: float = 8.0,
    rho: float = 0.6,
    missing_covariate_rate: float = 0.0,
    experiment_id: str = "exp_effect_normal",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a Gaussian A/B test with a known additive treatment effect delta > 0.
    """
    rng = np.random.default_rng(seed)
    
    x = rng.normal(loc=mu_x, scale=sigma_x, size=n_users)
    z = rng.normal(loc=0.0, scale=1.0, size=n_users)
    y0 = mu_y + rho * (sigma_y / sigma_x) * (x - mu_x) + np.sqrt(max(0.0, 1.0 - rho**2)) * sigma_y * z
    
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    # Additive treatment effect on treatment arm
    treatment_mask = (variants == "treatment")
    y = y0.copy()
    y[treatment_mask] += delta
    
    pre_covariate = x.copy()
    if missing_covariate_rate > 0:
        mask = rng.random(size=n_users) < missing_covariate_rate
        pre_covariate = np.where(mask, np.nan, pre_covariate)
        
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "metric_value": y,
        "pre_metric": pre_covariate,
    })
    
    metadata = {
        "scenario": "effect_normal",
        "n_users": n_users,
        "seed": seed,
        "true_ate": delta,
        "true_relative_lift": delta / mu_y,
        "target_correlation": rho,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "known_difference_variance": 2.0 * (sigma_y**2),
    }
    return df, metadata


def generate_null_heavy_tailed(
    n_users: int = 10000,
    seed: int = 42,
    df_student: float = 3.0,
    scale: float = 10.0,
    experiment_id: str = "exp_heavy_tailed",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates heavy-tailed outcomes using Student's t distribution with df=3.
    Demonstrates violation of normality assumptions for sequential methods.
    """
    rng = np.random.default_rng(seed)
    # Student's t random variable
    t_vals = rng.standard_t(df=df_student, size=n_users) * scale + 50.0
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "metric_value": t_vals,
    })
    
    metadata = {
        "scenario": "null_heavy_tailed",
        "n_users": n_users,
        "seed": seed,
        "true_ate": 0.0,
        "df_student": df_student,
        "normality_assumption_valid": False,
    }
    return df, metadata


def generate_conversion(
    n_users: int = 10000,
    seed: int = 42,
    p_control: float = 0.10,
    delta: float = 0.015,
    experiment_id: str = "exp_conversion",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates binary user-level conversions (0 or 1).
    """
    rng = np.random.default_rng(seed)
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    p_treatment = p_control + delta
    is_treatment = (variants == "treatment")
    
    probs = np.where(is_treatment, p_treatment, p_control)
    conversions = (rng.random(size=n_users) < probs).astype(int)
    
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "conversion": conversions,
    })
    
    metadata = {
        "scenario": "conversion",
        "n_users": n_users,
        "seed": seed,
        "true_p_control": p_control,
        "true_p_treatment": p_treatment,
        "true_ate_pp": delta,  # percentage points
        "true_relative_lift": delta / p_control,
    }
    return df, metadata


def generate_ratio_metric(
    n_users: int = 10000,
    seed: int = 42,
    lambda_denom: float = 10.0,
    click_prob_control: float = 0.05,
    delta_click_prob: float = 0.008,
    experiment_id: str = "exp_ratio",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates correlated numerator (clicks/orders) and denominator (sessions/impressions).
    Randomization occurs at the user level, and each user has >= 1 visit.
    """
    rng = np.random.default_rng(seed)
    variants = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    
    # Denominator: user impressions / visits (Poisson + 1)
    denominator = rng.poisson(lam=lambda_denom, size=n_users) + 1
    
    # Numerator: user clicks conditional on visits
    prob_t = click_prob_control + delta_click_prob
    probs = np.where(variants == "treatment", prob_t, click_prob_control)
    numerator = rng.binomial(n=denominator, p=probs)
    
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "numerator": numerator,
        "denominator": denominator,
    })
    
    metadata = {
        "scenario": "ratio_metric",
        "n_users": n_users,
        "seed": seed,
        "true_ratio_control": click_prob_control,
        "true_ratio_treatment": prob_t,
        "true_ratio_diff": delta_click_prob,
        "true_relative_lift": delta_click_prob / click_prob_control,
    }
    return df, metadata


def generate_srm_injected(
    n_users: int = 10000,
    seed: int = 42,
    planned_control_share: float = 0.50,
    realized_treatment_prob: float = 0.53,  # severe mismatch
    experiment_id: str = "exp_srm_corrupted",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates an experiment with deliberate Sample Ratio Mismatch (SRM).
    Planned allocation is 50/50, but true realized traffic is 47/53.
    """
    rng = np.random.default_rng(seed)
    is_treatment = rng.random(size=n_users) < realized_treatment_prob
    variants = np.where(is_treatment, "treatment", "control")
    
    # Metrics
    y = rng.normal(loc=100.0, scale=15.0, size=n_users)
    
    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "user_id": [f"u_{i:06d}" for i in range(n_users)],
        "variant": variants,
        "metric_value": y,
    })
    
    metadata = {
        "scenario": "srm_injected",
        "n_users": n_users,
        "seed": seed,
        "planned_ratio": {"control": planned_control_share, "treatment": 1.0 - planned_control_share},
        "realized_treatment_prob": realized_treatment_prob,
        "srm_corrupted": True,
    }
    return df, metadata
