"""
Statistical uncertainty quantification for Monte Carlo simulations.
Computes Wilson Score confidence intervals and Monte Carlo Standard Errors (MCSE).
"""

from typing import Tuple
import numpy as np
from scipy import stats


def compute_mcse(p_hat: float, n_replications: int) -> float:
    """
    Computes the Monte Carlo Standard Error (MCSE) for an empirical rate.
    MCSE = sqrt(p_hat * (1 - p_hat) / n)
    """
    if n_replications <= 0:
        return 0.0
    p = np.clip(p_hat, 0.0, 1.0)
    return float(np.sqrt(p * (1.0 - p) / n_replications))


def compute_wilson_score_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Computes the Wilson Score interval for a binomial proportion.
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    if trials <= 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / trials
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))

    denom = 1.0 + (z ** 2) / trials
    center = (p_hat + (z ** 2) / (2.0 * trials)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / trials) + ((z ** 2) / (4.0 * (trials ** 2))))

    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return float(p_hat), lower, upper
