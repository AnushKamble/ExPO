"""
Mathematically Scoped Gaussian Normal-Mixture Sequential Evidence Engine (mSPRT).
Implements exact log-evidence computations, running-maximum anytime p-values,
and time-uniform confidence sequences via mixture-evidence interval inversion.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from expo_engine.reporting.schemas import SequentialLook, SequentialResult


def compute_log_mixture_evidence(
    cumulative_mean_diff: float,
    n_pairs: int,
    known_variance: float,
    mixture_variance: float,
) -> float:
    """
    Computes exact log mixture evidence (log Bayes Factor) for Gaussian increments:
    log E_n = 0.5 * log(sigma_D^2 / (sigma_D^2 + n * tau^2)) + (n^2 * tau^2 * d_bar^2) / (2 * sigma_D^2 * (sigma_D^2 + n * tau^2))
    """
    if n_pairs <= 0:
        return 0.0
    if known_variance <= 0:
        raise ValueError(f"known_variance must be strictly positive, got {known_variance}")
    if mixture_variance <= 0:
        raise ValueError(f"mixture_variance (tau^2) must be strictly positive, got {mixture_variance}")

    v = float(known_variance)
    tau2 = float(mixture_variance)
    n = float(n_pairs)
    d_bar = float(cumulative_mean_diff)

    v_post = v + n * tau2
    # First term: 0.5 * log(v / (v + n * tau^2))
    term1 = 0.5 * np.log(v / v_post)
    # Second term: (n^2 * tau^2 * d_bar^2) / (2 * v * (v + n * tau^2))
    term2 = ((n ** 2) * tau2 * (d_bar ** 2)) / (2.0 * v * v_post)

    return float(term1 + term2)


def compute_confidence_sequence_radius(
    n_pairs: int,
    known_variance: float,
    mixture_variance: float,
    alpha: float = 0.05,
) -> float:
    """
    Computes per-look radius r_n for time-uniform confidence sequence:
    r_n = sqrt( [2 * sigma_D^2 * (sigma_D^2 + n * tau^2) / (n^2 * tau^2)] * [log(1/alpha) + 0.5 * log(1 + n * tau^2 / sigma_D^2)] )
    """
    if n_pairs <= 0:
        return float("inf")
    v = float(known_variance)
    tau2 = float(mixture_variance)
    n = float(n_pairs)

    v_post = v + n * tau2
    multiplier = (2.0 * v * v_post) / ((n ** 2) * tau2)
    inner = np.log(1.0 / alpha) + 0.5 * np.log(1.0 + (n * tau2 / v))
    radius = float(np.sqrt(multiplier * inner))
    return radius


def evaluate_sequential_monitoring(
    treatment_stream: Union[np.ndarray, List[float]],
    control_stream: Union[np.ndarray, List[float]],
    known_variance: float,
    mixture_variance: float = 1.0,
    alpha: float = 0.05,
    look_frequency: int = 10,
    metric_name: str = "metric",
    metric_kind: str = "continuous_mean",
) -> SequentialResult:
    """
    Evaluates sequential monitoring across balanced looks.
    Enforces strict model preconditions.
    """
    warnings: List[str] = []

    # Precondition check 1: Metric kind eligibility
    if metric_kind not in ("continuous_mean", "count_mean"):
        return SequentialResult(
            metric_name=metric_name,
            eligible=False,
            ineligibility_reason=(
                f"Metric kind '{metric_kind}' violates model assumptions. "
                "Gaussian mixture sequential engine strictly requires independent Gaussian increments."
            ),
            known_difference_variance=known_variance,
            mixture_variance=mixture_variance,
            alpha=alpha,
            stopping_threshold=1.0 / alpha,
            total_looks_analyzed=0,
            total_pairs=0,
            stopped_early=False,
            final_mean_diff=0.0,
            final_evidence=1.0,
            final_running_max_evidence=1.0,
            final_anytime_p_value=1.0,
            final_cs_lower=0.0,
            final_cs_upper=0.0,
            warnings=["Sequential monitoring blocked due to ineligibility."],
        )

    # Precondition check 2: Predeclared variance
    if known_variance <= 0:
        raise ValueError(
            f"known_variance must be strictly positive from pre-experiment source, got {known_variance}."
        )

    y_t = np.asarray(treatment_stream, dtype=float)
    y_c = np.asarray(control_stream, dtype=float)

    # Balanced pair formation (balanced accrual)
    min_n = min(len(y_t), len(y_c))
    if len(y_t) != len(y_c):
        warnings.append(
            f"Accrual streams unbalanced (Treatment={len(y_t)}, Control={len(y_c)}). "
            f"Sequential evaluation truncated to {min_n} balanced observation pairs."
        )

    if min_n == 0:
        raise ValueError("Cannot evaluate sequential test on empty stream.")

    # Independent differences sequence: D_i = Y_T,i - Y_C,i
    diffs = y_t[:min_n] - y_c[:min_n]
    cum_sums = np.cumsum(diffs)

    threshold = 1.0 / alpha
    log_threshold = np.log(threshold)

    looks_history: List[SequentialLook] = []
    running_max_log_ev = -float("inf")
    running_max_ev = 0.0
    stopped_early = False
    stopping_look: Optional[int] = None

    running_cs_lower = -float("inf")
    running_cs_upper = float("inf")

    # Determine look indices (at least 1, then every look_frequency, and the final look)
    look_indices = list(range(look_frequency, min_n + 1, look_frequency))
    if min_n not in look_indices:
        look_indices.append(min_n)
    if not look_indices:
        look_indices = [min_n]

    for look_idx, k_pairs in enumerate(look_indices, start=1):
        d_bar = float(cum_sums[k_pairs - 1] / k_pairs)

        log_ev = compute_log_mixture_evidence(
            cumulative_mean_diff=d_bar,
            n_pairs=k_pairs,
            known_variance=known_variance,
            mixture_variance=mixture_variance,
        )
        ev = float(np.exp(np.clip(log_ev, -100.0, 100.0)))

        # Monotonically non-decreasing running maximum
        if log_ev > running_max_log_ev:
            running_max_log_ev = log_ev
            running_max_ev = ev

        # Monotonically non-increasing anytime p-value: p_AV = min(1, 1 / M_n)
        if running_max_log_ev <= 0.0:
            anytime_p = 1.0
        else:
            anytime_p = float(min(1.0, np.exp(-running_max_log_ev)))

        crossed = bool(running_max_log_ev >= log_threshold)
        if crossed and not stopped_early:
            stopped_early = True
            stopping_look = look_idx

        # Time-uniform confidence sequence per-look inversion
        r_k = compute_confidence_sequence_radius(
            n_pairs=k_pairs,
            known_variance=known_variance,
            mixture_variance=mixture_variance,
            alpha=alpha,
        )
        look_lower = d_bar - r_k
        look_upper = d_bar + r_k

        # Running intersection: L_n = max(L_{k}), U_n = min(U_{k})
        running_cs_lower = max(running_cs_lower, look_lower)
        running_cs_upper = min(running_cs_upper, look_upper)

        # Handle empty intersection edge case if it ever arises
        final_look_l = running_cs_lower
        final_look_u = running_cs_upper
        if final_look_l > final_look_u:
            final_look_l = look_lower
            final_look_u = look_upper
            warnings.append(
                f"Confidence sequence intersection became empty at look {look_idx}; "
                "reverted to current per-look inversion bound."
            )

        look_obj = SequentialLook(
            look_index=look_idx,
            sample_size_pairs=k_pairs,
            cumulative_mean_diff=d_bar,
            log_evidence=log_ev,
            evidence=ev,
            running_max_log_evidence=running_max_log_ev,
            running_max_evidence=running_max_ev,
            anytime_p_value=anytime_p,
            boundary_crossed=crossed,
            cs_lower=final_look_l,
            cs_upper=final_look_u,
        )
        looks_history.append(look_obj)

    last_look = looks_history[-1]

    return SequentialResult(
        metric_name=metric_name,
        estimand="Sequential Mean Difference under Known Variance Gaussian Increments",
        method="Gaussian Normal-Mixture Evidence Process (mSPRT)",
        eligible=True,
        ineligibility_reason=None,
        known_difference_variance=known_variance,
        mixture_variance=mixture_variance,
        alpha=alpha,
        stopping_threshold=threshold,
        total_looks_analyzed=len(looks_history),
        total_pairs=min_n,
        stopped_early=stopped_early,
        stopping_look=stopping_look,
        final_mean_diff=last_look.cumulative_mean_diff,
        final_evidence=last_look.evidence,
        final_running_max_evidence=last_look.running_max_evidence,
        final_anytime_p_value=last_look.anytime_p_value,
        final_cs_lower=last_look.cs_lower,
        final_cs_upper=last_look.cs_upper,
        looks_history=looks_history,
        warnings=warnings,
    )
