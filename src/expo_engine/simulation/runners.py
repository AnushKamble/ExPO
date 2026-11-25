"""
Reproducible Monte Carlo Research Suite for ExPO.
Implements five core simulation studies:
- Study A: Peeking Inflation vs Sequential Evidence
- Study B: CUPED Variance Reduction & CI Shrinkage
- Study C: A/A Calibration & Type I Error Uniformity
- Study D: Ratio Delta Method vs Bootstrap Sensitivity
- Study E: Sample Ratio Mismatch Detection Power
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from expo_engine.simulation.uncertainty import compute_mcse, compute_wilson_score_interval
from expo_engine.statistics.sequential import compute_log_mixture_evidence


def run_peeking_simulation(
    preset: str = "smoke",
    seed: int = 42,
    delta_true: float = 0.0,
    alpha: float = 0.05,
    n_pairs_per_look: int = 50,
    sigma_d: float = 2.0,  # std dev of difference
    mixture_variance: float = 1.0,
) -> Dict[str, Any]:
    """
    Study A: Compares Fixed-Horizon Welch vs Naive Daily Peeking vs Gaussian Mixture Sequential.
    """
    if preset == "smoke":
        n_reps = 200
        n_looks = 15
    elif preset == "medium":
        n_reps = 1000
        n_looks = 30
    elif preset == "full":
        n_reps = 5000
        n_looks = 30
    else:
        raise ValueError(f"Unknown preset: '{preset}'. Choose 'smoke', 'medium', or 'full'.")

    rng = np.random.default_rng(seed)

    # Generate incremental look sums: shape (n_reps, n_looks)
    # Each look has n_pairs_per_look increments ~ N(delta_true, sigma_d^2)
    # Sum of n_pairs increments ~ N(n_pairs * delta_true, n_pairs * sigma_d^2)
    look_sums = rng.normal(
        loc=n_pairs_per_look * delta_true,
        scale=np.sqrt(n_pairs_per_look) * sigma_d,
        size=(n_reps, n_looks),
    )
    cum_sums = np.cumsum(look_sums, axis=1)

    # Look indices and sample sizes
    pairs_at_look = np.arange(1, n_looks + 1) * n_pairs_per_look  # shape (n_looks,)
    cum_means = cum_sums / pairs_at_look[np.newaxis, :]  # shape (n_reps, n_looks)

    # 1. Fixed-horizon Welch test at final look K
    final_means = cum_means[:, -1]
    final_n = pairs_at_look[-1]
    final_se = sigma_d / np.sqrt(final_n)
    final_z = np.abs(final_means) / final_se
    final_p = 2.0 * (1.0 - stats.norm.cdf(final_z))
    fixed_rejections = int(np.sum(final_p < alpha))

    # 2. Naive daily peeking: reject if any look has nominal p < alpha
    all_se = sigma_d / np.sqrt(pairs_at_look[np.newaxis, :])
    all_z = np.abs(cum_means) / all_se
    all_nominal_p = 2.0 * (1.0 - stats.norm.cdf(all_z))
    min_nominal_p = np.min(all_nominal_p, axis=1)
    naive_rejections = int(np.sum(min_nominal_p < alpha))

    # 3. Gaussian mixture sequential e-process
    # Compute log evidence across all replicates and looks
    v = sigma_d ** 2
    tau2 = mixture_variance
    threshold_log_ev = np.log(1.0 / alpha)

    # Vectorized log evidence calculation
    n_grid = pairs_at_look[np.newaxis, :]  # shape (1, n_looks)
    v_post = v + n_grid * tau2
    term1 = 0.5 * np.log(v / v_post)
    term2 = ((n_grid ** 2) * tau2 * (cum_means ** 2)) / (2.0 * v * v_post)
    log_evidence_grid = term1 + term2  # shape (n_reps, n_looks)

    max_log_ev = np.max(log_evidence_grid, axis=1)
    sequential_rejections = int(np.sum(max_log_ev >= threshold_log_ev))

    # Compute metrics, stopping sample sizes, and first crossing distributions
    # 1. Naive Peeking stopping sample size
    has_naive_cross = np.any(all_nominal_p < alpha, axis=1)
    # First crossing index where p < alpha, else final look
    naive_cross_idx = np.where(has_naive_cross, np.argmax(all_nominal_p < alpha, axis=1), n_looks - 1)
    naive_stopping_pairs = pairs_at_look[naive_cross_idx]
    naive_mean_stopping = float(np.mean(naive_stopping_pairs))
    naive_median_stopping = float(np.median(naive_stopping_pairs))
    naive_crossing_dist = [int(np.sum((has_naive_cross) & (naive_cross_idx == l))) for l in range(n_looks)]

    # 2. Sequential mSPRT stopping sample size
    seq_cross_mask = (log_evidence_grid >= threshold_log_ev)
    has_seq_cross = np.any(seq_cross_mask, axis=1)
    seq_cross_idx = np.where(has_seq_cross, np.argmax(seq_cross_mask, axis=1), n_looks - 1)
    seq_stopping_pairs = pairs_at_look[seq_cross_idx]
    seq_mean_stopping = float(np.mean(seq_stopping_pairs))
    seq_median_stopping = float(np.median(seq_stopping_pairs))
    seq_crossing_dist = [int(np.sum((has_seq_cross) & (seq_cross_idx == l))) for l in range(n_looks)]

    # Compute Wilson intervals
    fixed_rate, fixed_l, fixed_u = compute_wilson_score_interval(fixed_rejections, n_reps)
    naive_rate, naive_l, naive_u = compute_wilson_score_interval(naive_rejections, n_reps)
    seq_rate, seq_l, seq_u = compute_wilson_score_interval(sequential_rejections, n_reps)

    return {
        "study": "peeking_inflation",
        "preset": preset,
        "seed": seed,
        "n_replications": n_reps,
        "n_looks": n_looks,
        "total_pairs_max": int(pairs_at_look[-1]),
        "delta_true": delta_true,
        "alpha": alpha,
        "fixed_horizon": {
            "rejections": fixed_rejections,
            "rejection_rate": fixed_rate,
            "wilson_ci_lower": fixed_l,
            "wilson_ci_upper": fixed_u,
            "mcse": compute_mcse(fixed_rate, n_reps),
            "stopping_sample_size_pairs": int(pairs_at_look[-1]),
        },
        "naive_daily_peeking": {
            "rejections": naive_rejections,
            "rejection_rate": naive_rate,
            "wilson_ci_lower": naive_l,
            "wilson_ci_upper": naive_u,
            "mcse": compute_mcse(naive_rate, n_reps),
            "mean_stopping_sample_size_pairs": naive_mean_stopping,
            "median_stopping_sample_size_pairs": naive_median_stopping,
            "first_crossing_distribution_by_look": naive_crossing_dist,
        },
        "gaussian_mixture_sequential": {
            "rejections": sequential_rejections,
            "rejection_rate": seq_rate,
            "wilson_ci_lower": seq_l,
            "wilson_ci_upper": seq_u,
            "mcse": compute_mcse(seq_rate, n_reps),
            "mean_stopping_sample_size_pairs": seq_mean_stopping,
            "median_stopping_sample_size_pairs": seq_median_stopping,
            "first_crossing_distribution_by_look": seq_crossing_dist,
        },
    }


def run_cuped_simulation(
    preset: str = "smoke",
    seed: int = 42,
    rhos: Sequence[float] = (0.0, 0.2, 0.5, 0.7, 0.9),
    n_users_per_rep: int = 1000,
    delta_true: float = 1.0,
) -> Dict[str, Any]:
    """
    Study B: Evaluates CUPED variance reduction, bias, coverage, and CI shrinkage across a correlation grid.
    """
    if preset == "smoke":
        n_reps = 100
    elif preset == "medium":
        n_reps = 500
    elif preset == "full":
        n_reps = 2000
    else:
        n_reps = 1000
    rng = np.random.default_rng(seed)
    results_by_rho = []

    for rho in rhos:
        var_raw_list = []
        var_adj_list = []
        ci_width_raw_list = []
        ci_width_adj_list = []
        covered_count = 0
        ate_errors = []

        for _ in range(n_reps):
            # Fast vectorized user simulation
            x = rng.normal(50.0, 10.0, n_users_per_rep)
            z = rng.normal(0.0, 1.0, n_users_per_rep)
            sigma_y = 10.0
            y0 = 50.0 + rho * (x - 50.0) + np.sqrt(max(0.0, 1.0 - rho**2)) * sigma_y * z

            is_treatment = rng.random(n_users_per_rep) < 0.5
            y = y0 + is_treatment * delta_true

            # Variance raw
            var_raw = float(np.var(y, ddof=1))

            # CUPED theta
            x_var = float(np.var(x, ddof=1))
            theta = float(np.cov(y, x, ddof=1)[0, 1]) / x_var if x_var > 0 else 0.0
            y_adj = y - theta * (x - np.mean(x))
            var_adj = float(np.var(y_adj, ddof=1))

            # Raw Welch CI
            y_c = y[~is_treatment]
            y_t = y[is_treatment]
            se_raw = np.sqrt(np.var(y_c, ddof=1) / len(y_c) + np.var(y_t, ddof=1) / len(y_t))
            width_raw = 2.0 * 1.95996 * se_raw

            # Adj Welch CI
            adj_c = y_adj[~is_treatment]
            adj_t = y_adj[is_treatment]
            ate_adj = np.mean(adj_t) - np.mean(adj_c)
            se_adj = np.sqrt(np.var(adj_c, ddof=1) / len(adj_c) + np.var(adj_t, ddof=1) / len(adj_t))
            width_adj = 2.0 * 1.95996 * se_adj

            if (ate_adj - 1.95996 * se_adj) <= delta_true <= (ate_adj + 1.95996 * se_adj):
                covered_count += 1

            var_raw_list.append(var_raw)
            var_adj_list.append(var_adj)
            ci_width_raw_list.append(width_raw)
            ci_width_adj_list.append(width_adj)
            ate_errors.append(ate_adj - delta_true)

        mean_var_raw = float(np.mean(var_raw_list))
        mean_var_adj = float(np.mean(var_adj_list))
        emp_var_reduction = 1.0 - (mean_var_adj / mean_var_raw)
        theo_var_reduction = float(rho ** 2)

        mean_w_raw = float(np.mean(ci_width_raw_list))
        mean_w_adj = float(np.mean(ci_width_adj_list))
        ci_shrinkage = 1.0 - (mean_w_adj / mean_w_raw)

        cov_rate, cov_l, cov_u = compute_wilson_score_interval(covered_count, n_reps)

        results_by_rho.append({
            "target_rho": rho,
            "theoretical_variance_reduction": theo_var_reduction,
            "empirical_variance_reduction": emp_var_reduction,
            "ci_width_reduction": ci_shrinkage,
            "coverage_rate": cov_rate,
            "coverage_wilson_ci": [cov_l, cov_u],
            "mean_ate_bias": float(np.mean(ate_errors)),
        })

    return {
        "study": "cuped_variance_reduction",
        "preset": preset,
        "seed": seed,
        "n_replications_per_rho": n_reps,
        "n_users_per_rep": n_users_per_rep,
        "delta_true": delta_true,
        "grid_results": results_by_rho,
    }


def run_aa_simulation(
    preset: str = "smoke",
    seed: int = 42,
    n_users_per_rep: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Study C: A/A Null Calibration and Uniformity Assessment.
    """
    if preset == "smoke":
        n_reps = 200
    elif preset == "medium":
        n_reps = 1000
    elif preset == "full":
        n_reps = 5000
    else:
        n_reps = 2000
    rng = np.random.default_rng(seed)

    # Generate matrix of A/A user outcomes under exact null
    # Fast vectorized t-tests
    y_c = rng.normal(50.0, 10.0, size=(n_reps, n_users_per_rep // 2))
    y_t = rng.normal(50.0, 10.0, size=(n_reps, n_users_per_rep // 2))

    t_stats, p_values = stats.ttest_ind(y_t, y_c, axis=1, equal_var=False)

    rejections = int(np.sum(p_values < alpha))
    fpr, fpr_l, fpr_u = compute_wilson_score_interval(rejections, n_reps)

    # Kolmogorov-Smirnov test of uniformity on [0, 1]
    ks_stat, ks_p = stats.kstest(p_values, "uniform")

    # Histogram binning (10 decile bins)
    hist_counts, _ = np.histogram(p_values, bins=10, range=(0.0, 1.0))
    expected_bin_count = n_reps / 10.0

    return {
        "study": "aa_calibration",
        "preset": preset,
        "seed": seed,
        "n_replications": n_reps,
        "n_users_per_rep": n_users_per_rep,
        "nominal_alpha": alpha,
        "observed_type_1_error": fpr,
        "wilson_ci_lower": fpr_l,
        "wilson_ci_upper": fpr_u,
        "mcse": compute_mcse(fpr, n_reps),
        "ks_uniformity_statistic": float(ks_stat),
        "ks_uniformity_p_value": float(ks_p),
        "p_value_decile_counts": [int(c) for c in hist_counts],
        "expected_count_per_decile": expected_bin_count,
    }


def run_ratio_simulation(
    preset: str = "smoke",
    seed: int = 42,
    n_users: int = 1500,
) -> Dict[str, Any]:
    """
    Study D: Ratio Delta Method Coverage Evaluation.
    """
    if preset == "smoke":
        n_reps = 100
    elif preset == "medium":
        n_reps = 500
    elif preset == "full":
        n_reps = 2000
    else:
        n_reps = 1000
    rng = np.random.default_rng(seed)

    true_ratio_diff = 0.02
    p_c = 0.05
    p_t = p_c + true_ratio_diff

    delta_covered = 0
    se_errors = []

    for _ in range(n_reps):
        denom_c = rng.poisson(lam=10.0, size=n_users // 2) + 1
        denom_t = rng.poisson(lam=10.0, size=n_users // 2) + 1

        num_c = rng.binomial(n=denom_c, p=p_c)
        num_t = rng.binomial(n=denom_t, p=p_t)

        r_c = np.sum(num_c) / np.sum(denom_c)
        r_t = np.sum(num_t) / np.sum(denom_t)
        diff_r = r_t - r_c

        # Influence variance for C and T
        z_c = (num_c - r_c * denom_c) / np.mean(denom_c)
        z_t = (num_t - r_t * denom_t) / np.mean(denom_t)
        var_diff = np.var(z_c, ddof=1) / len(z_c) + np.var(z_t, ddof=1) / len(z_t)
        se = np.sqrt(var_diff)

        ci_l = diff_r - 1.95996 * se
        ci_u = diff_r + 1.95996 * se

        if ci_l <= true_ratio_diff <= ci_u:
            delta_covered += 1

    cov_rate, cov_l, cov_u = compute_wilson_score_interval(delta_covered, n_reps)

    return {
        "study": "ratio_delta_method",
        "preset": preset,
        "seed": seed,
        "n_replications": n_reps,
        "n_users": n_users,
        "true_ratio_diff": true_ratio_diff,
        "empirical_coverage": cov_rate,
        "wilson_ci_lower": cov_l,
        "wilson_ci_upper": cov_u,
        "mcse": compute_mcse(cov_rate, n_reps),
    }


def run_srm_simulation(
    preset: str = "smoke",
    seed: int = 42,
    sample_sizes: Sequence[int] = (1000, 5000, 20000),
    treatment_probs: Sequence[float] = (0.50, 0.51, 0.52, 0.55),
    alpha_srm: float = 0.001,
) -> Dict[str, Any]:
    """
    Study E: SRM Detection Power vs Sample Size and Traffic Imbalance.
    """
    if preset == "smoke":
        n_reps = 100
    elif preset == "medium":
        n_reps = 500
    elif preset == "full":
        n_reps = 2000
    else:
        n_reps = 1000
    rng = np.random.default_rng(seed)
    matrix_results = []

    for n in sample_sizes:
        for p in treatment_probs:
            # Binomial draws: k successes out of n trials with prob p
            k_obs = rng.binomial(n=n, p=p, size=n_reps)
            exp_k = n * 0.50
            o_c = n - k_obs

            # Chi-square stat: (k - E)^2 / E + (o_c - E)^2 / E
            chi2_stats = ((k_obs - exp_k) ** 2) / exp_k + ((o_c - exp_k) ** 2) / exp_k
            p_vals = 1.0 - stats.chi2.cdf(chi2_stats, df=1)

            detections = int(np.sum(p_vals < alpha_srm))
            det_rate, det_l, det_u = compute_wilson_score_interval(detections, n_reps)

            matrix_results.append({
                "sample_size": n,
                "realized_treatment_prob": p,
                "detection_power": det_rate,
                "wilson_ci_lower": det_l,
                "wilson_ci_upper": det_u,
                "mcse": compute_mcse(det_rate, n_reps),
            })

    return {
        "study": "srm_detection_power",
        "preset": preset,
        "seed": seed,
        "n_replications": n_reps,
        "alpha_srm": alpha_srm,
        "matrix_results": matrix_results,
    }
