"""
Multiple testing and False Discovery Rate (FDR) control for predeclared metric families.
Implements the Benjamini-Hochberg (BH) step-up procedure while preserving original metric order.
"""

from typing import List, Tuple
import numpy as np

from expo_engine.reporting.schemas import AdjustedHypothesis, MultiplicityResult


def apply_benjamini_hochberg(
    hypotheses: List[Tuple[str, float]],
    fdr_q: float = 0.05,
    family_name: str = "secondary_metrics",
) -> MultiplicityResult:
    """
    Applies the Benjamini-Hochberg (BH) procedure to control False Discovery Rate.
    Preserves original hypothesis ordering and handles ties cleanly.
    """
    if not (0.0 < fdr_q < 1.0):
        raise ValueError(f"fdr_q must be in (0, 1), got {fdr_q}")

    m = len(hypotheses)
    if m == 0:
        return MultiplicityResult(
            method="Benjamini-Hochberg (BH) Step-Up Procedure",
            family_name=family_name,
            fdr_q=fdr_q,
            hypotheses=[],
            total_tests=0,
            rejections_count=0,
        )

    # Validate p-values
    for name, p in hypotheses:
        if not np.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError(f"Invalid p-value for hypothesis '{name}': {p}. Must be in [0, 1].")

    # Keep track of original indices: (orig_idx, name, p_val)
    indexed_pvals = [(i, name, float(p)) for i, (name, p) in enumerate(hypotheses)]
    # Sort ascending by raw p-value
    sorted_pvals = sorted(indexed_pvals, key=lambda x: x[2])

    # Compute BH adjusted p-values:
    # p_adj(k) = min_{j >= k} (m / j * p(j))
    raw_p_sorted = np.array([item[2] for item in sorted_pvals])
    ranks = np.arange(1, m + 1)
    
    # Unmonotonized values: (m / rank) * p
    unmonotonized = raw_p_sorted * (m / ranks)
    
    # Monotonize in reverse order: p_adj(k) = min(p_adj(k+1), unmonotonized(k))
    adj_p_sorted = np.zeros(m)
    current_min = 1.0
    for k in range(m - 1, -1, -1):
        current_val = min(1.0, unmonotonized[k])
        current_min = min(current_min, current_val)
        adj_p_sorted[k] = current_min

    # Map back to original order
    results_dict = {}
    rejections_count = 0
    for k, (orig_idx, name, raw_p) in enumerate(sorted_pvals):
        adj_p = float(adj_p_sorted[k])
        rejected = bool(adj_p <= fdr_q)
        if rejected:
            rejections_count += 1
        results_dict[orig_idx] = AdjustedHypothesis(
            metric_name=name,
            raw_p_value=raw_p,
            adjusted_p_value=adj_p,
            rejected=rejected,
        )

    ordered_hypotheses = [results_dict[i] for i in range(m)]

    return MultiplicityResult(
        method="Benjamini-Hochberg (BH) Step-Up Procedure",
        family_name=family_name,
        fdr_q=fdr_q,
        hypotheses=ordered_hypotheses,
        total_tests=m,
        rejections_count=rejections_count,
    )
