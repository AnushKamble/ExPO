"""
Property-based and invariant tests for ExPO using Hypothesis.
Verifies fundamental mathematical properties across randomly generated inputs.
"""

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from expo_engine.data.assignment import hash_assignment
from expo_engine.statistics.multiplicity import apply_benjamini_hochberg
from expo_engine.statistics.ratio import compute_arm_ratio_and_variance
from expo_engine.statistics.sequential import (
    compute_confidence_sequence_radius,
    compute_log_mixture_evidence,
)
from expo_engine.statistics.welch import compute_welch_ttest


@given(
    user_id=st.text(min_size=1, max_size=20),
    experiment_id=st.text(min_size=1, max_size=20),
)
def test_hash_assignment_property(user_id: str, experiment_id: str):
    h1 = hash_assignment(user_id, experiment_id)
    h2 = hash_assignment(user_id, experiment_id)
    assert 0.0 <= h1 < 1.0
    assert h1 == h2  # Pure determinism


@given(
    mean_diff=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    scale=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
)
@settings(max_examples=50)
def test_welch_direction_property(mean_diff: float, scale: float):
    # Construct synthetic arms with controlled shift
    c = np.array([10.0, 12.0, 14.0, 11.0, 13.0]) * scale
    t = c + mean_diff
    res = compute_welch_ttest(c, t, metric_name="prop_test")
    if abs(mean_diff) > 1e-4:
        assert np.sign(res.absolute_effect) == np.sign(mean_diff)
        assert np.sign(res.t_statistic) == np.sign(mean_diff)


@given(
    pvals=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=30,
    )
)
@settings(max_examples=50)
def test_bh_invariants_property(pvals):
    hypotheses = [(f"m_{i}", p) for i, p in enumerate(pvals)]
    res = apply_benjamini_hochberg(hypotheses, fdr_q=0.05)

    assert len(res.hypotheses) == len(pvals)
    for hyp in res.hypotheses:
        # 1. Adjusted p-values must be >= raw p-values
        assert hyp.adjusted_p_value >= hyp.raw_p_value - 1e-9
        # 2. Adjusted p-values must remain in [0, 1]
        assert 0.0 <= hyp.adjusted_p_value <= 1.0


@given(
    d_bar=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    n=st.integers(min_value=1, max_value=500),
    v=st.floats(min_value=0.1, max_value=50.0, allow_nan=False),
    tau2=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
)
@settings(max_examples=50)
def test_sequential_radius_shrinks_property(d_bar: float, n: int, v: float, tau2: float):
    log_ev = compute_log_mixture_evidence(d_bar, n, v, tau2)
    assert np.isfinite(log_ev)
    # Radius must decrease strictly as n increases
    r1 = compute_confidence_sequence_radius(n, v, tau2)
    r2 = compute_confidence_sequence_radius(n * 4, v, tau2)
    assert r1 > r2
