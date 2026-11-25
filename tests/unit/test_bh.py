import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from expo_engine.statistics.multiplicity import apply_benjamini_hochberg


def test_bh_comparison_with_statsmodels():
    hypotheses = [
        ("m1", 0.001),
        ("m2", 0.045),
        ("m3", 0.030),
        ("m4", 0.005),
        ("m5", 0.120),
        ("m6", 0.850),
    ]
    raw_pvals = [p for _, p in hypotheses]

    res = apply_benjamini_hochberg(hypotheses, fdr_q=0.05)

    # Reference statsmodels
    sm_rejected, sm_adj_p, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    # Verify order is preserved and adjusted values match
    for i, hyp in enumerate(res.hypotheses):
        assert hyp.metric_name == hypotheses[i][0]
        assert hyp.raw_p_value == hypotheses[i][1]
        assert hyp.adjusted_p_value == pytest.approx(sm_adj_p[i], rel=1e-6)
        assert hyp.rejected == sm_rejected[i]

    assert res.rejections_count == int(np.sum(sm_rejected))


def test_bh_monotonicity_and_ties():
    # Tied p-values
    hypotheses = [("a", 0.02), ("b", 0.02), ("c", 0.05), ("d", 0.05)]
    res = apply_benjamini_hochberg(hypotheses, fdr_q=0.05)

    adj_pvals = [h.adjusted_p_value for h in res.hypotheses]
    # Tied raw values must yield identical adjusted values
    assert adj_pvals[0] == adj_pvals[1]
    assert adj_pvals[2] == adj_pvals[3]
    # Adjusted values must be >= raw values
    for h in res.hypotheses:
        assert h.adjusted_p_value >= h.raw_p_value


def test_bh_empty_input():
    res = apply_benjamini_hochberg([])
    assert res.total_tests == 0
    assert res.rejections_count == 0
    assert len(res.hypotheses) == 0


def test_bh_invalid_pvalues_rejected():
    with pytest.raises(ValueError, match="Invalid p-value"):
        apply_benjamini_hochberg([("m1", -0.05)])
    with pytest.raises(ValueError, match="Invalid p-value"):
        apply_benjamini_hochberg([("m1", 1.25)])
    with pytest.raises(ValueError, match="Invalid p-value"):
        apply_benjamini_hochberg([("m1", np.nan)])
