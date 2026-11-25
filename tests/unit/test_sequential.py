import numpy as np
import pytest
from expo_engine.statistics.sequential import (
    compute_confidence_sequence_radius,
    compute_log_mixture_evidence,
    evaluate_sequential_monitoring,
)


def test_sequential_known_answer_log_evidence():
    # Analytical known values:
    # n = 50, d_bar = 0.4, v = 4.0, tau2 = 1.0
    # v_post = 4.0 + 50 * 1.0 = 54.0
    # term1 = 0.5 * log(4.0 / 54.0) = 0.5 * log(0.07407407) = -1.301548
    # term2 = (50^2 * 1.0 * 0.4^2) / (2 * 4.0 * 54.0) = (2500 * 0.16) / 432.0 = 400.0 / 432.0 = 0.9259259
    # total = -1.301548 + 0.9259259 = -0.375622
    expected = 0.5 * np.log(4.0 / 54.0) + (400.0 / 432.0)

    val = compute_log_mixture_evidence(
        cumulative_mean_diff=0.4,
        n_pairs=50,
        known_variance=4.0,
        mixture_variance=1.0,
    )
    assert val == pytest.approx(expected, rel=1e-6)


def test_running_maximum_and_anytime_p_monotonicity():
    rng = np.random.default_rng(123)
    # 200 pairs of null differences
    control = rng.normal(10.0, 1.0, 200)
    treatment = rng.normal(10.0, 1.0, 200)
    known_var = 2.0  # 1^2 + 1^2

    res = evaluate_sequential_monitoring(
        treatment_stream=treatment,
        control_stream=control,
        known_variance=known_var,
        mixture_variance=1.0,
        alpha=0.05,
        look_frequency=10,
    )

    assert res.eligible is True
    assert len(res.looks_history) == 20

    # Invariants:
    # 1. Running max log evidence is non-decreasing
    # 2. Anytime p-value is non-increasing
    prev_max = -float("inf")
    prev_p = 1.0
    for look in res.looks_history:
        assert look.running_max_log_evidence >= prev_max
        assert look.anytime_p_value <= prev_p + 1e-9
        prev_max = look.running_max_log_evidence
        prev_p = look.anytime_p_value


def test_sequential_boundary_crossing_on_large_effect():
    # Large effect -> should cross 1/alpha boundary early
    rng = np.random.default_rng(42)
    control = rng.normal(10.0, 1.0, 300)
    treatment = rng.normal(12.0, 1.0, 300)  # delta = +2.0 (massive effect)
    known_var = 2.0

    res = evaluate_sequential_monitoring(
        treatment_stream=treatment,
        control_stream=control,
        known_variance=known_var,
        alpha=0.05,
        look_frequency=10,
    )

    assert res.stopped_early is True
    assert res.stopping_look is not None
    assert res.final_anytime_p_value < 0.05
    assert res.final_running_max_evidence >= (1.0 / 0.05)


def test_confidence_sequence_radius_shrinks():
    r10 = compute_confidence_sequence_radius(n_pairs=10, known_variance=2.0, mixture_variance=1.0)
    r100 = compute_confidence_sequence_radius(n_pairs=100, known_variance=2.0, mixture_variance=1.0)
    r500 = compute_confidence_sequence_radius(n_pairs=500, known_variance=2.0, mixture_variance=1.0)
    assert r10 > r100 > r500


def test_sequential_ineligibility_blocks():
    res = evaluate_sequential_monitoring(
        treatment_stream=[1.0, 2.0],
        control_stream=[1.0, 2.0],
        known_variance=2.0,
        metric_kind="ratio_of_sums",  # Unsupported
    )
    assert res.eligible is False
    assert "violates model assumptions" in res.ineligibility_reason


def test_sequential_parameter_validation():
    with pytest.raises(ValueError, match="look_frequency must be strictly positive"):
        evaluate_sequential_monitoring(
            treatment_stream=[1.0, 2.0],
            control_stream=[1.0, 2.0],
            known_variance=2.0,
            look_frequency=0,
        )
    with pytest.raises(ValueError, match="alpha must be in"):
        evaluate_sequential_monitoring(
            treatment_stream=[1.0, 2.0],
            control_stream=[1.0, 2.0],
            known_variance=2.0,
            alpha=1.5,
        )


def test_sequential_empty_intersection_recovery():
    # Construct an artificial series of look estimates where confidence intervals jump and disjoint
    # evaluate_sequential_monitoring running intersection must recover gracefully and keep cs_lower <= cs_upper
    rng = np.random.default_rng(999)
    # 60 alternating extreme observations designed to stress running intersection
    control = np.array([0.0, 0.0, 10.0, 10.0] * 10)
    treatment = np.array([10.0, 10.0, 0.0, 0.0] * 10)

    res = evaluate_sequential_monitoring(
        treatment_stream=treatment,
        control_stream=control,
        known_variance=50.0,
        look_frequency=4,
    )
    assert res.eligible is True
    for look in res.looks_history:
        assert look.cs_lower <= look.cs_upper
    assert res.final_cs_lower <= res.final_cs_upper

