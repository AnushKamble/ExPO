import numpy as np
import pandas as pd
import pytest
from expo_engine.checks.validation import validate_experiment_data
from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole, MetricSpec


@pytest.fixture
def base_config():
    return ExperimentConfig(
        experiment_id="test_exp",
        user_id_col="user_id",
        variant_col="variant",
        control_variant="control",
        treatment_variant="treatment",
        metrics=[
            MetricSpec(
                name="revenue",
                label="Total Revenue",
                kind=MetricKind.CONTINUOUS_MEAN,
                role=MetricRole.PRIMARY,
                column="revenue",
                pre_covariate="pre_revenue",
            )
        ],
    )


def test_missing_required_columns(base_config):
    df = pd.DataFrame({"user_id": ["u1", "u2"], "revenue": [10.0, 20.0]})  # missing variant
    res = validate_experiment_data(df, base_config)
    assert not res.is_valid
    assert any("variant" in err for err in res.errors)


def test_duplicate_users_rejected(base_config):
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2"],
        "variant": ["control", "treatment", "control"],
        "revenue": [10.0, 15.0, 20.0],
    })
    res = validate_experiment_data(df, base_config)
    assert not res.is_valid
    assert any("Duplicate user IDs detected" in err for err in res.errors)
    assert any("Variant contamination" in err for err in res.errors)


def test_non_finite_metric_rejected(base_config):
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "variant": ["control", "treatment"],
        "revenue": [10.0, np.nan],
    })
    res = validate_experiment_data(df, base_config)
    assert not res.is_valid
    assert any("Non-finite values" in err for err in res.errors)


def test_binary_metric_validation():
    config = ExperimentConfig(
        experiment_id="test_bin",
        metrics=[
            MetricSpec(name="conversion", label="Conversion", kind=MetricKind.BINARY_MEAN, column="conv")
        ],
    )
    df_invalid = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "variant": ["control", "treatment"],
        "conv": [0, 2],  # 2 is not binary
    })
    res = validate_experiment_data(df_invalid, config)
    assert not res.is_valid
    assert any("non-binary values" in err for err in res.errors)


def test_ratio_metric_validation():
    config = ExperimentConfig(
        experiment_id="test_ratio",
        metrics=[
            MetricSpec(
                name="ctr",
                label="Click-Through Rate",
                kind=MetricKind.RATIO_OF_SUMS,
                numerator_column="clicks",
                denominator_column="impressions",
            )
        ],
    )
    # Negative denominator
    df_neg = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "variant": ["control", "treatment"],
        "clicks": [1, 2],
        "impressions": [-5, 10],
    })
    res = validate_experiment_data(df_neg, config)
    assert not res.is_valid
    assert any("Negative values detected in denominator" in err for err in res.errors)


def test_valid_dataset_passes(base_config):
    df_valid = pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(100)],
        "variant": ["control" if i % 2 == 0 else "treatment" for i in range(100)],
        "revenue": [50.0 + i for i in range(100)],
        "pre_revenue": [45.0 + i for i in range(100)],
    })
    res = validate_experiment_data(df_valid, base_config)
    assert res.is_valid
    assert len(res.errors) == 0
    assert res.total_rows == 100
    assert res.variant_counts["control"] == 50
    assert res.variant_counts["treatment"] == 50
