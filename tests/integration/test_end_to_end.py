from pathlib import Path
import pytest
from expo_engine.config.models import (
    ExperimentConfig,
    MetricKind,
    MetricRole,
    MetricSpec,
    SequentialConfig,
)
from expo_engine.data.synthetic import generate_effect_normal
from expo_engine.reporting.exporter import (
    export_report_csv,
    export_report_html,
    export_report_json,
    export_report_markdown,
)
from expo_engine.services.analysis_service import ExperimentAnalysisService


def test_full_pipeline_and_report_generation(tmp_path: Path):
    # 1. Generate synthetic dataset with known effect and pre-covariate
    df, meta = generate_effect_normal(
        n_users=2000,
        seed=42,
        delta=2.0,
        mu_y=50.0,
        sigma_y=10.0,
        rho=0.7,
        missing_covariate_rate=0.05,
    )

    # 2. Add a secondary metric for multiplicity check
    df["latency"] = df["metric_value"] * 1.5 + 10.0

    # 3. Configure experiment
    config = ExperimentConfig(
        experiment_id="e2e_exp_001",
        user_id_col="user_id",
        variant_col="variant",
        control_variant="control",
        treatment_variant="treatment",
        metrics=[
            MetricSpec(
                name="revenue",
                label="Average Revenue Per User",
                kind=MetricKind.CONTINUOUS_MEAN,
                role=MetricRole.PRIMARY,
                column="metric_value",
                pre_covariate="pre_metric",
            ),
            MetricSpec(
                name="latency",
                label="Page Latency",
                kind=MetricKind.CONTINUOUS_MEAN,
                role=MetricRole.SECONDARY,
                column="latency",
            ),
        ],
        sequential=SequentialConfig(
            enabled=True,
            known_variance=meta["known_difference_variance"],
            mixture_variance=1.0,
            alpha=0.05,
        ),
    )

    # 4. Run service
    service = ExperimentAnalysisService(config)
    manifest = service.run_full_analysis(df, run_id="run_e2e_test")

    assert manifest.experiment_id == "e2e_exp_001"
    assert manifest.primary_metric_result is not None
    assert manifest.cuped_result is not None
    assert manifest.secondary_multiplicity is not None
    assert manifest.sequential_result is not None

    # Verify CUPED variance reduction
    assert manifest.cuped_result["sample_correlation"] > 0.60
    assert manifest.cuped_result["empirical_variance_reduction"] > 0.35

    # 5. Export reports
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    html_path = tmp_path / "report.html"

    export_report_json(manifest, json_path)
    export_report_csv(manifest, csv_path)
    export_report_markdown(manifest, md_path)
    export_report_html(manifest, html_path)

    assert json_path.exists() and json_path.stat().st_size > 0
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0


def test_sequential_ineligibility_reported_for_ratio_primary_metric():
    df, _ = generate_effect_normal(n_users=500, seed=42)
    df["clicks"] = 5
    df["impressions"] = 100

    config = ExperimentConfig(
        experiment_id="e2e_ratio_seq_exp",
        user_id_col="user_id",
        variant_col="variant",
        control_variant="control",
        treatment_variant="treatment",
        metrics=[
            MetricSpec(
                name="ctr",
                label="Click Through Rate",
                kind=MetricKind.RATIO_OF_SUMS,
                role=MetricRole.PRIMARY,
                numerator_column="clicks",
                denominator_column="impressions",
            )
        ],
        sequential=SequentialConfig(
            enabled=True,
            known_variance=1.0,
            mixture_variance=1.0,
            alpha=0.05,
        ),
    )

    service = ExperimentAnalysisService(config)
    manifest = service.run_full_analysis(df, run_id="run_ratio_seq_test")

    assert manifest.sequential_result is not None
    seq_res = manifest.sequential_result
    assert seq_res["eligible"] is False
    assert "ratio_of_sums" in seq_res["ineligibility_reason"]
    assert any("ineligible" in w.lower() for w in manifest.warnings)

