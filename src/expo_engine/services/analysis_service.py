"""
Unified analysis and experimentation service layer.
Coordinates data loading, validation, SRM checking, fixed-horizon Welch tests,
CUPED covariate adjustment, ratio metrics, multiple testing, and sequential testing.
Called identically by both CLI commands and Streamlit UI pages.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from expo_engine.checks.srm import SRMResult, check_srm
from expo_engine.checks.validation import ValidationResult, validate_experiment_data
from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole
from expo_engine.data.loader import load_dataset
from expo_engine.reporting.schemas import (
    CUPEDResult,
    ExperimentReportManifest,
    MultiplicityResult,
    RatioResult,
    SequentialResult,
    WelchResult,
)
from expo_engine.statistics.cuped import compute_cuped
from expo_engine.statistics.multiplicity import apply_benjamini_hochberg
from expo_engine.statistics.ratio import compute_ratio_metric
from expo_engine.statistics.sequential import evaluate_sequential_monitoring
from expo_engine.statistics.welch import compute_welch_ttest


class ExperimentAnalysisService:
    """
    Core engine service for analyzing experiments.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        run_id: str = "run_001",
        run_ratio_bootstrap: bool = False,
    ) -> ExperimentReportManifest:
        """
        Executes end-to-end experiment analysis on a canonical user-level dataframe.
        """
        warnings: List[str] = []

        # 1. Validation check
        val_result: ValidationResult = validate_experiment_data(df, self.config)
        if not val_result.is_valid:
            error_msg = "; ".join(val_result.errors)
            raise ValueError(f"Experiment validation failed: {error_msg}")
        warnings.extend(val_result.warnings)

        # 2. Sample Ratio Mismatch check
        srm_res: SRMResult = check_srm(
            df=df,
            variant_col=self.config.variant_col,
            planned_ratio=self.config.planned_ratio,
            threshold=self.config.srm_threshold,
            level="assignment",
        )
        if srm_res.status == "SRM_DETECTED":
            warnings.append(srm_res.message)

        # 3. Primary Metric Analysis
        primary_metric = self.config.get_primary_metric()
        primary_welch_res: Optional[WelchResult] = None
        cuped_res: Optional[CUPEDResult] = None
        ratio_results_list: List[RatioResult] = []

        if primary_metric:
            if primary_metric.kind == MetricKind.RATIO_OF_SUMS:
                primary_ratio_res = compute_ratio_metric(
                    df=df,
                    numerator_col=primary_metric.numerator_column,
                    denominator_col=primary_metric.denominator_column,
                    variant_col=self.config.variant_col,
                    control_variant=self.config.control_variant,
                    treatment_variant=self.config.treatment_variant,
                    alpha=self.config.alpha,
                    run_bootstrap=run_ratio_bootstrap,
                )
                ratio_results_list.append(primary_ratio_res)
                warnings.extend(primary_ratio_res.warnings)
            else:
                col = primary_metric.column or primary_metric.name
                c_vals = df.loc[df[self.config.variant_col] == self.config.control_variant, col].values
                t_vals = df.loc[df[self.config.variant_col] == self.config.treatment_variant, col].values
                primary_welch_res = compute_welch_ttest(
                    control_vals=c_vals,
                    treatment_vals=t_vals,
                    metric_name=primary_metric.name,
                    alpha=self.config.alpha,
                )
                warnings.extend(primary_welch_res.warnings)

                # If pre-covariate declared, compute CUPED
                if primary_metric.pre_covariate and primary_metric.pre_covariate in df.columns:
                    cuped_res = compute_cuped(
                        df=df,
                        metric_col=col,
                        pre_covariate_col=primary_metric.pre_covariate,
                        variant_col=self.config.variant_col,
                        control_variant=self.config.control_variant,
                        treatment_variant=self.config.treatment_variant,
                        alpha=self.config.alpha,
                        missing_policy=self.config.missing_covariate_policy,
                    )
                    warnings.extend(cuped_res.warnings)

        # 4. Secondary & Guardrail Metrics (Fixed-Horizon Multiple Testing)
        secondary_metrics = self.config.get_secondary_metrics()
        secondary_pvals: List[Tuple[str, float]] = []

        for sec_metric in secondary_metrics:
            if sec_metric.kind == MetricKind.RATIO_OF_SUMS:
                r_res = compute_ratio_metric(
                    df=df,
                    numerator_col=sec_metric.numerator_column,
                    denominator_col=sec_metric.denominator_column,
                    variant_col=self.config.variant_col,
                    control_variant=self.config.control_variant,
                    treatment_variant=self.config.treatment_variant,
                    alpha=self.config.alpha,
                    run_bootstrap=False,
                )
                ratio_results_list.append(r_res)
                secondary_pvals.append((sec_metric.name, r_res.p_value))
            else:
                s_col = sec_metric.column or sec_metric.name
                if s_col in df.columns:
                    c_s = df.loc[df[self.config.variant_col] == self.config.control_variant, s_col].values
                    t_s = df.loc[df[self.config.variant_col] == self.config.treatment_variant, s_col].values
                    w_res = compute_welch_ttest(c_s, t_s, metric_name=sec_metric.name, alpha=self.config.alpha)
                    secondary_pvals.append((sec_metric.name, w_res.p_value))

        multiplicity_res: Optional[MultiplicityResult] = None
        if secondary_pvals:
            multiplicity_res = apply_benjamini_hochberg(
                hypotheses=secondary_pvals,
                fdr_q=self.config.alpha,
                family_name="secondary_and_guardrail_metrics",
            )

        # 5. Sequential Monitoring (if enabled and eligible)
        seq_res: Optional[SequentialResult] = None
        if self.config.sequential and self.config.sequential.enabled:
            primary_col = primary_metric.column if primary_metric else None
            if primary_metric and primary_col in df.columns:
                c_seq = df.loc[df[self.config.variant_col] == self.config.control_variant, primary_col].values
                t_seq = df.loc[df[self.config.variant_col] == self.config.treatment_variant, primary_col].values
                seq_res = evaluate_sequential_monitoring(
                    treatment_stream=t_seq,
                    control_stream=c_seq,
                    known_variance=self.config.sequential.known_variance,
                    mixture_variance=self.config.sequential.mixture_variance,
                    alpha=self.config.sequential.alpha,
                    metric_name=primary_metric.name,
                    metric_kind=primary_metric.kind.value,
                )
                warnings.extend(seq_res.warnings)

        # Data quality status string
        dq_status = "SRM_ALERT" if srm_res.status == "SRM_DETECTED" else ("WARNINGS" if warnings else "CLEAN")

        # Assemble manifest
        manifest = ExperimentReportManifest(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            data_quality_status=dq_status,
            srm_result=srm_res.model_dump(),
            primary_metric_result=primary_welch_res.model_dump() if primary_welch_res else (
                ratio_results_list[0].model_dump() if ratio_results_list else None
            ),
            cuped_result=cuped_res.model_dump() if cuped_res else None,
            ratio_results=[r.model_dump() for r in ratio_results_list],
            secondary_multiplicity=multiplicity_res,
            sequential_result=seq_res.model_dump() if seq_res else None,
            warnings=list(dict.fromkeys(warnings)),  # unique warnings
            reproducibility_command=f"expo analyze --config config.yaml",
        )
        return manifest
