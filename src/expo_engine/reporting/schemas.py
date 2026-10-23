"""
Versioned result schemas and analysis manifests for ExPO.
All statistical engines return validated Pydantic models with explicit estimands,
assumptions, confidence intervals, sample sizes, and diagnostic warnings.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GroupSummary(BaseModel):
    variant: str
    sample_size: int
    mean: float
    variance: float
    std_dev: float
    min_val: float
    max_val: float
    median: float
    iqr: float


class WelchResult(BaseModel):
    metric_name: str
    estimand: str = "Average Treatment Effect (Delta = E[Y_T] - E[Y_C])"
    method: str = "Welch's Unequal-Variance Two-Sample t-Test"
    control_summary: GroupSummary
    treatment_summary: GroupSummary
    absolute_effect: float
    relative_lift: Optional[float] = None
    relative_lift_valid: bool = True
    standard_error: float
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    alpha: float = 0.05
    ci_lower: float
    ci_upper: float
    is_statistically_significant: bool
    warnings: List[str] = Field(default_factory=list)
    mann_whitney_p_value: Optional[float] = None
    mann_whitney_note: Optional[str] = None


class CUPEDResult(BaseModel):
    metric_name: str
    covariate_name: str
    estimand: str = "Covariate-Adjusted Average Treatment Effect"
    unadjusted_welch: WelchResult
    adjusted_ate: float
    adjusted_se: float
    adjusted_ci_lower: float
    adjusted_ci_upper: float
    adjusted_p_value: float
    is_statistically_significant: bool
    theta: float
    sample_correlation: float
    theoretical_variance_reduction: float
    empirical_variance_reduction: float
    complete_cases_count: int
    missing_covariate_count: int
    unadjusted_all_valid_count: int
    ancova_robust_se: float
    ancova_p_value: float
    warnings: List[str] = Field(default_factory=list)


class RatioResult(BaseModel):
    metric_name: str
    estimand: str = "Difference in Ratios of Sums (Delta_R = R_T - R_C)"
    method: str = "Multivariate Delta Method (Randomization Unit Level)"
    control_ratio: float
    treatment_ratio: float
    control_se: float
    treatment_se: float
    control_numerator_sum: float
    control_denominator_sum: float
    treatment_numerator_sum: float
    treatment_denominator_sum: float
    delta_ratio: float
    relative_lift: Optional[float] = None
    standard_error: float
    z_statistic: float
    p_value: float
    alpha: float = 0.05
    ci_lower: float
    ci_upper: float
    is_statistically_significant: bool
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class SequentialLook(BaseModel):
    look_index: int
    sample_size_pairs: int
    cumulative_mean_diff: float
    log_evidence: float
    evidence: float
    running_max_log_evidence: float
    running_max_evidence: float
    anytime_p_value: float
    boundary_crossed: bool
    cs_lower: float
    cs_upper: float


class SequentialResult(BaseModel):
    metric_name: str
    estimand: str = "Sequential Mean Difference under Known Variance Gaussian Increments"
    method: str = "Gaussian Normal-Mixture Evidence Process (mSPRT)"
    eligible: bool
    ineligibility_reason: Optional[str] = None
    known_difference_variance: float
    mixture_variance: float
    alpha: float
    stopping_threshold: float
    total_looks_analyzed: int
    total_pairs: int
    stopped_early: bool
    stopping_look: Optional[int] = None
    final_mean_diff: float
    final_evidence: float
    final_running_max_evidence: float
    final_anytime_p_value: float
    final_cs_lower: float
    final_cs_upper: float
    looks_history: List[SequentialLook] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AdjustedHypothesis(BaseModel):
    metric_name: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool


class MultiplicityResult(BaseModel):
    method: str = "Benjamini-Hochberg (BH) Step-Up Procedure"
    family_name: str
    fdr_q: float
    hypotheses: List[AdjustedHypothesis] = Field(default_factory=list)
    total_tests: int
    rejections_count: int


class ExperimentReportManifest(BaseModel):
    run_id: str
    experiment_id: str
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expo_version: str = "0.1.0"
    input_file_hash: Optional[str] = None
    data_quality_status: str
    srm_result: Optional[Dict[str, Any]] = None
    primary_metric_result: Optional[Dict[str, Any]] = None
    cuped_result: Optional[Dict[str, Any]] = None
    ratio_results: List[Dict[str, Any]] = Field(default_factory=list)
    secondary_multiplicity: Optional[MultiplicityResult] = None
    sequential_result: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    reproducibility_command: Optional[str] = None
