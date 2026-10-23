"""
Configuration models and schemas for ExPO using Pydantic.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MetricKind(str, Enum):
    CONTINUOUS_MEAN = "continuous_mean"
    BINARY_MEAN = "binary_mean"
    COUNT_MEAN = "count_mean"
    RATIO_OF_SUMS = "ratio_of_sums"


class MetricRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


class MetricSpec(BaseModel):
    name: str = Field(..., description="Unique metric identifier in dataframe")
    label: str = Field(..., description="Human readable display label")
    kind: MetricKind = Field(MetricKind.CONTINUOUS_MEAN, description="Metric mathematical kind")
    role: MetricRole = Field(MetricRole.PRIMARY, description="Role in experiment decision")
    column: Optional[str] = Field(None, description="Column name for continuous or binary outcomes")
    numerator_column: Optional[str] = Field(None, description="Numerator column for ratio metrics")
    denominator_column: Optional[str] = Field(None, description="Denominator column for ratio metrics")
    pre_covariate: Optional[str] = Field(None, description="Column name of pre-experiment covariate for CUPED")
    direction: str = Field("increase", description="Expected direction of improvement: 'increase' or 'decrease'")
    unit: str = Field("", description="Unit string (e.g. $, %, ms)")

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("increase", "decrease"):
            raise ValueError("direction must be 'increase' or 'decrease'")
        return v

    def model_post_init(self, __context) -> None:
        if self.kind == MetricKind.RATIO_OF_SUMS:
            if not self.numerator_column or not self.denominator_column:
                raise ValueError("ratio_of_sums metric requires both numerator_column and denominator_column")
        else:
            if not self.column:
                # Default column to name if not provided
                self.column = self.name


class SequentialConfig(BaseModel):
    enabled: bool = False
    known_variance: float = Field(..., description="Pre-experiment frozen variance sigma_D^2 = sigma_T^2 + sigma_C^2")
    mixture_variance: float = Field(1.0, gt=0, description="Predeclared mixture prior variance tau^2 > 0")
    alpha: float = Field(0.05, gt=0, lt=1, description="Sequential significance level")

    @field_validator("known_variance")
    @classmethod
    def validate_variance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("known_variance must be strictly positive")
        return v


class ExperimentConfig(BaseModel):
    experiment_id: str = Field("exp_001", description="Identifier for experiment")
    user_id_col: str = Field("user_id", description="Column identifying unit of randomization")
    variant_col: str = Field("variant", description="Column identifying treatment assignment")
    control_variant: str = Field("control", description="Control arm label")
    treatment_variant: str = Field("treatment", description="Treatment arm label")
    planned_ratio: Dict[str, float] = Field(
        default_factory=lambda: {"control": 0.5, "treatment": 0.5},
        description="Planned allocation proportions"
    )
    alpha: float = Field(0.05, gt=0, lt=1, description="Significance level")
    target_power: float = Field(0.80, gt=0, lt=1, description="Target statistical power")
    srm_threshold: float = Field(0.001, gt=0, lt=1, description="p-value threshold for SRM alert")
    missing_covariate_policy: str = Field("complete_case", description="Policy: 'complete_case' or 'warn'")
    metrics: List[MetricSpec] = Field(default_factory=list, description="List of declared metrics")
    sequential: Optional[SequentialConfig] = None

    @field_validator("planned_ratio")
    @classmethod
    def validate_ratios(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("planned_ratio cannot be empty")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"planned_ratio values must sum to 1.0, got {total}")
        for k, p in v.items():
            if p <= 0:
                raise ValueError(f"Allocation proportion for {k} must be positive, got {p}")
        return v

    def get_primary_metric(self) -> Optional[MetricSpec]:
        primaries = [m for m in self.metrics if m.role == MetricRole.PRIMARY]
        if primaries:
            return primaries[0]
        return self.metrics[0] if self.metrics else None

    def get_secondary_metrics(self) -> List[MetricSpec]:
        return [m for m in self.metrics if m.role in (MetricRole.SECONDARY, MetricRole.GUARDRAIL)]
