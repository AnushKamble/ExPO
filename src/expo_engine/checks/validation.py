"""
Data validation and contract enforcement for experimentation datasets.
Catches duplicate units, variant contamination, missing columns, non-finite values,
and invalid metric values before statistical estimation.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole


class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="True if no fatal errors were detected")
    errors: List[str] = Field(default_factory=list, description="Fatal errors blocking analysis")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal data quality warnings")
    total_rows: int = 0
    unique_users: int = 0
    variant_counts: Dict[str, int] = Field(default_factory=dict)
    missing_rates: Dict[str, float] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


def validate_experiment_data(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> ValidationResult:
    """
    Validates a canonical user-level experiment dataframe against the experiment configuration.
    """
    errors: List[str] = []
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {}

    total_rows = len(df)
    if total_rows == 0:
        return ValidationResult(
            is_valid=False,
            errors=["Dataset is empty (0 rows)."],
            warnings=[],
            total_rows=0,
            unique_users=0,
        )

    # 1. Required column checks
    required_cols = [config.user_id_col, config.variant_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            total_rows=total_rows,
            unique_users=0,
        )

    # 2. Check user ID uniqueness (canonical grain)
    unique_users = df[config.user_id_col].nunique()
    if unique_users < total_rows:
        num_dupes = total_rows - unique_users
        errors.append(
            f"Duplicate user IDs detected: {num_dupes} duplicate rows found. "
            "Data must be aggregated to exactly one row per analysis unit."
        )

    # 3. Check variant contamination and unrecognized variants
    user_variant_counts = df.groupby(config.user_id_col)[config.variant_col].nunique()
    contaminated_users = (user_variant_counts > 1).sum()
    if contaminated_users > 0:
        errors.append(
            f"Variant contamination detected: {contaminated_users} users are assigned to multiple variants."
        )

    observed_variants = set(df[config.variant_col].dropna().unique())
    expected_variants = {config.control_variant, config.treatment_variant}
    unexpected_variants = observed_variants - expected_variants
    if unexpected_variants:
        errors.append(
            f"Unexpected variants found: {unexpected_variants}. Expected: {expected_variants}"
        )

    variant_counts = df[config.variant_col].value_counts().to_dict()
    for exp_var in expected_variants:
        if exp_var not in variant_counts:
            errors.append(f"No units found for variant '{exp_var}'.")

    # Minimum sample size check
    for v_name, count in variant_counts.items():
        if count < 10:
            warnings.append(f"Sample size for variant '{v_name}' is very small ({count} units). Asymptotics may be invalid.")

    # 4. Metric-specific validations
    missing_rates: Dict[str, float] = {}
    for metric in config.metrics:
        if metric.kind == MetricKind.RATIO_OF_SUMS:
            # Check numerator and denominator columns
            num_col = metric.numerator_column
            denom_col = metric.denominator_column
            if num_col not in df.columns:
                errors.append(f"Numerator column '{num_col}' missing for metric '{metric.name}'.")
            if denom_col not in df.columns:
                errors.append(f"Denominator column '{denom_col}' missing for metric '{metric.name}'.")

            if num_col in df.columns and denom_col in df.columns:
                # Check non-finite
                if not np.isfinite(df[num_col]).all():
                    errors.append(f"Non-finite values (NaN, Inf) found in numerator column '{num_col}'.")
                if not np.isfinite(df[denom_col]).all():
                    errors.append(f"Non-finite values (NaN, Inf) found in denominator column '{denom_col}'.")
                # Check non-negative denominator
                if (df[denom_col] < 0).any():
                    errors.append(f"Negative values detected in denominator column '{denom_col}'.")
                # Check total denominator per arm > 0
                for v_name in observed_variants:
                    arm_denom_sum = df.loc[df[config.variant_col] == v_name, denom_col].sum()
                    if arm_denom_sum <= 0:
                        errors.append(f"Total denominator for variant '{v_name}' is <= 0 for metric '{metric.name}'.")
        else:
            col = metric.column or metric.name
            if col not in df.columns:
                errors.append(f"Metric column '{col}' missing for metric '{metric.name}'.")
                continue

            # Non-finite checks
            non_finite_count = (~np.isfinite(df[col])).sum()
            if non_finite_count > 0:
                errors.append(f"Non-finite values (NaN/Inf) found in metric '{col}': {non_finite_count} rows.")

            # Binary checks
            if metric.kind == MetricKind.BINARY_MEAN:
                unique_vals = set(df[col].dropna().unique())
                if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                    errors.append(f"Binary metric '{col}' contains non-binary values: {unique_vals}.")

            # CUPED Pre-covariate checks
            if metric.pre_covariate:
                pre_col = metric.pre_covariate
                if pre_col not in df.columns:
                    warnings.append(f"Pre-covariate column '{pre_col}' not found. CUPED will be disabled.")
                else:
                    pre_missing = df[pre_col].isna().mean()
                    missing_rates[pre_col] = float(pre_missing)
                    if pre_missing > 0.50:
                        warnings.append(f"High missingness in pre-covariate '{pre_col}': {pre_missing:.1%}.")

                    # Differential missingness check across arms
                    if pre_missing > 0 and len(observed_variants) == 2:
                        contingency = pd.crosstab(df[config.variant_col], df[pre_col].isna())
                        if contingency.shape == (2, 2):
                            _, p_diff, _, _ = stats.chi2_contingency(contingency)
                            diagnostics["covariate_differential_missing_pvalue"] = float(p_diff)
                            if p_diff < 0.05:
                                warnings.append(
                                    f"Differential missingness detected in pre-covariate '{pre_col}' "
                                    f"(p={p_diff:.4f}). Covariate missingness may be affected by treatment."
                                )

                    # Check variance > 0
                    valid_pre = df[pre_col].dropna()
                    if len(valid_pre) > 1 and valid_pre.var() == 0:
                        warnings.append(f"Pre-covariate '{pre_col}' has zero variance (constant). CUPED cannot reduce variance.")

    is_valid = (len(errors) == 0)
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        total_rows=total_rows,
        unique_users=unique_users,
        variant_counts=variant_counts,
        missing_rates=missing_rates,
        diagnostics=diagnostics,
    )
