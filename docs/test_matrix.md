# ExPO Comprehensive Test Matrix & Validation Mapping

This document provides a 1-to-1 mapping between the specification requirements, implemented modules, and the automated test suite.

| Requirement ID | Module / Feature | Test File | Target Function / Test Case | Acceptance Criteria |
|---|---|---|---|---|
| REQ-01 | Deterministic Hashing | `tests/unit/test_assignment.py` | `test_sha256_assignment_determinism`, `test_assignment_row_order_invariance` | Stable variant assignment regardless of row order or system process |
| REQ-02 | Schema Validation | `tests/unit/test_checks.py` | `test_missing_required_columns`, `test_duplicate_users_rejected`, `test_variant_contamination_flagged` | Raises actionable `ValidationError` with details |
| REQ-03 | Sample Ratio Mismatch (SRM) | `tests/unit/test_srm.py` | `test_srm_balanced_passes`, `test_srm_imbalanced_fails`, `test_srm_small_expected_counts_exact_test` | Chi-square p-value and alert flag computed correctly; exact binomial triggered if $E < 10$ |
| REQ-04 | DuckDB Ingestion & Aggregation | `tests/integration/test_pipeline.py` | `test_csv_ingestion_and_aggregation`, `test_parquet_ingestion_and_aggregation`, `test_sql_identifier_sanitization` | Correct user-level sum/count aggregation; SQL injection prevented |
| REQ-05 | Welch's Two-Sample t-Test | `tests/unit/test_welch.py` | `test_welch_known_answer_fixture`, `test_welch_relative_lift_zero_mean_warning`, `test_welch_directionality` | ATE, SE, Welch-Satterthwaite df, p-value, and 95% CI match reference SciPy within $10^{-6}$ |
| REQ-06 | CUPED / ANCOVA | `tests/unit/test_cuped.py` | `test_cuped_constant_covariate`, `test_cuped_known_correlation_reduction`, `test_ancova_robust_se`, `test_cuped_missingness_policy` | Realized variance reduction trends toward $1 - \rho^2$; zero variance handled gracefully; unadjusted and adjusted reported |
| REQ-07 | Ratio Delta Method | `tests/unit/test_ratio.py` | `test_ratio_delta_method_known_fixture`, `test_ratio_influence_function_equivalence`, `test_ratio_near_zero_denominator_warning` | Variance from bivariate delta method matches influence function formula exactly ($< 10^{-10}$ diff) |
| REQ-08 | Power, MDE, Sample Size | `tests/unit/test_power.py` | `test_continuous_sample_size_monotonicity`, `test_binary_sample_size_reference`, `test_cuped_projected_sample_size` | Matches analytical formulas; increases as $\alpha$ decreases or power increases |
| REQ-09 | Benjamini-Hochberg (BH) | `tests/unit/test_bh.py` | `test_bh_known_fixture`, `test_bh_monotonicity`, `test_bh_preserves_order_and_ties` | Adjusted p-values match reference statsmodels within $10^{-7}$; original order preserved |
| REQ-10 | Gaussian Mixture Sequential Engine | `tests/unit/test_sequential.py` | `test_sequential_known_answer_log_evidence`, `test_running_maximum_monotonicity`, `test_anytime_pvalue_non_increasing`, `test_confidence_sequence_inversion` | Running max $M_n$ non-decreasing; $p^{AV}$ non-increasing; confidence sequence contains truth with $\ge 95\%$ coverage |
| REQ-11 | Study A: Peeking Inflation | `tests/statistical_calibration/test_study_peeking.py` | `test_peeking_simulation_smoke` | Confirms naive peeking inflates Type I error while sequential engine preserves $\le \alpha$ |
| REQ-12 | Study B: CUPED Simulation | `tests/statistical_calibration/test_study_cuped.py` | `test_cuped_simulation_smoke` | Confirms empirical variance reduction matches $1 - \rho^2$ within Monte Carlo tolerance |
| REQ-13 | Study C: A/A Calibration | `tests/statistical_calibration/test_study_aa.py` | `test_aa_simulation_smoke` | Confirms empirical Type I error covers $\alpha=0.05$ within Wilson 95% CI |
| REQ-14 | Study D: Ratio Delta vs Bootstrap | `tests/statistical_calibration/test_study_ratio.py` | `test_ratio_simulation_smoke` | Confirms delta method coverage matches bootstrap coverage |
| REQ-15 | Study E: SRM Power Simulation | `tests/statistical_calibration/test_study_srm.py` | `test_srm_simulation_smoke` | Confirms SRM detection probability increases with sample size and imbalance magnitude |
| REQ-16 | End-to-End Pipeline & Reporting | `tests/integration/test_end_to_end.py` | `test_full_cli_and_report_generation` | Markdown, HTML, JSON, and CSV reports generated with non-empty results and manifests |
