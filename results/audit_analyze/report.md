# ExPO Experiment Analysis Report: `demo_experiment_001`
- **Run ID:** `run_cli`
- **Timestamp (UTC):** `2026-10-23T18:10:38.107180+00:00`
- **Engine Version:** `0.1.0`
- **Data Quality Status:** **WARNINGS**

---

## 1. Data Integrity & Sample Ratio Mismatch (SRM)
- **Status:** `PASS`
- **Goodness-of-Fit p-value:** `2.2821e-01` (Threshold: `0.001`)
- **Method:** `chi_square`
- **Observed Counts:** {'control': 1467, 'treatment': 1533}
- **Expected Counts:** {'control': 1500.0, 'treatment': 1500.0}
- **Audit Message:** No Sample Ratio Mismatch detected (assignment level, p=0.2282 >= 0.001). Observed traffic matches planned allocation within expected random variation.

---

## 2. Primary Metric Inference
### Metric: `revenue` (Unadjusted Baseline)
- **Method:** Welch's Unequal-Variance Two-Sample t-Test
- **Treatment Effect Estimate:** `1.9263` (Relative Lift: `3.86%`)
- **Standard Error:** `0.3650`
- **95% Confidence Interval:** `[1.2106, 2.6421]`
- **p-value:** `1.4070e-07`
- **Statistically Significant (alpha=0.05):** `True`

### CUPED & ANCOVA Covariate Adjustment (`pre_metric`)
- **Adjusted Treatment Effect:** `1.9236`
- **Robust Standard Error (HC1):** `0.2640` (vs Raw SE `0.3650`)
- **Adjusted 95% CI:** `[1.4060, 2.4413]`
- **Adjusted p-value:** `3.1892e-13`
- **Covariate Correlation (rho):** `0.6989`
- **Theoretical Variance Reduction (1 - rho^2):** `48.84%`
- **Empirical Variance Reduction:** `48.84%`
- **Complete Cases Used:** `2913` (Missing pre-covariate: `87`)

---

## 3. Secondary & Guardrail Metrics (Benjamini-Hochberg FDR)
- **Correction Method:** Benjamini-Hochberg (BH) Step-Up Procedure
- **Declared Target FDR (q):** `0.05`
- **Total Tests:** `1` | **Rejections:** `1`

| Metric | Raw p-value | BH Adjusted p-value | Rejected (FDR Controlled) |
|---|---|---|---|
| `page_views` | `0.0000` | `0.0000` | `YES` |

---

## 4. Sequential Monitoring (Gaussian Mixture e-Process)
- **Model Eligibility:** `True`
- **Frozen Pilot Variance (sigma_D^2):** `200.0`
- **Predeclared Mixture Variance (tau^2):** `1.0`
- **Total Balanced Pairs Monitored:** `1467` across `147` looks
- **Stopped Early:** `YES (Look 75)`
- **Final Anytime-Valid p-value:** `8.6349e-06`
- **Final Time-Uniform Confidence Sequence:** `[0.9473, 3.0050]`

---

## 5. Audit Warnings & Caveats
- > [!WARNING] Pre-covariate 'pre_metric' is missing for 87 of 3000 rows (2.9%). Policy 'complete_case' applied: adjusted estimate computed on complete cases only; unadjusted retains all valid outcomes.
- > [!WARNING] Accrual streams unbalanced (Treatment=1533, Control=1467). Sequential evaluation truncated to 1467 balanced observation pairs.