# ExPO — Experimentation & Product Outcomes Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ExPO** is a local-first experimentation engine and portfolio-grade research suite for planning, analyzing, and validating online controlled experiments (A/B tests).

---

## Key Features

1. **Fixed-Horizon Trustworthy Analysis:** Welch's unequal-variance t-test for continuous and binary outcomes with exact Welch-Satterthwaite degrees of freedom and clear absolute vs relative lift semantics.
2. **CUPED / ANCOVA Variance Reduction:** Optimal pre-experiment linear adjustment and robust regression (HC1/HC2) with predeclared missing covariate policies.
3. **Ratio Metrics & Delta Method:** User-level ratio-of-sums inference respecting the randomization unit, using asymptotic covariance and non-parametric bootstrap sensitivity.
4. **Data Integrity & SRM Diagnostics:** Chi-square goodness-of-fit and exact binomial tests for Sample Ratio Mismatch (SRM) with small-count guardrails.
5. **Experiment Design & Power Planning:** Continuous and binary MDE, sample size, and duration calculators with CUPED projections.
6. **Mathematically Scoped Sequential Engine:** Normal-mixture evidence process (mSPRT) with running-maximum anytime p-values and confidence sequences under explicit known-variance assumptions.
7. **Benjamini-Hochberg FDR Control:** Multiplicity correction for declared fixed-horizon metric families.
8. **Reproducible Monte Carlo Suite:** 5 empirical research studies (Peeking Inflation, CUPED Gain, A/A Calibration, Ratio Coverage, SRM Power) with Wilson score confidence intervals.
9. **Interactive UI & CLI:** Streamlit application (7 dedicated labs) and Typer CLI (`expo`).

---

## Installation

```bash
git clone https://github.com/expo/expo.git
cd ExPO
pip install -e .
```

## Quickstart

Run a complete synthetic experiment analysis demo:
```bash
expo demo
```

Launch the Streamlit interactive application:
```bash
streamlit run app/streamlit_app.py
```
