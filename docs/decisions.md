# ExPO Architecture & Statistical Decision Records (ADR)

## ADR-001: Architecture Boundary & Service Layer Isolation
- **Context:** Statistical algorithms and data transformations must not be duplicated across CLI, Streamlit UI, or scripts.
- **Decision:** Implement all data ingestion, validation, estimation, sequential testing, and simulation logic in `src/expo_engine/`. CLI and Streamlit pages invoke unified service classes in `expo_engine.services`. Neither UI nor CLI contains custom mathematical formulas.
- **Consequence:** Full testability in headless CI, perfect numerical consistency across user interfaces, zero UI-dependent code in test suites.

## ADR-002: Statistical Scope of Sequential Engine
- **Context:** Naive repeated p-value checks cause substantial Type I error inflation. However, claiming universal anytime validity without strict assumptions is mathematically dishonest.
- **Decision:** The sequential engine strictly supports a two-arm balanced Gaussian mean-difference model with independent increments $D_i = Y_{T,i} - Y_{C,i}$, where $\sigma_D^2 = \sigma_T^2 + \sigma_C^2$ is fixed prior to monitoring from an independent pilot or historical dataset. The mixture prior variance $\tau^2 > 0$ and stopping boundary $1/\alpha$ are fixed before the test. Anytime p-value is computed via the running supremum $M_n = \max_{1 \le k \le n} E_k$ using $p_n^{AV} = \min(1, 1/M_n)$. Confidence sequence is the running intersection of mixture-evidence inversion intervals.
- **Consequence:** If data is non-normal, variance is estimated from ongoing experiment outcomes, arms are severely unbalanced, or metric is an unsupported ratio metric without derivation, sequential testing is explicitly blocked with a clear precondition error.

## ADR-003: CUPED & ANCOVA Variance Reduction
- **Context:** Pre-experiment covariates correlate with post-treatment outcomes and can reduce estimation variance without altering the causal estimand under random assignment.
- **Decision:**
  1. Primary adjusted inference uses linear ANCOVA regression ($Y_i = \beta_0 + \tau T_i + \beta_1 X_i + \epsilon_i$) with heteroscedasticity-robust standard errors (HC1/HC2).
  2. Implement standard CUPED transformation $Y_i^{adj} = Y_i - \hat{\theta}(X_i - \bar{X})$ where $\hat{\theta} = \operatorname{\widehat{Cov}}(Y, X) / \operatorname{\widehat{Var}}(X)$ as an explicit cross-check.
  3. Predeclared missing covariate policy: report missingness rates by arm. Compute unadjusted estimate on all valid outcomes and adjusted estimate on complete cases. Disallow silent imputation.
- **Consequence:** Prevents biased causal inference while providing verifiable variance reduction.

## ADR-004: Ratio Metrics via User-Level Delta Method
- **Context:** Many core product metrics (e.g. click-through rate, revenue per session) are ratios of sums across users. Treating individual events as independent violates SUTVA and unit-of-randomization assumptions when users are randomized.
- **Decision:** Aggregate numerator $Y_i$ and denominator $X_i$ to the user level (randomization unit). Compute the ratio of sums $\hat{R} = \sum Y_i / \sum X_i = \bar{Y} / \bar{X}$. Estimate variance within each arm using the multivariate delta method:
  $$\widehat{\operatorname{Var}}(\hat{R}) \approx \frac{1}{n\bar{X}^2}\left(s_Y^2 - 2\hat{R}s_{YX} + \hat{R}^2 s_X^2\right) = \frac{1}{n} s_Z^2 \quad \text{where } Z_i = \frac{Y_i - \hat{R}X_i}{\bar{X}}.$$
  Provide user-level non-parametric bootstrap as a sensitivity comparator. Warn when denominator is near zero or coefficient of variation of $X$ is high.
- **Consequence:** Statistically honest inference for ratio metrics that respects user clustering.

## ADR-005: Sample Ratio Mismatch (SRM) and Gatekeeping
- **Context:** Imbalance between observed and planned variant traffic indicates assignment bias, tracking failures, or differential survivorship, invalidating causal conclusions.
- **Decision:** Perform chi-square goodness-of-fit test against declared allocation ratios with default warning threshold $p < 0.001$. For small expected cell counts ($< 10$), issue warnings regarding asymptotic validity. Differentiate assignment-level vs exposure-level SRM when timestamps/tables are available. If SRM is triggered, analysis results are flagged as CAUTION/BLOCKED, preventing automated optimistic conclusions.
- **Consequence:** Guards against interpreting corrupted experiments.

## ADR-006: Multiple Testing Correction Policy
- **Context:** Testing multiple secondary or guardrail metrics inflates the False Discovery Rate (FDR).
- **Decision:** Implement Benjamini-Hochberg (BH) step-up procedure for predeclared families of fixed-horizon secondary and guardrail metrics. Explicitly prohibit applying standard BH across sequential peeks, as BH does not control sequential FDR. MVP sequential testing is restricted to one primary metric.
- **Consequence:** Clear boundaries between multiplicity correction at fixed horizons and sequential monitoring.
