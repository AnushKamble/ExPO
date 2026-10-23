# ExPO Simulation Protocol & Research Suite

This document defines the data-generating processes (DGPs), Monte Carlo protocols, calibration standards, uncertainty quantification, and acceptance rules for all simulation studies in ExPO.

---

## 1. Data-Generating Processes (DGPs)

All synthetic generation utilizes `numpy.random.Generator` with explicitly logged integer seeds. Generation parameters are stored strictly in simulation metadata and are not accessible to analysis estimators.

### 1.1 Scenario 1: `null_normal` (Gaussian Null with Pre-Covariate)
- **Covariate Generation:** $X_i \sim \mathcal{N}(\mu_X, \sigma_X^2)$
- **Baseline Outcome:** $Y_i(0) = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(X_i - \mu_X) + \sqrt{1 - \rho^2}\sigma_Y \epsilon_i$, where $\epsilon_i \sim \mathcal{N}(0, 1)$
- **Treatment Assignment:** $T_i \sim \text{Bernoulli}(0.5)$
- **Treatment Effect:** $\delta = 0 \implies Y_i(1) = Y_i(0)$
- **Observed Outcome:** $Y_i = Y_i(0)$
- **Ground Truth Parameters:** $\Delta = 0, \operatorname{Var}(Y) = \sigma_Y^2, \operatorname{Corr}(X, Y) = \rho$.

### 1.2 Scenario 2: `effect_normal` (Gaussian Additive Alternative)
- Same covariate and baseline outcome as `null_normal`.
- **Treatment Effect:** $\delta > 0 \implies Y_i(1) = Y_i(0) + \delta$.
- **Observed Outcome:** $Y_i = Y_i(0) + \delta \cdot T_i$.
- **Ground Truth Parameters:** $\Delta = \delta, \operatorname{Var}(Y) = \sigma_Y^2, \operatorname{Corr}(X, Y) = \rho$.

### 1.3 Scenario 3: `null_heavy_tailed` (Heavy-Tailed / Skewed Outcomes)
- Outcomes sampled from Student's $t(\nu=3)$ or Log-Normal($\mu=0, \sigma=1.2$).
- Treatment effect $\delta = 0$.
- Serves to evaluate breakdown points of Gaussian sequential assumptions and fixed-horizon robustness.

### 1.4 Scenario 4: `conversion` (Binary Outcome)
- Control conversion probability: $p_0 \in (0, 1)$.
- Treatment conversion probability: $p_1 = p_0 + \delta$.
- $Y_{C,i} \sim \text{Bernoulli}(p_0), \quad Y_{T,i} \sim \text{Bernoulli}(p_1)$.

### 1.5 Scenario 5: `ratio_metric` (Correlated Numerator & Denominator)
- Denominator $X_i \sim \text{Poisson}(\lambda_X) + 1$ (strictly positive user visits/sessions).
- Numerator $Y_i \mid X_i \sim \text{Binomial}(X_i, p_{rate})$ or $Y_i \sim \text{Gamma}(k_Y, \theta_Y)$.
- Ground truth ratio: $R = E[Y] / E[X]$.

### 1.6 Scenario 6: `srm_injected` (Corrupted Assignment)
- Configured expected allocation: 50% Control, 50% Treatment.
- Realized traffic generation: $T_i \sim \text{Bernoulli}(p_{\text{true}})$ where $p_{\text{true}} \in \{0.51, 0.52, 0.55\}$.

---

## 2. Monte Carlo Uncertainty Quantification

For any empirical rejection rate $\hat{p} = \frac{K}{M}$ across $M$ independent Monte Carlo replications:

### 2.1 Monte Carlo Standard Error (MCSE)
$$\text{MCSE}(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{M}}$$

### 2.2 Wilson Score Confidence Interval
To ensure coverage near boundary values (e.g. $\alpha = 0.05$ or $0.01$), ExPO reports two-sided 95% Wilson score intervals:
$$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2M} \pm z\sqrt{\frac{\hat{p}(1 - \hat{p})}{M} + \frac{z^2}{4M^2}}}{1 + \frac{z^2}{M}}$$
where $z = 1.95996$ for 95% confidence.

---

## 3. Presets & Execution Tiers

| Preset | Replications ($M$) | Target Runtime | Purpose |
|---|---|---|---|
| `smoke` | 100 – 200 | $< 5$ seconds | Fast CI, pre-commit validation, quick CLI demo |
| `medium` | 1,000 | $< 25$ seconds | Standard integration and calibration verification |
| `full` | 5,000 – 10,000 | 1 – 3 minutes | Research-grade calibration, publication/portfolio tables |

All simulation loops are vectorized across replications where feasible to achieve fast local CPU execution without parallel race conditions.

---

## 4. Five Core Research Studies

### Study A: Peeking Inflation vs Sequential Evidence
- **Scenarios:** Null ($\delta = 0$) and Non-Null ($\delta = 0.15 \sigma$).
- **Look Schedule:** $K = 30$ looks (e.g. daily for 30 days, $n=100$ pairs per look, total $N=3,000$).
- **Methods Compared:**
  1. *Fixed-Horizon Welch:* Single test evaluated strictly at look 30.
  2. *Naive Daily Welch:* Evaluated at each look $k \in \{1,\ldots,30\}$; early stop upon $p < 0.05$.
  3. *Gaussian Mixture Sequential:* Evaluated at each look $k$; early stop upon $p_k^{AV} \le 0.05$.
- **Outputs:** Empirical Type I error with Wilson CI, empirical power, stopping time distribution.

### Study B: CUPED Variance Reduction & CI Shrinkage
- **Scenarios:** Correlation grid $\rho \in [0.0, 0.2, 0.5, 0.7, 0.9]$.
- **Metrics Evaluated:**
  - Empirical variance ratio: $\widehat{\operatorname{Var}}(\hat{\Delta}_{CUPED}) / \widehat{\operatorname{Var}}(\hat{\Delta}_{raw})$.
  - Mean CI width reduction: $1 - \bar{W}_{CUPED} / \bar{W}_{raw}$.
  - Coverage of 95% CI (must remain $\approx 95\%$).
  - Empirical power gain at fixed sample size.

### Study C: A/A Calibration
- **Scenarios:** Null condition ($\delta = 0$), $M=2,000$ replications.
- **Metrics Evaluated:**
  - Empirical Type I error at nominal $\alpha = 0.05$.
  - Kolmogorov-Smirnov test of uniformity for p-values on $[0, 1]$.
  - P-value histogram and QQ plot.

### Study D: Ratio Delta Method vs Bootstrap Sensitivity
- **Scenarios:** Ratio metrics with skewness and varying correlation between numerator and denominator.
- **Metrics Evaluated:**
  - Coverage probability of Delta-method 95% CI vs Bootstrap percentile CI.
  - Standard error ratio $SE_{\text{Delta}} / SE_{\text{Bootstrap}}$.

### Study E: SRM Detection Power
- **Scenarios:** Imbalance shifts $p \in [0.50, 0.505, 0.51, 0.52, 0.55]$, sample sizes $N \in [1,000, 10,000, 50,000]$.
- **Metrics Evaluated:** Detection probability at $\alpha = 0.001$.
