# ExPO Statistical Methods & Mathematical Contracts

This document specifies the exact mathematical foundations, estimands, derivations, assumptions, limitations, and edge-case handling for all statistical procedures implemented in ExPO.

---

## 1. Welch's Two-Sample t-Test (Fixed-Horizon Continuous / Mean Inference)

### 1.1 Estimand
Let $Y_{T,i}$ ($i=1,\ldots,n_T$) and $Y_{C,j}$ ($j=1,\ldots,n_C$) denote independent post-treatment metric values for units assigned to Treatment ($T$) and Control ($C$).
The primary estimand is the Average Treatment Effect (ATE):
$$\Delta = E[Y_T] - E[Y_C]$$
Estimated by sample mean difference:
$$\hat{\Delta} = \bar{Y}_T - \bar{Y}_C = \frac{1}{n_T}\sum_{i=1}^{n_T} Y_{T,i} - \frac{1}{n_C}\sum_{j=1}^{n_C} Y_{C,j}$$

Relative lift is defined as:
$$\widehat{\text{Lift}}_{\text{rel}} = \frac{\bar{Y}_T - \bar{Y}_C}{\bar{Y}_C} = \frac{\hat{\Delta}}{\bar{Y}_C}$$
*Precondition / Edge Case:* If $|\bar{Y}_C| < 10^{-7}$, relative lift is mathematically unstable or undefined. ExPO issues an explicit warning and suppresses relative lift or marks it undefined.

### 1.2 Variance and Degrees of Freedom (Welch-Satterthwaite)
We do not assume homoscedasticity ($\sigma_T^2 = \sigma_C^2$). The unpooled standard error of $\hat{\Delta}$ is:
$$SE(\hat{\Delta}) = \sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}$$
where $s_T^2 = \frac{1}{n_T - 1}\sum_{i=1}^{n_T}(Y_{T,i}-\bar{Y}_T)^2$ and $s_C^2 = \frac{1}{n_C - 1}\sum_{j=1}^{n_C}(Y_{C,j}-\bar{Y}_C)^2$.

The effective degrees of freedom $\nu$ are given by the Welch-Satterthwaite equation:
$$\nu = \frac{\left(\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}\right)^2}{\frac{(s_T^2 / n_T)^2}{n_T - 1} + \frac{(s_C^2 / n_C)^2}{n_C - 1}}$$

The test statistic for $H_0: \Delta = 0$ is:
$$t = \frac{\hat{\Delta}}{SE(\hat{\Delta})}$$
Two-sided p-value:
$$p = 2 \cdot (1 - F_{t,\nu}(|t|))$$
where $F_{t,\nu}$ is the Student's t cumulative distribution function.

The two-sided $(1-\alpha)$ confidence interval is:
$$\hat{\Delta} \pm t_{1-\alpha/2,\nu} \cdot SE(\hat{\Delta})$$

### 1.3 Assumptions & Limitations
- **Assumptions:** Units are independently sampled and assigned; outcomes have finite second moments; sample sizes are large enough for Central Limit Theorem (CLT) approximation to hold, or outcomes are approximately normal.
- **Limitations:** Sensitive to extreme outliers and heavy-tailed Cauchy-like distributions. Rank-based Mann-Whitney tests can serve as non-parametric distribution sensitivity checks.

---

## 2. CUPED and ANCOVA Covariate Adjustment

### 2.1 CUPED Transformation
Controlled Experiments Using Pre-Experiment Data (Deng et al., 2013) utilizes pre-treatment covariates $X_i$ measured before assignment/exposure. Because $X_i$ is determined before randomization, $E[X_T] = E[X_C] = E[X]$ under clean randomization.

The adjusted outcome is:
$$Y_i^{adj} = Y_i - \theta (X_i - \bar{X})$$
where $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the pooled covariate mean.
The optimal coefficient minimizing $\operatorname{Var}(Y^{adj})$ is:
$$\theta^* = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}$$
Sample estimate:
$$\hat{\theta} = \frac{\sum_{i=1}^n (Y_i - \bar{Y})(X_i - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}$$

The sample treatment effect on adjusted outcomes is:
$$\hat{\Delta}_{CUPED} = \bar{Y}_T^{adj} - \bar{Y}_C^{adj} = (\bar{Y}_T - \bar{Y}_C) - \hat{\theta}(\bar{X}_T - \bar{X}_C)$$
Under linear model assumptions, the theoretical variance reduction ratio is:
$$\frac{\operatorname{Var}(Y^{adj})}{\operatorname{Var}(Y)} = 1 - \rho^2 \quad \text{where } \rho = \operatorname{Corr}(X, Y)$$

### 2.2 Regression / ANCOVA with Heteroscedasticity-Robust Standard Errors
As shown by Freedman (2008) and Lin (2013), ordinary least squares (OLS) regression of $Y_i$ on treatment indicator $T_i$ and centered covariate $(X_i - \bar{X})$:
$$Y_i = \alpha + \Delta_{adj} T_i + \beta (X_i - \bar{X}) + \epsilon_i$$
yields an asymptotically normal estimator of ATE. To guard against heteroscedasticity and potential finite-sample distortions, ExPO computes robust standard errors (HC1/HC2 sandwich variance estimator):
$$\widehat{\operatorname{Var}}_{HC}(\hat{\beta}) = (Z^T Z)^{-1} Z^T \Omega Z (Z^T Z)^{-1}$$
where $Z = [1, T, X - \bar{X}]$ and $\Omega = \operatorname{diag}(e_i^2 / (1 - h_{ii}))$.

### 2.3 Predeclared Missing Covariate Policy
- When $X_i$ is missing (e.g. new users with no pre-period activity), ExPO computes:
  1. Complete-case adjusted analysis on units where $X_i$ is observed.
  2. Unadjusted analysis on all units with observed $Y_i$.
  3. Reports sample sizes $n_{total}, n_{complete}$, missingness rate by arm, and differential missingness check.
- ExPO does not silently impute $X_i$ with zero or mean without logging and user consent.

---

## 3. Ratio Metrics and the Delta Method

### 3.1 Estimand: Ratio of Sums at the Unit of Randomization
When the randomization unit is a user, events or sessions within a user are correlated. Therefore, we must define the user-level aggregated numerator $Y_i$ (e.g., total purchases) and denominator $X_i$ (e.g., total visits).
The population ratio estimand for arm $g \in \{T, C\}$ is:
$$R_g = \frac{E[Y_{g,i}]}{E[X_{g,i}]}$$
Estimated by the sample ratio of sums:
$$\hat{R}_g = \frac{\sum_{i=1}^{n_g} Y_{g,i}}{\sum_{i=1}^{n_g} X_{g,i}} = \frac{\bar{Y}_g}{\bar{X}_g}$$
The treatment effect is the difference in ratios:
$$\Delta_R = R_T - R_C, \qquad \hat{\Delta}_R = \hat{R}_T - \hat{R}_C$$

### 3.2 Asymptotic Variance via the Delta Method
Let $g \in \{T, C\}$. Consider the bivariate function $f(\bar{Y}_g, \bar{X}_g) = \bar{Y}_g / \bar{X}_g$.
First-order Taylor expansion around $(\mu_{Y,g}, \mu_{X,g})$ gives:
$$\frac{\bar{Y}_g}{\bar{X}_g} - \frac{\mu_{Y,g}}{\mu_{X,g}} \approx \frac{1}{\mu_{X,g}}(\bar{Y}_g - \mu_{Y,g}) - \frac{\mu_{Y,g}}{\mu_{X,g}^2}(\bar{X}_g - \mu_{X,g}) = \frac{1}{n_g}\sum_{i=1}^{n_g} \left[\frac{Y_{g,i} - R_g X_{g,i}}{\mu_{X,g}}\right]$$

Thus, the asymptotic variance of $\hat{R}_g$ within arm $g$ is:
$$\operatorname{Var}(\hat{R}_g) \approx \frac{1}{n_g \bar{X}_g^2} \left[ s_{Y,g}^2 - 2\hat{R}_g s_{YX,g} + \hat{R}_g^2 s_{X,g}^2 \right]$$
where:
$$s_{Y,g}^2 = \frac{1}{n_g - 1}\sum_{i=1}^{n_g}(Y_{g,i}-\bar{Y}_g)^2, \quad s_{X,g}^2 = \frac{1}{n_g - 1}\sum_{i=1}^{n_g}(X_{g,i}-\bar{X}_g)^2$$
$$s_{YX,g} = \frac{1}{n_g - 1}\sum_{i=1}^{n_g}(Y_{g,i}-\bar{Y}_g)(X_{g,i}-\bar{X}_g)$$

Equivalently, defining the linearized influence values:
$$Z_{g,i} = \frac{Y_{g,i} - \hat{R}_g X_{g,i}}{\bar{X}_g}$$
The sample variance of $Z_{g,i}$ divided by $n_g$ satisfies:
$$\frac{s_{Z,g}^2}{n_g} = \frac{1}{n_g(n_g - 1)}\sum_{i=1}^{n_g} (Z_{g,i} - \bar{Z}_g)^2 \equiv \widehat{\operatorname{Var}}(\hat{R}_g)$$
Because arms are randomized independently, the variance of the difference $\hat{\Delta}_R = \hat{R}_T - \hat{R}_C$ is:
$$\widehat{\operatorname{Var}}(\hat{\Delta}_R) = \widehat{\operatorname{Var}}(\hat{R}_T) + \widehat{\operatorname{Var}}(\hat{R}_C)$$
$$SE(\hat{\Delta}_R) = \sqrt{\widehat{\operatorname{Var}}(\hat{\Delta}_R)}$$

Two-sided Wald $(1-\alpha)$ confidence interval:
$$\hat{\Delta}_R \pm z_{1-\alpha/2} \cdot SE(\hat{\Delta}_R)$$
Two-sided p-value:
$$p = 2 \cdot (1 - \Phi(|z|)) \quad \text{where } z = \frac{\hat{\Delta}_R}{SE(\hat{\Delta}_R)}$$

### 3.3 Bootstrap Sensitivity
ExPO provides a user-level resampled bootstrap (drawing $B$ bootstrap samples of user tuples $(Y_i, X_i)$ within each arm) to compute empirical percentile and studentized confidence intervals as a non-parametric sensitivity check.

---

## 4. Sample Ratio Mismatch (SRM) Test

### 4.1 Goodness-of-Fit Test
Let $K$ be the number of variants with declared planned allocation probabilities $p_1, \ldots, p_K$ ($\sum_{k=1}^K p_k = 1$).
Let $N = \sum_{k=1}^K O_k$ be total observed units, with $O_k$ observed in variant $k$.
Expected counts are $E_k = N \cdot p_k$.
Pearson's chi-square test statistic:
$$\chi^2 = \sum_{k=1}^K \frac{(O_k - E_k)^2}{E_k}$$
Under $H_0$, $\chi^2 \sim \chi^2(K - 1)$.
The p-value is:
$$p = 1 - F_{\chi^2, K-1}(\chi^2)$$

### 4.2 Edge Cases and Decision Rules
- **Rule:** If $\min_k E_k < 10$, chi-square asymptotic approximation may fail. ExPO flags small-sample caution and for $K=2$ evaluates the exact binomial test p-value:
  $$p_{\text{exact}} = 2 \cdot \sum_{x=0}^{\min(O_1, N - O_1)} \binom{N}{x} p_1^x (1-p_1)^{N-x}$$
- **Threshold:** Default alert threshold is $p < 0.001$.
- **Consequence:** If SRM is detected ($p < 0.001$), ExPO marks the experiment status as `SRM_FAILED / CAUTION` and warns that observed treatment differences cannot be causally interpreted without resolving allocation corruption.

---

## 5. Experiment Design: Power, Sample Size, and MDE

### 5.1 Continuous Outcomes
For two-sided hypothesis test $H_0: \Delta = 0$ at level $\alpha$, target power $1-\beta$, standard deviation $\sigma$, and allocation ratio $\kappa = n_T / n_C$:
$$n_C = \frac{(1 + 1/\kappa)\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}, \quad n_T = \kappa n_C$$
For equal allocation $\kappa = 1$:
$$n_{\text{per arm}} = \frac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$$
Minimum Detectable Effect (MDE) for fixed per-arm sample size $n$:
$$\delta_{MDE} = \sqrt{\frac{2\sigma^2}{n}} (z_{1-\alpha/2} + z_{1-\beta})$$

### 5.2 Binary Outcomes
For proportions $p_0$ (control) and $p_1 = p_0 + \delta$ (treatment), pooled proportion $\bar{p} = (p_0 + p_1) / 2$:
$$n_{\text{per arm}} = \frac{\left[z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta}\sqrt{p_0(1-p_0) + p_1(1-p_1)}\right]^2}{\delta^2}$$

### 5.3 CUPED Sample Size Projection
If pre-experiment covariate has expected correlation $\rho$ with outcome, the required sample size under CUPED is approximately:
$$n_{CUPED} \approx n \cdot (1 - \rho^2)$$
ExPO explicitly labels this a **projection based on historical correlation**.

---

## 6. Multiple Testing: Benjamini-Hochberg (BH) Procedure

### 6.1 Procedure
For a predeclared family of $M$ fixed-horizon hypothesis tests with sorted raw p-values:
$$p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$$
To control False Discovery Rate (FDR) at level $q$:
Find the largest index $k$ such that:
$$p_{(k)} \le \frac{k}{M} q$$
Reject all null hypotheses $H_{(i)}$ for $i \le k$.
The adjusted p-value (q-value) is computed monotonically as:
$$p^{BH}_{(i)} = \min_{j \ge i} \left\{ \min\left(1, \frac{M}{j} p_{(j)}\right) \right\}$$

### 6.2 Preconditions & Scope
- BH is applied to secondary and guardrail metrics measured at fixed horizons.
- **Prohibition:** BH MUST NOT be applied across sequential peeks of the same metric.

---

## 7. Mathematically Scoped Sequential Engine (Gaussian Normal-Mixture e-Process)

### 7.1 Model Specification
Let $D_1, D_2, \ldots, D_n \overset{iid}{\sim} \mathcal{N}(\delta, \sigma_D^2)$ be a sequence of independent increments formed from paired-by-index differences $D_i = Y_{T,i} - Y_{C,i}$.
- **Known Variance:** $\sigma_D^2 = \sigma_T^2 + \sigma_C^2$ is fixed *prior* to experiment monitoring from an independent pilot/historical dataset. It is NOT estimated or updated from accumulating experimental data.
- **Hypotheses:** We test point null $H_0: \delta = 0$ against the Gaussian mixture alternative:
  $$\delta \sim \mathcal{N}(0, \tau^2)$$
  where $\tau^2 > 0$ is a predeclared mixture variance.

### 7.2 Derivation of the Mixture Evidence Process (mSPRT / Bayes Factor)
Under $H_0$, the joint density of $D_1, \ldots, D_n$ is:
$$f_0(D_1, \ldots, D_n) = \left(2\pi \sigma_D^2\right)^{-n/2} \exp\left(-\frac{1}{2\sigma_D^2}\sum_{i=1}^n D_i^2\right)$$
Under $H_1(\delta)$, conditional on $\delta$:
$$f_1(D_1, \ldots, D_n \mid \delta) = \left(2\pi \sigma_D^2\right)^{-n/2} \exp\left(-\frac{1}{2\sigma_D^2}\sum_{i=1}^n (D_i - \delta)^2\right)$$
Noting that $\sum_{i=1}^n (D_i - \delta)^2 = \sum_{i=1}^n D_i^2 - 2n\delta \bar{D}_n + n\delta^2$, the conditional likelihood ratio is:
$$\Lambda_n(\delta) = \frac{f_1(D_1, \ldots, D_n \mid \delta)}{f_0(D_1, \ldots, D_n)} = \exp\left(\frac{n\delta \bar{D}_n}{\sigma_D^2} - \frac{n\delta^2}{2\sigma_D^2}\right)$$
Integrating over the prior $\delta \sim \mathcal{N}(0, \tau^2)$ with density $g(\delta) = \frac{1}{\sqrt{2\pi \tau^2}}\exp\left(-\frac{\delta^2}{2\tau^2}\right)$:
$$E_n = \int_{-\infty}^\infty \Lambda_n(\delta) g(\delta) d\delta = \frac{1}{\sqrt{2\pi \tau^2}} \int_{-\infty}^\infty \exp\left(-\frac{1}{2}\left[\delta^2\left(\frac{n}{\sigma_D^2} + \frac{1}{\tau^2}\right) - 2\delta \frac{n\bar{D}_n}{\sigma_D^2}\right]\right) d\delta$$
Let $A = \frac{n}{\sigma_D^2} + \frac{1}{\tau^2} = \frac{n\tau^2 + \sigma_D^2}{\sigma_D^2 \tau^2}$ and $B = \frac{n\bar{D}_n}{\sigma_D^2}$.
Completing the square in the exponent:
$$- \frac{1}{2}[A\delta^2 - 2B\delta] = -\frac{A}{2}\left(\delta - \frac{B}{A}\right)^2 + \frac{B^2}{2A}$$
Using the Gaussian integral $\int_{-\infty}^\infty \exp\left(-\frac{A}{2}(\delta - B/A)^2\right) d\delta = \sqrt{\frac{2\pi}{A}}$:
$$E_n = \frac{1}{\sqrt{2\pi \tau^2}} \sqrt{\frac{2\pi}{A}} \exp\left(\frac{B^2}{2A}\right) = \frac{1}{\sqrt{A\tau^2}} \exp\left(\frac{B^2}{2A}\right)$$
Evaluating $A\tau^2$:
$$A\tau^2 = \left(\frac{n\tau^2 + \sigma_D^2}{\sigma_D^2 \tau^2}\right)\tau^2 = \frac{\sigma_D^2 + n\tau^2}{\sigma_D^2} \implies \frac{1}{\sqrt{A\tau^2}} = \sqrt{\frac{\sigma_D^2}{\sigma_D^2 + n\tau^2}}$$
Evaluating $\frac{B^2}{2A}$:
$$\frac{B^2}{2A} = \frac{\left(\frac{n\bar{D}_n}{\sigma_D^2}\right)^2}{2 \left(\frac{n\tau^2 + \sigma_D^2}{\sigma_D^2 \tau^2}\right)} = \frac{n^2 \bar{D}_n^2}{\sigma_D^4} \cdot \frac{\sigma_D^2 \tau^2}{2(\sigma_D^2 + n\tau^2)} = \frac{n^2 \tau^2 \bar{D}_n^2}{2\sigma_D^2(\sigma_D^2 + n\tau^2)}$$

Therefore, the exact mixture evidence process is:
$$E_n = \sqrt{\frac{\sigma_D^2}{\sigma_D^2 + n\tau^2}} \exp\left(\frac{n^2 \tau^2 \bar{D}_n^2}{2\sigma_D^2(\sigma_D^2 + n\tau^2)}\right)$$
In log scale for numerical stability:
$$\log E_n = \frac{1}{2}\log\left(\frac{\sigma_D^2}{\sigma_D^2 + n\tau^2}\right) + \frac{n^2 \tau^2 \bar{D}_n^2}{2\sigma_D^2(\sigma_D^2 + n\tau^2)}$$

### 7.3 Ville's Inequality, Running Maximum, and Anytime-Valid p-Value
Under $H_0$, $(E_n)_{n \ge 1}$ is a nonnegative test martingale with $E[E_1] = 1$.
By Ville's maximal inequality for nonnegative supermartingales:
$$P\left(\sup_{n \ge 1} E_n \ge \frac{1}{\alpha}\right) \le \alpha$$
Let the running maximum evidence through look $n$ be:
$$M_n = \max_{1 \le k \le n} E_k, \qquad \log M_n = \max_{1 \le k \le n} \log E_k$$
The anytime-valid p-value is:
$$p_n^{AV} = \min\left(1, \frac{1}{M_n}\right) = \min\left(1, \exp(-\log M_n)\right)$$
- **Properties:**
  - $M_n$ is monotonically non-decreasing in $n$.
  - $p_n^{AV}$ is monotonically non-increasing in $n$.
  - $P_{H_0}(\exists n \ge 1: p_n^{AV} \le \alpha) = P_{H_0}\left(\sup_{n \ge 1} E_n \ge \frac{1}{\alpha}\right) \le \alpha$.

### 7.4 Time-Uniform Confidence Sequence via Mixture Evidence Inversion
To construct a $(1-\alpha)$ confidence sequence for $\delta$, we test candidate nulls $H_0: \delta = \delta_0$ using transformed differences $D_i - \delta_0$.
The sample mean becomes $\bar{D}_n - \delta_0$. The evidence process for $\delta_0$ is:
$$\log E_n(\delta_0) = \frac{1}{2}\log\left(\frac{\sigma_D^2}{\sigma_D^2 + n\tau^2}\right) + \frac{n^2 \tau^2 (\bar{D}_n - \delta_0)^2}{2\sigma_D^2(\sigma_D^2 + n\tau^2)}$$
The test does not reject $\delta_0$ at look $n$ when $\log E_n(\delta_0) \le \log(1/\alpha) = -\log \alpha$.
Solving for $(\bar{D}_n - \delta_0)^2$:
$$\frac{n^2 \tau^2 (\bar{D}_n - \delta_0)^2}{2\sigma_D^2(\sigma_D^2 + n\tau^2)} \le \log(1/\alpha) + \frac{1}{2}\log\left(1 + \frac{n\tau^2}{\sigma_D^2}\right)$$
$$(\bar{D}_n - \delta_0)^2 \le \frac{2\sigma_D^2(\sigma_D^2 + n\tau^2)}{n^2 \tau^2} \left[ \log(1/\alpha) + \frac{1}{2}\log\left(1 + \frac{n\tau^2}{\sigma_D^2}\right) \right]$$
Taking the square root gives radius $r_n$:
$$r_n = \sqrt{ \frac{2\sigma_D^2(\sigma_D^2 + n\tau^2)}{n^2 \tau^2} \left[ \log(1/\alpha) + \frac{1}{2}\log\left(1 + \frac{n\tau^2}{\sigma_D^2}\right) \right] }$$

The per-look interval is $[\bar{D}_n - r_n, \bar{D}_n + r_n]$.
The anytime-valid, time-uniform confidence sequence is the running intersection across all looks $1 \le k \le n$:
$$L_n = \max_{1 \le k \le n} (\bar{D}_k - r_k), \qquad U_n = \min_{1 \le k \le n} (\bar{D}_k + r_k)$$
By construction and Ville's inequality:
$$P\left(\forall n \ge 1, \, \delta \in [L_n, U_n]\right) \ge 1 - \alpha$$
