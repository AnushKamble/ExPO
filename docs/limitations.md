# ExPO — Statistical Limitations & Boundary Conditions

This document records the exact mathematical assumptions, scope boundaries, and statistical limitations of ExPO.
ExPO maintains statistical honesty by explicitly disclosing where asymptotic approximations, model assumptions, or data violations degrade inferential validity.

---

## 1. Randomization & Causal Identification
- **Assumption:** SUTVA (Stable Unit Treatment Value Assumption) and proper unconfounded assignment. Units must not interfere with each other (e.g. network effects, two-sided marketplace cannibalization).
- **Limitation:** ExPO does not adjust for network interference or cluster randomization unless data is aggregated to the cluster level before input. Statistical significance does not repair broken randomization, tracking defects, or survivorship bias.

---

## 2. Sample Ratio Mismatch (SRM)
- **Interpretation:** SRM detects statistical evidence that observed traffic allocations differ from declared planned probabilities.
- **Limitation:** An SRM alert identifies the existence of an integrity issue, not its root cause (e.g. redirect bot filtering, variant crash, caching, or hash collisions). When SRM is triggered ($p < 0.001$), causal treatment effects must not be interpreted as unbiased.

---

## 3. CUPED / Covariate Adjustment
- **Assumption:** Pre-experiment covariates $X_i$ must be measured strictly *before* assignment/exposure and cannot be affected by the treatment.
- **Limitation:** 
  - CUPED variance reduction is not guaranteed to achieve arbitrary percentages (e.g. 50%); realized reduction depends strictly on the empirical correlation $\rho$ according to $1 - \rho^2$.
  - Post-treatment covariates must never be used for CUPED; doing so causes collider bias and severe estimation error.
  - Complete-case adjustment assumes covariate missingness is completely at random (MCAR). Imbalanced missingness triggers differential missingness warnings.

---

## 4. Ratio Metrics & Delta Method
- **Assumption:** The randomization unit is the user. Bivariate Delta method standard errors rely on Central Limit Theorem (CLT) asymptotics for user sums.
- **Limitation:**
  - Delta method coverage may degrade if the user denominator distribution has extreme heavy tails, high sparsity ($X_i = 0$ for most users), or low sample sizes ($n < 100$).
  - ExPO provides non-parametric user bootstrap as a sensitivity comparator, but bootstrap cannot overcome structural zero-denominator issues.

---

## 5. Sequential Monitoring (mSPRT Evidence Process)
- **Assumption:** 
  1. Increments $D_i = Y_{T,i} - Y_{C,i}$ are independent and Gaussian.
  2. The difference variance $\sigma_D^2$ is **fixed before experiment launch** from historical or pilot data.
  3. Predeclared mixture prior $\tau^2 > 0$ and stopping threshold $1/\alpha$.
- **Strict Prohibition / Boundary:**
  - The sequential engine **strictly blocks** non-normal heavy-tailed data or ratio metrics without explicit derivations.
  - Estimating or continuously updating $\sigma_D^2$ from the accumulating experimental data violates the martingale property and voids exact anytime validity theorems.
  - Applying fixed-horizon multiplicity corrections (like Benjamini-Hochberg) across repeated looks is mathematically invalid and strictly prohibited in ExPO.

---

## 6. Multiple Testing (Benjamini-Hochberg)
- **Assumption:** PRDS (Positive Regression Dependency on Subsets) or independence among hypotheses in a declared fixed-horizon family.
- **Limitation:** Controls FDR across secondary/guardrail metrics measured at a single fixed horizon. Adjusted p-values represent FDR control thresholds, not Bayesian posterior probabilities of individual hypotheses.
