"""
ExPO — Streamlit Application: Home / Project Overview (Page 1)
"""

import streamlit as st
from expo_engine import __version__

st.set_page_config(
    page_title="ExPO — Experimentation Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ ExPO — Experimentation & Product Outcomes Engine")
st.markdown(f"**Version {__version__}** | *Trustworthy A/B Testing, CUPED/ANCOVA, Ratio Delta Method & Scoped Sequential Inference*")

st.markdown("""
---
### Welcome to ExPO
ExPO is a local-first experimentation platform designed to evaluate A/B tests with strict statistical honesty.
It moves beyond bare p-values to provide auditable causal inference, automated data-quality checks,
pre-experiment variance reduction, and mathematically grounded sequential monitoring.

```
       ┌────────────────┐       ┌────────────────┐       ┌─────────────────┐
       │   1. Data Lab   │  ──>  │  2. Designer   │  ──>  │   3. Analyze    │
       │ Schema & SRM   │       │ Power & MDE    │       │ Welch & CUPED   │
       └────────────────┘       └────────────────┘       └─────────────────┘
                │                                                  │
                ▼                                                  ▼
       ┌────────────────┐       ┌────────────────┐       ┌─────────────────┐
       │4. Sequential   │  <──  │  5. Sim Lab    │  <──  │  6. Reports     │
       │ mSPRT Evidence │       │ Monte Carlo    │       │ HTML/JSON/CSV   │
       └────────────────┘       └────────────────┘       └─────────────────┘
```
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 🔬 1. Data Integrity & SRM")
    st.markdown("""
    - Automated detection of duplicate analysis units and variant contamination.
    - Pearson chi-square and exact binomial tests for **Sample Ratio Mismatch (SRM)**.
    - Exposure vs assignment level diagnostics.
    """)
    if st.button("Go to Data Lab", key="nav_data"):
        st.switch_page("pages/1_Data_Lab.py")

with col2:
    st.success("### ⚡ 2. Fixed-Horizon & CUPED")
    st.markdown("""
    - **Welch's t-test** with exact Welch-Satterthwaite degrees of freedom.
    - **CUPED / ANCOVA**: optimal pre-experiment linear adjustment with heteroscedasticity-robust SE (HC1).
    - **Ratio Metrics**: user-level ratio-of-sums via bivariate Delta method.
    """)
    if st.button("Go to Analyze", key="nav_analyze"):
        st.switch_page("pages/3_Analyze.py")

with col3:
    st.warning("### ⏳ 3. Scoped Sequential Engine")
    st.markdown("""
    - Gaussian normal-mixture evidence process (**mSPRT**) with known frozen variance.
    - Monotonic running-maximum evidence and anytime-valid p-values.
    - Time-uniform confidence sequences with Ville's inequality guarantees.
    """)
    if st.button("Go to Sequential Lab", key="nav_seq"):
        st.switch_page("pages/4_Sequential_Lab.py")

st.markdown("---")
st.markdown("### Quick Demo Setup")
st.markdown("Want to explore ExPO immediately? Click below to generate a pre-configured synthetic A/B experiment with known ground truth.")

if st.button("🚀 Load Canonical Demo Experiment", type="primary"):
    from expo_engine.data.synthetic import generate_effect_normal
    df, meta = generate_effect_normal(
        n_users=3000,
        seed=42,
        delta=2.0,
        mu_y=50.0,
        sigma_y=10.0,
        rho=0.7,
        missing_covariate_rate=0.03,
    )
    df["page_views"] = df["metric_value"] * 0.4 + 5.0
    st.session_state["experiment_df"] = df
    st.session_state["experiment_meta"] = meta
    st.success(f"Generated synthetic experiment ({len(df):,} users, true Delta=2.0, pre-covariate rho=0.7). Navigate to 'Data Lab' or 'Analyze' to inspect!")
