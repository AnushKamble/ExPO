"""
ExPO — Streamlit Application: Experiment Designer (Page 3)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from expo_engine.statistics.power import (
    compute_binary_sample_size,
    compute_continuous_sample_size,
)

st.set_page_config(page_title="Experiment Designer — ExPO", page_icon="📐", layout="wide")
st.title("📐 Experiment Designer — Power, Sample Size & MDE Planning")

st.markdown("""
Plan experiment sample sizes, detectability thresholds (MDE), and duration requirements
using standard normal approximations and CUPED variance reduction projections.
""")

col_opt1, col_opt2 = st.columns([1, 2])

with col_opt1:
    st.subheader("1. Parameters & Hypothesis")
    metric_kind = st.selectbox("Primary Metric Kind:", ("Continuous Metric", "Binary Conversion Rate"))

    alpha = st.selectbox("Significance Level (alpha):", (0.05, 0.01, 0.10), index=0)
    power = st.selectbox("Target Power (1 - beta):", (0.80, 0.90, 0.95), index=0)
    alloc_ratio = st.number_input("Allocation Ratio (Treatment / Control):", value=1.0, min_value=0.1, max_value=10.0, step=0.1)

    if metric_kind == "Continuous Metric":
        baseline_var = st.number_input("Baseline Variance (sigma^2):", value=100.0, min_value=0.1, step=5.0)
        baseline_mean = st.number_input("Baseline Mean (optional):", value=50.0, step=5.0)
        mde_abs = st.number_input("Minimum Detectable Effect (MDE Delta):", value=1.5, min_value=0.01, step=0.1)
        cuped_rho = st.slider("Anticipated Pre-Covariate Correlation (rho):", 0.0, 0.95, 0.60, step=0.05)
    else:
        baseline_p = st.slider("Baseline Conversion Rate (p0):", 0.01, 0.90, 0.10, step=0.01)
        mde_abs = st.number_input("Absolute Detectable Lift (percentage points):", value=0.015, min_value=0.001, max_value=0.20, step=0.005, format="%.3f")
        cuped_rho = None

    daily_traffic = st.number_input("Expected Daily Traffic (users/day):", value=1000, min_value=10, step=100)

with col_opt2:
    st.subheader("2. Sample Size & Duration Projection")

    if metric_kind == "Continuous Metric":
        res = compute_continuous_sample_size(
            baseline_variance=baseline_var,
            mde_absolute=mde_abs,
            baseline_mean=baseline_mean,
            alpha=alpha,
            power=power,
            allocation_ratio=alloc_ratio,
            cuped_rho=cuped_rho,
            daily_sample_rate=daily_traffic,
        )
    else:
        res = compute_binary_sample_size(
            baseline_rate=baseline_p,
            mde_absolute=mde_abs,
            alpha=alpha,
            power=power,
            allocation_ratio=alloc_ratio,
            daily_sample_rate=daily_traffic,
        )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Control Sample", f"{res.control_sample_size:,}")
    with col_m2:
        st.metric("Treatment Sample", f"{res.treatment_sample_size:,}")
    with col_m3:
        st.metric("Total Sample Size", f"{res.total_sample_size:,}")
    with col_m4:
        st.metric("Estimated Duration", f"{res.estimated_duration_days} days" if res.estimated_duration_days else "N/A")

    if res.cuped_projected_total:
        savings = (1.0 - res.cuped_projected_total / res.total_sample_size)
        st.info(
            f"💡 **CUPED Projected Sample Size:** **{res.cuped_projected_total:,}** users "
            f"({savings:.1%} sample size reduction, assuming pre-experiment correlation rho={res.cuped_correlation_assumed}). "
            f"Projected experiment duration: **{int(np.ceil(res.cuped_projected_total / daily_traffic))} days**."
        )

    st.markdown("---")
    st.subheader("3. Power vs Sample Size Sensitivity Curves")

    # Interactive sensitivity curve
    if metric_kind == "Continuous Metric":
        mde_candidates = [mde_abs * 0.75, mde_abs, mde_abs * 1.25]
        sample_sizes = np.linspace(max(100, res.total_sample_size // 4), res.total_sample_size * 2, 40)
        
        fig = go.Figure()
        for cand_mde in mde_candidates:
            powers = []
            z_a = 1.95996
            sigma = np.sqrt(baseline_var)
            for n_tot in sample_sizes:
                n_arm = n_tot / 2.0
                se = np.sqrt(2.0 * baseline_var / n_arm)
                z_b = (cand_mde / se) - z_a
                p_val = stats.norm.cdf(z_b)
                powers.append(p_val)
            fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode="lines", name=f"MDE = {cand_mde:.2f}"))

        fig.add_hline(y=power, line_dash="dash", line_color="red", annotation_text=f"Target Power ({power:.0%})")
        fig.update_layout(
            title="Statistical Power as a Function of Total Sample Size",
            xaxis_title="Total Experiment Sample Size (Users)",
            yaxis_title="Power (1 - beta)",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(res.method_notes)
