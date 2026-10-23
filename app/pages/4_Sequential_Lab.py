"""
ExPO — Streamlit Application: Sequential Lab (Page 5)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from expo_engine.data.synthetic import generate_effect_normal, generate_null_normal
from expo_engine.statistics.sequential import evaluate_sequential_monitoring

st.set_page_config(page_title="Sequential Lab — ExPO", page_icon="⏳", layout="wide")
st.title("⏳ Sequential Lab — Scoped Gaussian Mixture e-Process")

st.markdown("""
Monitor accumulating experiment data repeatedly **without Type I error inflation**
using a mathematically specified Gaussian normal-mixture evidence process (**mSPRT**).
""")

# Explain strict model assumptions
with st.expander("ℹ️ Mathematical Model & Eligibility Preconditions", expanded=True):
    st.markdown("""
    **Core Theoretical Requirements for Anytime Validity:**
    1. **Independent Gaussian Increments:** Outcomes in each look form independent differences $D_i = Y_{T,i} - Y_{C,i}$.
    2. **Frozen Known Variance:** $\sigma_D^2 = \sigma_T^2 + \sigma_C^2$ is fixed **before monitoring begins** from independent historical/pilot data. It is **never** estimated or updated from accumulating experiment data.
    3. **Predeclared Mixture Prior:** $\tau^2 > 0$ and stopping threshold $1/\alpha$ are fixed in advance.
    4. **Balanced Accrual:** Equal numbers of treatment and control observations are monitored per look.
    5. **Running Maximum Supremum:** Anytime-valid p-value is computed from the running supremum: $p_n^{AV} = \min(1, 1 / \max_{1 \le k \le n} E_k)$.
    """)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    scenario_type = st.selectbox("Stream Scenario:", ("Non-Zero Treatment Effect (delta = +1.5)", "Exact Null Effect (delta = 0.0)"))
with col_ctrl2:
    total_pairs = st.slider("Total Pairs to Monitored:", 200, 3000, 1000, step=100)
with col_ctrl3:
    look_freq = st.slider("Look Frequency (pairs per look):", 10, 100, 25, step=5)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    known_var = st.number_input("Frozen Pilot Variance (sigma_D^2):", value=200.0, min_value=1.0, step=10.0)
with col_s2:
    mixture_var = st.number_input("Predeclared Mixture Prior (tau^2):", value=1.0, min_value=0.1, step=0.1)
with col_s3:
    alpha = st.selectbox("Sequential Significance Level (alpha):", (0.05, 0.01, 0.10), index=0)

# Generate stream
seed = 42
delta_val = 1.5 if "Non-Zero" in scenario_type else 0.0
sigma_y = np.sqrt(known_var / 2.0)

rng = np.random.default_rng(seed)
c_stream = rng.normal(50.0, sigma_y, total_pairs)
t_stream = rng.normal(50.0 + delta_val, sigma_y, total_pairs)

res = evaluate_sequential_monitoring(
    treatment_stream=t_stream,
    control_stream=c_stream,
    known_variance=known_var,
    mixture_variance=mixture_var,
    alpha=alpha,
    look_frequency=look_freq,
)

st.markdown("---")
col_res1, col_res2, col_res3, col_res4 = st.columns(4)
with col_res1:
    st.metric("Total Looks", f"{res.total_looks_analyzed}")
with col_res2:
    st.metric("Early Stopping Status", "🚨 STOPPED EARLY" if res.stopped_early else "RUNNING (No Cross)")
with col_res3:
    st.metric("Final Anytime p-value", f"{res.final_anytime_p_value:.4e}")
with col_res4:
    st.metric("Max Evidence (M_n)", f"{res.final_running_max_evidence:.2f}", help=f"Stopping Threshold = {res.stopping_threshold:.1f}")

# Build trajectory dataframe
look_data = []
for look in res.looks_history:
    # Compute naive Welch p-value at this look for educational comparison
    se_naive = np.sqrt(known_var / look.sample_size_pairs)
    z_naive = np.abs(look.cumulative_mean_diff) / se_naive
    naive_p = 2.0 * (1.0 - stats.norm.cdf(z_naive))

    look_data.append({
        "Pairs Monitored": look.sample_size_pairs,
        "Mean Difference": look.cumulative_mean_diff,
        "CS Lower Bound": look.cs_lower,
        "CS Upper Bound": look.cs_upper,
        "Log Evidence": look.log_evidence,
        "Running Max Log Evidence": look.running_max_log_evidence,
        "Anytime Valid p-value": look.anytime_p_value,
        "Naive Peeking p-value": naive_p,
    })
df_looks = pd.DataFrame(look_data)

# Chart 1: Time-Uniform Confidence Sequence Envelope
st.subheader("1. Time-Uniform Confidence Sequence Envelope")
st.caption("Valid simultaneously across all continuous looks under Ville's inequality.")

fig_cs = go.Figure()
fig_cs.add_trace(go.Scatter(
    x=df_looks["Pairs Monitored"],
    y=df_looks["CS Upper Bound"],
    mode="lines",
    line=dict(color="rgba(16, 185, 129, 0.4)", width=1),
    name="CS Upper Bound",
))
fig_cs.add_trace(go.Scatter(
    x=df_looks["Pairs Monitored"],
    y=df_looks["CS Lower Bound"],
    mode="lines",
    line=dict(color="rgba(16, 185, 129, 0.4)", width=1),
    fill="tonexty",
    fillcolor="rgba(16, 185, 129, 0.15)",
    name="Time-Uniform 95% Confidence Sequence",
))
fig_cs.add_trace(go.Scatter(
    x=df_looks["Pairs Monitored"],
    y=df_looks["Mean Difference"],
    mode="lines+markers",
    line=dict(color="#10b981", width=2),
    name="Cumulative Mean Diff (Delta)",
))
fig_cs.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Null (Delta = 0)")
if delta_val > 0:
    fig_cs.add_hline(y=delta_val, line_dash="dot", line_color="#3b82f6", annotation_text=f"True Delta ({delta_val})")

fig_cs.update_layout(
    xaxis_title="Accumulated Sample Pairs (n)",
    yaxis_title="Estimated Treatment Effect",
    height=400,
)
st.plotly_chart(fig_cs, use_container_width=True)

# Chart 2: Anytime Valid p-value vs Naive Peeking p-value
st.subheader("2. Anytime-Valid p-value vs Naive Peeking p-value")
st.caption("Demonstrates why naive p-values are invalid under repeated peeking.")

fig_p = go.Figure()
fig_p.add_trace(go.Scatter(
    x=df_looks["Pairs Monitored"],
    y=df_looks["Anytime Valid p-value"],
    mode="lines+markers",
    line=dict(color="#8b5cf6", width=2),
    name="Anytime-Valid p-value (mSPRT)",
))
fig_p.add_trace(go.Scatter(
    x=df_looks["Pairs Monitored"],
    y=df_looks["Naive Peeking p-value"],
    mode="lines+markers",
    line=dict(color="#ef4444", width=1, dash="dash"),
    name="Naive Repeated Welch p-value (FLAWED)",
))
fig_p.add_hline(y=alpha, line_dash="dash", line_color="red", annotation_text=f"Stopping Threshold alpha = {alpha}")

fig_p.update_layout(
    xaxis_title="Accumulated Sample Pairs (n)",
    yaxis_title="p-value",
    yaxis=dict(range=[0, 1.05]),
    height=350,
)
st.plotly_chart(fig_p, use_container_width=True)

st.dataframe(df_looks, use_container_width=True, hide_index=True)
