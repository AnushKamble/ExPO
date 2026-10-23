"""
ExPO — Streamlit Application: Simulation Lab (Page 6)
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from expo_engine.simulation.runners import (
    run_aa_simulation,
    run_cuped_simulation,
    run_peeking_simulation,
    run_ratio_simulation,
    run_srm_simulation,
)

st.set_page_config(page_title="Simulation Lab — ExPO", page_icon="🧪", layout="wide")
st.title("🧪 Simulation Lab — Monte Carlo Validation Suite")

st.markdown("""
ExPO validates all analytical and sequential methods through reproducible Monte Carlo simulations
with **Wilson score confidence intervals** and **Monte Carlo Standard Errors (MCSE)**.
""")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    study_choice = st.selectbox(
        "Select Research Study:",
        (
            "Study A: Peeking Inflation vs Sequential Evidence",
            "Study B: CUPED Variance Reduction & CI Shrinkage",
            "Study C: A/A Calibration & P-Value Uniformity",
            "Study D: Ratio Delta Method Coverage",
            "Study E: SRM Detection Power",
        )
    )
with col_ctrl2:
    preset = st.selectbox("Preset Tier:", ("smoke (Fast: ~2-5s)", "full (Research: ~15-30s)"))
    preset_tier = "smoke" if "smoke" in preset else "full"
with col_ctrl3:
    seed = st.number_input("Random Seed:", value=42, step=1)

run_button = st.button("🚀 Run Monte Carlo Simulation", type="primary")

if run_button:
    st.markdown("---")
    with st.spinner(f"Running {study_choice} ({preset_tier} preset)..."):
        if "Study A" in study_choice:
            res = run_peeking_simulation(preset=preset_tier, seed=seed)
            st.session_state["sim_result"] = res
            st.session_state["sim_type"] = "peeking"

        elif "Study B" in study_choice:
            res = run_cuped_simulation(preset=preset_tier, seed=seed)
            st.session_state["sim_result"] = res
            st.session_state["sim_type"] = "cuped"

        elif "Study C" in study_choice:
            res = run_aa_simulation(preset=preset_tier, seed=seed)
            st.session_state["sim_result"] = res
            st.session_state["sim_type"] = "aa"

        elif "Study D" in study_choice:
            res = run_ratio_simulation(preset=preset_tier, seed=seed)
            st.session_state["sim_result"] = res
            st.session_state["sim_type"] = "ratio"

        elif "Study E" in study_choice:
            res = run_srm_simulation(preset=preset_tier, seed=seed)
            st.session_state["sim_result"] = res
            st.session_state["sim_type"] = "srm"

if "sim_result" in st.session_state:
    res = st.session_state["sim_result"]
    sim_type = st.session_state["sim_type"]

    st.subheader(f"Results: {study_choice}")

    # Study A Display
    if sim_type == "peeking":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fixed-Horizon Welch FPR", f"{res['fixed_horizon']['rejection_rate']:.2%}")
        with col2:
            st.metric("Naive Daily Peeking FPR", f"{res['naive_daily_peeking']['rejection_rate']:.2%}", delta="SEVERE INFLATION", delta_color="inverse")
        with col3:
            st.metric("Sequential mSPRT FPR", f"{res['gaussian_mixture_sequential']['rejection_rate']:.2%}", delta="CONTROLLED", delta_color="normal")

        # Bar chart with Wilson CI error bars
        plot_df = pd.DataFrame([
            {
                "Method": "Fixed-Horizon Welch",
                "FPR": res["fixed_horizon"]["rejection_rate"],
                "CI_Lower": res["fixed_horizon"]["wilson_ci_lower"],
                "CI_Upper": res["fixed_horizon"]["wilson_ci_upper"],
            },
            {
                "Method": "Naive Daily Peeking",
                "FPR": res["naive_daily_peeking"]["rejection_rate"],
                "CI_Lower": res["naive_daily_peeking"]["wilson_ci_lower"],
                "CI_Upper": res["naive_daily_peeking"]["wilson_ci_upper"],
            },
            {
                "Method": "Gaussian Mixture Sequential",
                "FPR": res["gaussian_mixture_sequential"]["rejection_rate"],
                "CI_Lower": res["gaussian_mixture_sequential"]["wilson_ci_lower"],
                "CI_Upper": res["gaussian_mixture_sequential"]["wilson_ci_upper"],
            },
        ])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df["Method"],
            y=plot_df["FPR"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=plot_df["CI_Upper"] - plot_df["FPR"],
                arrayminus=plot_df["FPR"] - plot_df["CI_Lower"],
            ),
            marker_color=["#3b82f6", "#ef4444", "#10b981"],
        ))
        fig.add_hline(y=0.05, line_dash="dash", line_color="black", annotation_text="Nominal Alpha = 5%")
        fig.update_layout(title="False Positive Rate across Peeking Protocols (with 95% Wilson CIs)", yaxis_title="Empirical Type I Error")
        st.plotly_chart(fig, use_container_width=True)

    # Study B Display
    elif sim_type == "cuped":
        grid = pd.DataFrame(res["grid_results"])
        st.dataframe(grid, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grid["target_rho"], y=grid["theoretical_variance_reduction"],
            mode="lines+markers", name="Theoretical (1 - rho^2)", line=dict(color="#3b82f6", dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=grid["target_rho"], y=grid["empirical_variance_reduction"],
            mode="lines+markers", name="Empirical Realized Reduction", line=dict(color="#10b981", width=3)
        ))
        fig.update_layout(
            title="CUPED Variance Reduction: Theoretical vs Empirical",
            xaxis_title="Covariate Correlation (rho)",
            yaxis_title="Variance Reduction Ratio",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Study C Display
    elif sim_type == "aa":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Observed A/A Type I Error", f"{res['observed_type_1_error']:.2%}")
        with col2:
            st.metric("KS Uniformity p-value", f"{res['ks_uniformity_p_value']:.4f}")

        # Decile Histogram
        dec_df = pd.DataFrame({
            "Decile Bin": [f"[{i/10:.1f}, {(i+1)/10:.1f})" for i in range(10)],
            "Counts": res["p_value_decile_counts"],
            "Expected": res["expected_count_per_decile"],
        })
        fig = px.bar(dec_df, x="Decile Bin", y="Counts", title="A/A P-Value Distribution across Deciles (Should be Flat)")
        fig.add_hline(y=res["expected_count_per_decile"], line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    # Study D Display
    elif sim_type == "ratio":
        st.metric("Delta Method 95% CI Coverage", f"{res['empirical_coverage']:.2%}")
        st.info(f"95% Wilson Score Confidence Interval: [{res['wilson_ci_lower']:.2%}, {res['wilson_ci_upper']:.2%}] (MCSE: {res['mcse']:.4f})")

    # Study E Display
    elif sim_type == "srm":
        srm_df = pd.DataFrame(res["matrix_results"])
        st.dataframe(srm_df, use_container_width=True)
        fig = px.line(
            srm_df,
            x="realized_treatment_prob",
            y="detection_power",
            color="sample_size",
            markers=True,
            title="SRM Detection Power vs Imbalance Magnitude & Sample Size",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Download raw JSON
    st.download_button(
        "📥 Download Simulation Output (JSON)",
        data=json.dumps(res, indent=2),
        file_name=f"simulation_{sim_type}.json",
        mime="application/json",
    )
