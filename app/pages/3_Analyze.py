"""
ExPO — Streamlit Application: Analyze (Page 4)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole, MetricSpec
from expo_engine.data.synthetic import generate_effect_normal
from expo_engine.services.analysis_service import ExperimentAnalysisService

st.set_page_config(page_title="Analyze — ExPO", page_icon="📊", layout="wide")
st.title("📊 Analyze — Treatment Effects, CUPED & Multiplicity")

if "experiment_df" not in st.session_state or "experiment_config" not in st.session_state:
    st.warning("No experiment loaded. Initializing default synthetic experiment...")
    df, meta = generate_effect_normal(
        n_users=3000, seed=42, delta=2.0, mu_y=50.0, sigma_y=10.0, rho=0.7, missing_covariate_rate=0.03
    )
    df["latency"] = df["metric_value"] * 0.5 + 20.0
    st.session_state["experiment_df"] = df
    st.session_state["experiment_meta"] = meta
    st.session_state["experiment_config"] = ExperimentConfig(
        experiment_id="exp_dashboard_demo",
        user_id_col="user_id",
        variant_col="variant",
        control_variant="control",
        treatment_variant="treatment",
        metrics=[
            MetricSpec(
                name="revenue",
                label="Average Revenue Per User",
                kind=MetricKind.CONTINUOUS_MEAN,
                role=MetricRole.PRIMARY,
                column="metric_value",
                pre_covariate="pre_metric",
            ),
            MetricSpec(
                name="latency",
                label="Page Latency",
                kind=MetricKind.CONTINUOUS_MEAN,
                role=MetricRole.SECONDARY,
                column="latency",
            ),
        ],
    )

df = st.session_state["experiment_df"]
config = st.session_state["experiment_config"]

# Execution
service = ExperimentAnalysisService(config)
try:
    manifest = service.run_full_analysis(df, run_id="ui_analysis_run")
    st.session_state["manifest"] = manifest
except Exception as e:
    st.error(f"Analysis failed: {e}")
    st.stop()

# Data Quality Status Banner
dq_status = manifest.data_quality_status
if "ALERT" in dq_status:
    st.error(f"🚨 **Data Quality Alert:** `{dq_status}` — Severe assignment imbalance detected. Causal claims are unsafe.")
elif "WARNING" in dq_status:
    st.warning(f"⚠️ **Data Quality Warnings Present:** `{dq_status}` — Inspect caveats before interpreting outcomes.")
else:
    st.success("✅ **Data Quality Audit Passed:** Traffic allocation and data contracts verified clean.")

st.markdown("---")
st.subheader("1. Primary Metric Treatment Effect & Covariate Adjustment")

w_res = manifest.primary_metric_result
c_res = manifest.cuped_result

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    raw_ate = w_res.get("absolute_effect") or w_res.get("delta_ratio")
    st.metric("Raw ATE (Unadjusted)", f"{raw_ate:.4f}")
with col_p2:
    st.metric("Raw Standard Error", f"{w_res.get('standard_error'):.4f}")
with col_p3:
    if c_res:
        st.metric("CUPED Adjusted ATE", f"{c_res['adjusted_ate']:.4f}", delta=f"{c_res['adjusted_ate'] - raw_ate:.4f}")
    else:
        st.metric("CUPED Adjusted ATE", "N/A")
with col_p4:
    if c_res:
        st.metric("Robust Adjusted SE", f"{c_res['adjusted_se']:.4f}", delta=f"-{(1.0 - c_res['adjusted_se']/w_res['standard_error']):.1%}")
    else:
        st.metric("Robust Adjusted SE", "N/A")

# Forest Plot Comparison
st.markdown("### Confidence Interval Comparison (Raw vs CUPED Adjusted)")
fig_forest = go.Figure()

# Raw CI
fig_forest.add_trace(go.Scatter(
    x=[w_res["absolute_effect"]],
    y=["1. Raw Welch (Unadjusted)"],
    error_x=dict(
        type="data",
        symmetric=False,
        array=[w_res["ci_upper"] - w_res["absolute_effect"]],
        arrayminus=[w_res["absolute_effect"] - w_res["ci_lower"]],
        width=10,
        thickness=2,
    ),
    mode="markers",
    marker=dict(size=12, color="#3b82f6"),
    name="Raw Welch 95% CI",
))

# CUPED CI if available
if c_res:
    fig_forest.add_trace(go.Scatter(
        x=[c_res["adjusted_ate"]],
        y=["2. CUPED / ANCOVA Adjusted"],
        error_x=dict(
            type="data",
            symmetric=False,
            array=[c_res["adjusted_ci_upper"] - c_res["adjusted_ate"]],
            arrayminus=[c_res["adjusted_ate"] - c_res["adjusted_ci_lower"]],
            width=10,
            thickness=2,
        ),
        mode="markers",
        marker=dict(size=12, color="#10b981"),
        name="CUPED Robust 95% CI",
    ))

fig_forest.add_vline(x=0.0, line_dash="dash", line_color="gray", annotation_text="Null (Delta = 0)")
fig_forest.update_layout(
    xaxis_title="Treatment Effect (Delta)",
    yaxis_title="",
    height=250,
    margin=dict(l=20, r=20, t=30, b=30),
)
st.plotly_chart(fig_forest, use_container_width=True)

if c_res:
    st.info(
        f"🎯 **CUPED Variance Reduction Summary:** "
        f"Pre-covariate `{c_res['covariate_name']}` achieved sample correlation **rho = {c_res['sample_correlation']:.3f}**. "
        f"Realized outcome variance reduction: **{c_res['empirical_variance_reduction']:.1%}** (Theoretical expectation 1 - rho^2 = {c_res['theoretical_variance_reduction']:.1%}). "
        f"Analysis performed on **{c_res['complete_cases_count']:,}** complete cases ({c_res['missing_covariate_count']} units missing pre-covariate retained in unadjusted baseline)."
    )

# Descriptive Summaries Table
if "control_summary" in w_res and "treatment_summary" in w_res:
    st.markdown("### Descriptive Group Statistics")
    cs = w_res["control_summary"]
    ts = w_res["treatment_summary"]
    desc_df = pd.DataFrame([
        {"Variant": "Control", "N": cs["sample_size"], "Mean": f"{cs['mean']:.4f}", "Variance": f"{cs['variance']:.4f}", "Std Dev": f"{cs['std_dev']:.4f}", "Median": f"{cs['median']:.4f}", "IQR": f"{cs['iqr']:.4f}"},
        {"Variant": "Treatment", "N": ts["sample_size"], "Mean": f"{ts['mean']:.4f}", "Variance": f"{ts['variance']:.4f}", "Std Dev": f"{ts['std_dev']:.4f}", "Median": f"{ts['median']:.4f}", "IQR": f"{ts['iqr']:.4f}"},
    ])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

# Secondary Multiple Testing
if manifest.secondary_multiplicity and manifest.secondary_multiplicity.hypotheses:
    st.markdown("---")
    st.subheader("2. Secondary & Guardrail Hypotheses (Benjamini-Hochberg FDR Control)")
    st.caption(f"Controlled at False Discovery Rate q = {manifest.secondary_multiplicity.fdr_q:.2f} across declared fixed-horizon family.")

    sec_rows = []
    for hyp in manifest.secondary_multiplicity.hypotheses:
        sec_rows.append({
            "Metric": hyp.metric_name,
            "Raw p-value": f"{hyp.raw_p_value:.4f}",
            "BH Adjusted p-value (q-val)": f"{hyp.adjusted_p_value:.4f}",
            "Statistically Significant": "✅ YES" if hyp.rejected else "❌ NO",
        })
    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

# Audit Warnings
if manifest.warnings:
    with st.expander("Analysis Audit Warnings & Precondition Notes", expanded=True):
        for w in manifest.warnings:
            st.write(f"- ⚠️ {w}")
