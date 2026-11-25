"""
ExPO — Streamlit Application: Data Lab (Page 2)
"""

import io
import pandas as pd
import streamlit as st
import plotly.express as px

from expo_engine.checks.srm import check_srm
from expo_engine.checks.validation import validate_experiment_data
from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole, MetricSpec
from expo_engine.data.synthetic import (
    generate_conversion,
    generate_effect_normal,
    generate_null_heavy_tailed,
    generate_null_normal,
    generate_ratio_metric,
    generate_srm_injected,
)

st.set_page_config(page_title="Data Lab — ExPO", page_icon="📁", layout="wide")
st.title("📁 Data Lab — Ingestion, Validation & SRM Checks")

st.markdown("""
Inspect experiment datasets, validate schemas against canonical experimentation contracts,
and verify traffic integrity with Sample Ratio Mismatch (SRM) goodness-of-fit tests.
""")

# Source selection
data_source = st.radio(
    "Choose Data Source:",
    ("Use Existing / Demo Data", "Upload CSV / Parquet File", "Generate Synthetic Scenario"),
    horizontal=True,
)

df: pd.DataFrame = None

if data_source == "Use Existing / Demo Data":
    if "experiment_df" in st.session_state:
        df = st.session_state["experiment_df"]
        st.info("Loaded active experiment dataset from session state.")
    else:
        st.warning("No active experiment found. Generating default synthetic demo...")
        df, meta = generate_effect_normal(n_users=3000, seed=42, delta=2.0, rho=0.7)
        st.session_state["experiment_df"] = df
        st.session_state["experiment_meta"] = meta

elif data_source == "Upload CSV / Parquet File":
    uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_parquet(uploaded_file)
            st.session_state["experiment_df"] = df
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} rows).")
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif data_source == "Generate Synthetic Scenario":
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    with col_sc1:
        scenario = st.selectbox(
            "Scenario:",
            ("effect_normal", "null_normal", "conversion", "ratio_metric", "srm_injected", "null_heavy_tailed")
        )
    with col_sc2:
        n_users = st.slider("User Count:", 500, 20000, 3000, step=500)
    with col_sc3:
        seed = st.number_input("Random Seed:", value=42, step=1)

    if st.button("Generate Scenario", type="primary"):
        if scenario == "effect_normal":
            df, meta = generate_effect_normal(n_users=n_users, seed=seed, delta=2.0, rho=0.7)
        elif scenario == "null_normal":
            df, meta = generate_null_normal(n_users=n_users, seed=seed, rho=0.6)
        elif scenario == "conversion":
            df, meta = generate_conversion(n_users=n_users, seed=seed, p_control=0.10, delta=0.02)
        elif scenario == "ratio_metric":
            df, meta = generate_ratio_metric(n_users=n_users, seed=seed)
        elif scenario == "srm_injected":
            df, meta = generate_srm_injected(n_users=n_users, seed=seed, realized_treatment_prob=0.54)
        else:
            df, meta = generate_null_heavy_tailed(n_users=n_users, seed=seed)

        st.session_state["experiment_df"] = df
        st.session_state["experiment_meta"] = meta
        st.success(f"Generated scenario '{scenario}' ({len(df):,} rows).")

if df is not None:
    st.markdown("---")
    st.subheader("1. Data Preview & Schema Inspection")
    st.dataframe(df.head(6), use_container_width=True)

    # Column Mapping Section
    st.markdown("### 2. Configure Experiment Schema")
    all_cols = list(df.columns)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        user_col = st.selectbox("User ID Column:", all_cols, index=all_cols.index("user_id") if "user_id" in all_cols else 0)
    with col_m2:
        variant_col = st.selectbox("Variant Column:", all_cols, index=all_cols.index("variant") if "variant" in all_cols else 1)

    observed_variants = list(df[variant_col].dropna().unique())
    with col_m3:
        control_var = st.selectbox("Control Variant Label:", observed_variants, index=0 if len(observed_variants) > 0 else 0)
    with col_m4:
        treatment_var = st.selectbox("Treatment Variant Label:", observed_variants, index=1 if len(observed_variants) > 1 else 0)

    # Primary Metric Mapping
    st.markdown("### 3. Metric Mapping")
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        metric_kind_str = st.selectbox(
            "Primary Metric Kind:",
            ("continuous_mean", "binary_mean", "ratio_of_sums"),
        )

    metric_kind = MetricKind(metric_kind_str)
    num_col, denom_col, outcome_col, pre_col = None, None, None, None

    if metric_kind == MetricKind.RATIO_OF_SUMS:
        with col_k2:
            num_col = st.selectbox("Numerator Column:", all_cols, index=all_cols.index("numerator") if "numerator" in all_cols else 0)
        with col_k3:
            denom_col = st.selectbox("Denominator Column:", all_cols, index=all_cols.index("denominator") if "denominator" in all_cols else 1)
        metric_spec = MetricSpec(
            name=f"{num_col}/{denom_col}",
            label="Ratio of Sums",
            kind=MetricKind.RATIO_OF_SUMS,
            role=MetricRole.PRIMARY,
            numerator_column=num_col,
            denominator_column=denom_col,
        )
    else:
        with col_k2:
            default_idx = all_cols.index("metric_value") if "metric_value" in all_cols else (all_cols.index("conversion") if "conversion" in all_cols else 0)
            outcome_col = st.selectbox("Outcome Metric Column:", all_cols, index=default_idx)
        with col_k3:
            pre_options = ["None"] + all_cols
            default_pre = pre_options.index("pre_metric") if "pre_metric" in pre_options else 0
            pre_sel = st.selectbox("Pre-covariate (for CUPED):", pre_options, index=default_pre)
            pre_col = pre_sel if pre_sel != "None" else None

        metric_spec = MetricSpec(
            name=outcome_col,
            label=outcome_col.replace("_", " ").title(),
            kind=metric_kind,
            role=MetricRole.PRIMARY,
            column=outcome_col,
            pre_covariate=pre_col,
        )

    # Build Configuration Object
    config = ExperimentConfig(
        experiment_id="exp_data_lab",
        user_id_col=user_col,
        variant_col=variant_col,
        control_variant=control_var,
        treatment_variant=treatment_var,
        planned_ratio={control_var: 0.5, treatment_var: 0.5},
        metrics=[metric_spec],
    )
    st.session_state["experiment_config"] = config

    # Run Validation & SRM
    st.markdown("---")
    st.subheader("4. Automated Validation & Sample Ratio Mismatch (SRM) Audit")

    val_res = validate_experiment_data(df, config)
    srm_res = check_srm(
        df=df,
        variant_col=variant_col,
        planned_ratio=config.planned_ratio,
        threshold=config.srm_threshold,
    )
    st.session_state["validation_result"] = val_res
    st.session_state["srm_result"] = srm_res

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Rows", f"{val_res.total_rows:,}")
    with col_stat2:
        st.metric("Unique Users", f"{val_res.unique_users:,}")
    with col_stat3:
        st.metric("Control Units", f"{val_res.variant_counts.get(control_var, 0):,}")
    with col_stat4:
        st.metric("Treatment Units", f"{val_res.variant_counts.get(treatment_var, 0):,}")

    # SRM Display
    if srm_res.status == "SRM_DETECTED":
        st.error(f"🚨 **{srm_res.message}**")
        st.markdown(f"- **Test Statistic:** {srm_res.statistic:.2f} | **p-value:** `{srm_res.p_value:.2e}` (Threshold: `{srm_res.threshold}`)")
    elif srm_res.status == "WARNING":
        st.warning(f"⚠️ **{srm_res.message}**")
    else:
        st.success(f"✅ **{srm_res.message}** (p={srm_res.p_value:.4f})")

    # SRM Distribution Chart
    srm_df = pd.DataFrame({
        "Variant": list(srm_res.observed_counts.keys()),
        "Observed": list(srm_res.observed_counts.values()),
        "Expected": [int(srm_res.expected_counts[v]) for v in srm_res.observed_counts.keys()],
    })
    fig_srm = px.bar(
        srm_df,
        x="Variant",
        y=["Observed", "Expected"],
        barmode="group",
        title="Sample Allocation: Observed vs Expected",
        color_discrete_map={"Observed": "#3b82f6", "Expected": "#9ca3af"},
    )
    st.plotly_chart(fig_srm, use_container_width=True)

    if not val_res.is_valid:
        st.error("Validation Errors Encountered:")
        for err in val_res.errors:
            st.write(f"- ❌ {err}")
    if val_res.warnings:
        with st.expander("Data Quality Warnings", expanded=True):
            for w in val_res.warnings:
                st.write(f"- ⚠️ {w}")
