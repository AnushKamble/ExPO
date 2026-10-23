"""
ExPO — Streamlit Application: Report & Reproducibility (Page 7)
"""

import io
from pathlib import Path
import pandas as pd
import streamlit as st

from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole, MetricSpec
from expo_engine.data.synthetic import generate_effect_normal
from expo_engine.reporting.exporter import (
    export_report_csv,
    export_report_html,
    export_report_json,
    export_report_markdown,
)
from expo_engine.reporting.schemas import ExperimentReportManifest
from expo_engine.services.analysis_service import ExperimentAnalysisService

st.set_page_config(page_title="Report & Reproducibility — ExPO", page_icon="📑", layout="wide")
st.title("📑 Report & Reproducibility — Auditable Manifests")

st.markdown("""
Review the full reproducibility manifest, data provenance, and export
self-contained reports in Markdown, HTML, JSON, and CSV formats.
""")

manifest: ExperimentReportManifest = None

if "manifest" in st.session_state:
    manifest = st.session_state["manifest"]
else:
    # Run analysis on active experiment or demo
    if "experiment_df" in st.session_state and "experiment_config" in st.session_state:
        service = ExperimentAnalysisService(st.session_state["experiment_config"])
        manifest = service.run_full_analysis(st.session_state["experiment_df"])
        st.session_state["manifest"] = manifest
    else:
        st.info("No active experiment manifest found. Generating demo analysis...")
        df, meta = generate_effect_normal(n_users=3000, seed=42, delta=2.0, rho=0.7)
        config = ExperimentConfig(
            experiment_id="exp_demo_report",
            metrics=[
                MetricSpec(name="revenue", label="Revenue", kind=MetricKind.CONTINUOUS_MEAN, column="metric_value", pre_covariate="pre_metric")
            ],
        )
        service = ExperimentAnalysisService(config)
        manifest = service.run_full_analysis(df)
        st.session_state["manifest"] = manifest

# Metadata summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Experiment ID", manifest.experiment_id)
with col2:
    st.metric("Run ID", manifest.run_id)
with col3:
    st.metric("Engine Version", f"v{manifest.expo_version}")
with col4:
    st.metric("Quality Status", manifest.data_quality_status)

st.markdown("---")
st.subheader("1. Download Reports & Data Artifacts")

# Generate export buffers in memory
tmp_dir = Path("results/temp_export")
tmp_dir.mkdir(parents=True, exist_ok=True)

json_p = export_report_json(manifest, tmp_dir / "report.json")
csv_p = export_report_csv(manifest, tmp_dir / "report.csv")
md_p = export_report_markdown(manifest, tmp_dir / "report.md")
html_p = export_report_html(manifest, tmp_dir / "report.html")

with open(json_p, "r", encoding="utf-8") as f:
    json_bytes = f.read()
with open(csv_p, "r", encoding="utf-8") as f:
    csv_bytes = f.read()
with open(md_p, "r", encoding="utf-8") as f:
    md_bytes = f.read()
with open(html_p, "r", encoding="utf-8") as f:
    html_bytes = f.read()

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
    st.download_button(
        label="📄 Download HTML Report",
        data=html_bytes,
        file_name=f"{manifest.experiment_id}_report.html",
        mime="text/html",
        type="primary",
    )
with col_d2:
    st.download_button(
        label="📝 Download Markdown (.md)",
        data=md_bytes,
        file_name=f"{manifest.experiment_id}_report.md",
        mime="text/markdown",
    )
with col_d3:
    st.download_button(
        label="📊 Download CSV Summary",
        data=csv_bytes,
        file_name=f"{manifest.experiment_id}_summary.csv",
        mime="text/csv",
    )
with col_d4:
    st.download_button(
        label="⚙️ Download Manifest (JSON)",
        data=json_bytes,
        file_name=f"{manifest.experiment_id}_manifest.json",
        mime="application/json",
    )

st.markdown("---")
st.subheader("2. Report Content Preview")
st.markdown(md_bytes)
