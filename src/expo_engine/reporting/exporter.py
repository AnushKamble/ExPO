"""
Export utilities for ExPO analysis manifests.
Generates JSON, CSV summary tables, Markdown reports, and self-contained HTML reports.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from expo_engine.reporting.schemas import ExperimentReportManifest


def export_report_json(manifest: ExperimentReportManifest, out_path: Path) -> Path:
    """
    Exports the report manifest to formatted JSON.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return out_path


def export_report_csv(manifest: ExperimentReportManifest, out_path: Path) -> Path:
    """
    Exports key statistical metrics to a clean CSV summary table.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # Primary Welch result
    if manifest.primary_metric_result:
        res = manifest.primary_metric_result
        rows.append({
            "section": "primary_unadjusted",
            "metric": res.get("metric_name"),
            "method": res.get("method"),
            "estimate": res.get("absolute_effect") or res.get("delta_ratio"),
            "se": res.get("standard_error"),
            "ci_lower": res.get("ci_lower"),
            "ci_upper": res.get("ci_upper"),
            "p_value": res.get("p_value"),
            "is_significant": res.get("is_statistically_significant"),
        })

    # CUPED result
    if manifest.cuped_result:
        c = manifest.cuped_result
        rows.append({
            "section": "primary_cuped_adjusted",
            "metric": c.get("metric_name"),
            "method": "CUPED / ANCOVA Robust Regression",
            "estimate": c.get("adjusted_ate"),
            "se": c.get("adjusted_se"),
            "ci_lower": c.get("adjusted_ci_lower"),
            "ci_upper": c.get("adjusted_ci_upper"),
            "p_value": c.get("adjusted_p_value"),
            "is_significant": c.get("is_statistically_significant"),
        })

    # Secondary hypotheses
    if manifest.secondary_multiplicity:
        for hyp in manifest.secondary_multiplicity.hypotheses:
            rows.append({
                "section": "secondary_bh_adjusted",
                "metric": hyp.metric_name,
                "method": "Benjamini-Hochberg FDR",
                "estimate": None,
                "se": None,
                "ci_lower": None,
                "ci_upper": None,
                "p_value": hyp.adjusted_p_value,
                "is_significant": hyp.rejected,
            })

    # Sequential result
    if manifest.sequential_result and manifest.sequential_result.get("eligible"):
        seq = manifest.sequential_result
        rows.append({
            "section": "sequential_monitoring",
            "metric": seq.get("metric_name"),
            "method": seq.get("method"),
            "estimate": seq.get("final_mean_diff"),
            "se": None,
            "ci_lower": seq.get("final_cs_lower"),
            "ci_upper": seq.get("final_cs_upper"),
            "p_value": seq.get("final_anytime_p_value"),
            "is_significant": seq.get("stopped_early"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def export_report_markdown(manifest: ExperimentReportManifest, out_path: Path) -> Path:
    """
    Exports a comprehensive GitHub-flavored Markdown report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_lines = [
        f"# ExPO Experiment Analysis Report: `{manifest.experiment_id}`",
        f"- **Run ID:** `{manifest.run_id}`",
        f"- **Timestamp (UTC):** `{manifest.timestamp_utc}`",
        f"- **Engine Version:** `{manifest.expo_version}`",
        f"- **Data Quality Status:** **{manifest.data_quality_status}**",
        "",
        "---",
        "",
        "## 1. Data Integrity & Sample Ratio Mismatch (SRM)",
    ]

    if manifest.srm_result:
        srm = manifest.srm_result
        md_lines.extend([
            f"- **Status:** `{srm.get('status')}`",
            f"- **Goodness-of-Fit p-value:** `{srm.get('p_value'):.4e}` (Threshold: `{srm.get('threshold')}`)",
            f"- **Method:** `{srm.get('method')}`",
            f"- **Observed Counts:** {srm.get('observed_counts')}",
            f"- **Expected Counts:** {srm.get('expected_counts')}",
            f"- **Audit Message:** {srm.get('message')}",
        ])
    else:
        md_lines.append("No SRM evaluation available.")

    md_lines.extend(["", "---", "", "## 2. Primary Metric Inference"])

    if manifest.primary_metric_result:
        p = manifest.primary_metric_result
        est = p.get("absolute_effect") or p.get("delta_ratio")
        rel_lift = p.get("relative_lift")
        rel_str = f"{rel_lift:.2%}" if rel_lift is not None else "N/A"

        md_lines.extend([
            f"### Metric: `{p.get('metric_name')}` (Unadjusted Baseline)",
            f"- **Method:** {p.get('method')}",
            f"- **Treatment Effect Estimate:** `{est:.4f}` (Relative Lift: `{rel_str}`)",
            f"- **Standard Error:** `{p.get('standard_error'):.4f}`",
            f"- **95% Confidence Interval:** `[{p.get('ci_lower'):.4f}, {p.get('ci_upper'):.4f}]`",
            f"- **p-value:** `{p.get('p_value'):.4e}`",
            f"- **Statistically Significant (alpha={p.get('alpha', 0.05)}):** `{p.get('is_statistically_significant')}`",
        ])

    if manifest.cuped_result:
        c = manifest.cuped_result
        md_lines.extend([
            "",
            f"### CUPED & ANCOVA Covariate Adjustment (`{c.get('covariate_name')}`)",
            f"- **Adjusted Treatment Effect:** `{c.get('adjusted_ate'):.4f}`",
            f"- **Robust Standard Error (HC1):** `{c.get('adjusted_se'):.4f}` (vs Raw SE `{c.get('unadjusted_welch', {}).get('standard_error', 0.0):.4f}`)",
            f"- **Adjusted 95% CI:** `[{c.get('adjusted_ci_lower'):.4f}, {c.get('adjusted_ci_upper'):.4f}]`",
            f"- **Adjusted p-value:** `{c.get('adjusted_p_value'):.4e}`",
            f"- **Covariate Correlation (rho):** `{c.get('sample_correlation'):.4f}`",
            f"- **Theoretical Variance Reduction (1 - rho^2):** `{c.get('theoretical_variance_reduction'):.2%}`",
            f"- **Empirical Variance Reduction:** `{c.get('empirical_variance_reduction'):.2%}`",
            f"- **Complete Cases Used:** `{c.get('complete_cases_count')}` (Missing pre-covariate: `{c.get('missing_covariate_count')}`)",
        ])

    if manifest.secondary_multiplicity and manifest.secondary_multiplicity.hypotheses:
        md_lines.extend([
            "",
            "---",
            "",
            "## 3. Secondary & Guardrail Metrics (Benjamini-Hochberg FDR)",
            f"- **Correction Method:** {manifest.secondary_multiplicity.method}",
            f"- **Declared Target FDR (q):** `{manifest.secondary_multiplicity.fdr_q}`",
            f"- **Total Tests:** `{manifest.secondary_multiplicity.total_tests}` | **Rejections:** `{manifest.secondary_multiplicity.rejections_count}`",
            "",
            "| Metric | Raw p-value | BH Adjusted p-value | Rejected (FDR Controlled) |",
            "|---|---|---|---|",
        ])
        for hyp in manifest.secondary_multiplicity.hypotheses:
            md_lines.append(
                f"| `{hyp.metric_name}` | `{hyp.raw_p_value:.4f}` | `{hyp.adjusted_p_value:.4f}` | `{'YES' if hyp.rejected else 'NO'}` |"
            )

    if manifest.sequential_result:
        seq = manifest.sequential_result
        md_lines.extend([
            "",
            "---",
            "",
            "## 4. Sequential Monitoring (Gaussian Mixture e-Process)",
            f"- **Model Eligibility:** `{seq.get('eligible')}`",
        ])
        if seq.get("eligible"):
            md_lines.extend([
                f"- **Frozen Pilot Variance (sigma_D^2):** `{seq.get('known_difference_variance')}`",
                f"- **Predeclared Mixture Variance (tau^2):** `{seq.get('mixture_variance')}`",
                f"- **Total Balanced Pairs Monitored:** `{seq.get('total_pairs')}` across `{seq.get('total_looks_analyzed')}` looks",
                f"- **Stopped Early:** `{'YES (Look ' + str(seq.get('stopping_look')) + ')' if seq.get('stopped_early') else 'NO'}`",
                f"- **Final Anytime-Valid p-value:** `{seq.get('final_anytime_p_value'):.4e}`",
                f"- **Final Time-Uniform Confidence Sequence:** `[{seq.get('final_cs_lower'):.4f}, {seq.get('final_cs_upper'):.4f}]`",
            ])
        else:
            md_lines.append(f"- **Ineligibility Reason:** {seq.get('ineligibility_reason')}")

    if manifest.warnings:
        md_lines.extend([
            "",
            "---",
            "",
            "## 5. Audit Warnings & Caveats",
        ])
        for w in manifest.warnings:
            md_lines.append(f"- > [!WARNING] {w}")

    content = "\n".join(md_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def export_report_html(manifest: ExperimentReportManifest, out_path: Path) -> Path:
    """
    Exports a self-contained, beautifully styled HTML report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML content
    md_path = out_path.with_suffix(".tmp.md")
    export_report_markdown(manifest, md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    if md_path.exists():
        md_path.unlink()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ExPO Report - {manifest.experiment_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background-color: #f9fafb;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        h1, h2, h3 {{ color: #111827; }}
        h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px 14px;
            border: 1px solid #e5e7eb;
            text-align: left;
        }}
        th {{ background-color: #f3f4f6; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .badge-clean {{ background-color: #d1fae5; color: #065f46; }}
        .badge-alert {{ background-color: #fee2e2; color: #991b1b; }}
        .warning-box {{
            background-color: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            margin: 10px 0;
        }}
        pre {{
            background: #1f2937;
            color: #f9fafb;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ExPO Experimentation Outcomes Report</h1>
        <p><strong>Experiment ID:</strong> {manifest.experiment_id} | <strong>Run ID:</strong> {manifest.run_id}</p>
        <p><strong>Timestamp:</strong> {manifest.timestamp_utc} | <strong>Engine:</strong> ExPO v{manifest.expo_version}</p>
        <p><strong>Data Quality Status:</strong> <span class="badge {'badge-alert' if 'ALERT' in manifest.data_quality_status else 'badge-clean'}">{manifest.data_quality_status}</span></p>
        <hr/>
        <pre>{md_text}</pre>
    </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return out_path
