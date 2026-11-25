"""
Command-Line Interface (CLI) entry point for ExPO.
Implemented with Typer to expose all core analytical services, simulations, and validation.
"""

from pathlib import Path
from typing import Optional
import json
import typer
import yaml
import pandas as pd

from expo_engine.checks.validation import validate_experiment_data
from expo_engine.config.models import ExperimentConfig
from expo_engine.data.loader import load_dataset
from expo_engine.data.synthetic import generate_effect_normal
from expo_engine.reporting.exporter import (
    export_report_csv,
    export_report_html,
    export_report_json,
    export_report_markdown,
)
from expo_engine.reporting.schemas import ExperimentReportManifest
from expo_engine.services.analysis_service import ExperimentAnalysisService
from expo_engine.simulation.runners import (
    run_aa_simulation,
    run_cuped_simulation,
    run_peeking_simulation,
    run_ratio_simulation,
    run_srm_simulation,
)
from expo_engine.statistics.power import (
    compute_binary_sample_size,
    compute_continuous_sample_size,
)

app = typer.Typer(
    name="expo",
    help="ExPO — Experimentation & Product Outcomes Engine: Trustworthy A/B Analysis and Scoped Sequential Inference.",
    add_completion=False,
)

data_app = typer.Typer(name="data", help="Data loading, schema inspection, and contract validation.")
simulate_app = typer.Typer(name="simulate", help="Monte Carlo research simulation runners.")

app.add_typer(data_app)
app.add_typer(simulate_app)


@app.command()
def demo(
    out_dir: str = typer.Option("results/demo", help="Directory where demo artifacts are saved"),
    seed: int = typer.Option(42, help="Random seed for synthetic data generation"),
):
    """
    Executes a complete end-to-end demonstration on controlled synthetic data.
    """
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho("   ExPO — Canonical End-to-End Experiment Demo    ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic data
    typer.echo("[1/4] Generating reproducible synthetic dataset (N=3,000, known delta=2.0, rho=0.7)...")
    df, meta = generate_effect_normal(
        n_users=3000,
        seed=seed,
        delta=2.0,
        mu_y=50.0,
        sigma_y=10.0,
        rho=0.7,
        missing_covariate_rate=0.03,
    )
    df["page_views"] = df["metric_value"] * 0.4 + 5.0

    data_file = out_path / "synthetic_users.parquet"
    df.to_parquet(data_file, index=False)
    typer.secho(f"      Saved raw synthetic data to {data_file}", fg=typer.colors.GREEN)

    # 2. Load configuration
    config_file = Path("configs/demo_experiment.yaml")
    if not config_file.exists():
        typer.secho(f"Error: Config file not found at {config_file}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    config = ExperimentConfig(**config_data)

    # 3. Execute analysis
    typer.echo("[2/4] Running full statistical analysis pipeline...")
    service = ExperimentAnalysisService(config)
    manifest = service.run_full_analysis(df, run_id="canonical_demo_run")

    # 4. Export reports
    typer.echo("[3/4] Exporting Markdown, HTML, JSON, and CSV reports...")
    export_report_json(manifest, out_path / "report.json")
    export_report_csv(manifest, out_path / "report.csv")
    export_report_markdown(manifest, out_path / "report.md")
    export_report_html(manifest, out_path / "report.html")

    # 5. Display key findings
    typer.echo("[4/4] Analysis complete! Results Summary:")
    typer.secho("--------------------------------------------------", fg=typer.colors.WHITE)

    # Primary Welch
    w = manifest.primary_metric_result
    typer.echo(f"  Primary Metric (Unadjusted): {w['metric_name']}")
    typer.echo(f"    Raw ATE:            {w['absolute_effect']:.4f} (SE: {w['standard_error']:.4f})")
    typer.echo(f"    Raw 95% CI:         [{w['ci_lower']:.4f}, {w['ci_upper']:.4f}]")
    typer.echo(f"    p-value:            {w['p_value']:.4e} (Significant: {w['is_statistically_significant']})")

    # CUPED
    c = manifest.cuped_result
    if c:
        typer.secho("  CUPED / ANCOVA Adjustment:", fg=typer.colors.YELLOW)
        typer.echo(f"    Adjusted ATE:       {c['adjusted_ate']:.4f} (Robust SE: {c['adjusted_se']:.4f})")
        typer.echo(f"    Adjusted 95% CI:    [{c['adjusted_ci_lower']:.4f}, {c['adjusted_ci_upper']:.4f}]")
        typer.echo(f"    Realized Var Reduc: {c['empirical_variance_reduction']:.1%} (Theoretical 1-rho^2: {c['theoretical_variance_reduction']:.1%})")

    # Sequential
    s = manifest.sequential_result
    if s and s["eligible"]:
        typer.secho("  Sequential Inference (Gaussian Mixture mSPRT):", fg=typer.colors.MAGENTA)
        typer.echo(f"    Stopped Early:      {s['stopped_early']} (Anytime p-value: {s['final_anytime_p_value']:.4e})")
        typer.echo(f"    Confidence Sequence:[{s['final_cs_lower']:.4f}, {s['final_cs_upper']:.4f}]")

    typer.secho("--------------------------------------------------", fg=typer.colors.WHITE)
    typer.secho(f"Report exported to: {out_path / 'report.html'}", fg=typer.colors.GREEN, bold=True)


@data_app.command("validate")
def validate_data(
    input: str = typer.Option(..., "--input", "-i", help="Path to CSV or Parquet experiment dataset"),
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment YAML configuration"),
):
    """
    Validates an experiment dataset against the schema and data contract.
    """
    p_data = Path(input)
    p_conf = Path(config)

    if not p_data.exists():
        typer.secho(f"Input file not found: {p_data}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not p_conf.exists():
        typer.secho(f"Config file not found: {p_conf}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = load_dataset(p_data)
    with open(p_conf, "r", encoding="utf-8") as f:
        conf_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig(**conf_dict)

    res = validate_experiment_data(df, exp_config)
    if not res.is_valid:
        typer.secho("VALIDATION FAILED:", fg=typer.colors.RED, bold=True)
        for err in res.errors:
            typer.secho(f"  - [ERROR] {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho("VALIDATION PASSED!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Total Rows: {res.total_rows}")
    typer.echo(f"  Unique Users: {res.unique_users}")
    typer.echo(f"  Variants: {res.variant_counts}")
    if res.warnings:
        typer.secho("Warnings:", fg=typer.colors.YELLOW)
        for w in res.warnings:
            typer.secho(f"  - [WARNING] {w}", fg=typer.colors.YELLOW)


@app.command("analyze")
def analyze(
    input: str = typer.Option(..., "--input", "-i", help="Path to input experiment dataset"),
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment YAML configuration"),
    out: str = typer.Option("results/run_latest", "--out", "-o", help="Output directory for reports"),
    run_id: str = typer.Option("run_cli", "--run-id", help="Unique run identifier"),
):
    """
    Runs full fixed-horizon, CUPED, ratio, and sequential inference on an input dataset.
    """
    p_data = Path(input)
    p_conf = Path(config)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    if not p_data.exists():
        typer.secho(f"Input file not found: {p_data}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if not p_conf.exists():
        typer.secho(f"Config file not found: {p_conf}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    df = load_dataset(p_data)
    with open(p_conf, "r", encoding="utf-8") as f:
        conf_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig(**conf_dict)

    service = ExperimentAnalysisService(exp_config)
    try:
        manifest = service.run_full_analysis(df, run_id=run_id)
    except Exception as e:
        typer.secho(f"Analysis failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    export_report_json(manifest, out_path / "report.json")
    export_report_csv(manifest, out_path / "report.csv")
    export_report_markdown(manifest, out_path / "report.md")
    export_report_html(manifest, out_path / "report.html")

    typer.secho(f"Analysis successful. Reports saved in {out_path}", fg=typer.colors.GREEN, bold=True)


@app.command("design")
def design(
    config: str = typer.Option(..., "--config", "-c", help="Path to design YAML configuration"),
    out: str = typer.Option("results/design.json", "--out", "-o", help="Output JSON path"),
):
    """
    Computes statistical power, MDE, sample size, and duration planning estimates.
    """
    p_conf = Path(config)
    if not p_conf.exists():
        typer.secho(f"Design config not found: {p_conf}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(p_conf, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    kind = d.get("metric_kind", "continuous_mean")
    if kind == "continuous_mean":
        res = compute_continuous_sample_size(
            baseline_variance=d["baseline_variance"],
            mde_absolute=d["mde_absolute"],
            baseline_mean=d.get("baseline_mean"),
            alpha=d.get("alpha", 0.05),
            power=d.get("target_power", 0.80),
            allocation_ratio=d.get("allocation_ratio", 1.0),
            cuped_rho=d.get("cuped_rho"),
            daily_sample_rate=d.get("daily_sample_rate"),
        )
    elif kind == "binary_mean":
        res = compute_binary_sample_size(
            baseline_rate=d["baseline_rate"],
            mde_absolute=d["mde_absolute"],
            alpha=d.get("alpha", 0.05),
            power=d.get("target_power", 0.80),
            allocation_ratio=d.get("allocation_ratio", 1.0),
            daily_sample_rate=d.get("daily_sample_rate"),
        )
    else:
        typer.secho(f"Unsupported metric kind: {kind}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    out_file = Path(out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(res.model_dump_json(indent=2))

    typer.secho("EXPERIMENT DESIGN RESULTS:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Per-Arm Sample Size: Control={res.control_sample_size}, Treatment={res.treatment_sample_size}")
    typer.echo(f"  Total Sample Size:   {res.total_sample_size}")
    if res.cuped_projected_total:
        typer.echo(f"  CUPED Projected:     {res.cuped_projected_total} users (assumed rho={res.cuped_correlation_assumed})")
    if res.estimated_duration_days:
        typer.echo(f"  Estimated Duration:  {res.estimated_duration_days} days")
    typer.secho(f"Saved to {out_file}", fg=typer.colors.GREEN)


@simulate_app.command("peeking")
def sim_peeking(
    preset: str = typer.Option("smoke", help="'smoke', 'medium', or 'full'"),
    seed: int = typer.Option(42, help="Seed"),
    out: str = typer.Option("results/simulations/peeking.json", help="Output path"),
):
    """Study A: Peeking Inflation vs Sequential Evidence."""
    typer.echo(f"Running Peeking Simulation (preset={preset}, seed={seed})...")
    res = run_peeking_simulation(preset=preset, seed=seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    typer.secho("Peeking Simulation Completed:", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Fixed-Horizon FPR: {res['fixed_horizon']['rejection_rate']:.2%}")
    typer.echo(f"  Naive Peeking FPR: {res['naive_daily_peeking']['rejection_rate']:.2%}")
    typer.echo(f"  Sequential FPR:    {res['gaussian_mixture_sequential']['rejection_rate']:.2%}")
    typer.echo(f"Saved results to {p}")


@simulate_app.command("cuped")
def sim_cuped(
    preset: str = typer.Option("smoke", help="'smoke' or 'full'"),
    seed: int = typer.Option(42, help="Seed"),
    out: str = typer.Option("results/simulations/cuped.json", help="Output path"),
):
    """Study B: CUPED Variance Reduction across correlation grid."""
    typer.echo(f"Running CUPED Simulation (preset={preset}, seed={seed})...")
    res = run_cuped_simulation(preset=preset, seed=seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    typer.secho(f"CUPED Simulation completed. Saved to {p}", fg=typer.colors.GREEN)


@simulate_app.command("ratio")
def sim_ratio(
    preset: str = typer.Option("smoke", help="'smoke' or 'full'"),
    seed: int = typer.Option(42, help="Seed"),
    out: str = typer.Option("results/simulations/ratio.json", help="Output path"),
):
    """Study D: Ratio Delta Method vs Bootstrap Coverage."""
    typer.echo(f"Running Ratio Delta Simulation (preset={preset}, seed={seed})...")
    res = run_ratio_simulation(preset=preset, seed=seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    typer.secho(f"Ratio Simulation completed. Coverage: {res['empirical_coverage']:.2%}. Saved to {p}", fg=typer.colors.GREEN)


@simulate_app.command("aa")
def sim_aa(
    preset: str = typer.Option("smoke", help="'smoke', 'medium', or 'full'"),
    seed: int = typer.Option(42, help="Seed"),
    out: str = typer.Option("results/simulations/aa.json", help="Output path"),
):
    """Study C: A/A Null Calibration and P-Value Uniformity."""
    typer.echo(f"Running A/A Simulation (preset={preset}, seed={seed})...")
    res = run_aa_simulation(preset=preset, seed=seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    typer.secho(f"A/A Simulation completed. Empirical FPR: {res['observed_type_1_error']:.2%}, KS p-value: {res['ks_uniformity_p_value']:.4f}. Saved to {p}", fg=typer.colors.GREEN)


@simulate_app.command("srm")
def sim_srm(
    preset: str = typer.Option("smoke", help="'smoke', 'medium', or 'full'"),
    seed: int = typer.Option(42, help="Seed"),
    out: str = typer.Option("results/simulations/srm.json", help="Output path"),
):
    """Study E: SRM Detection Power vs Sample Size and Traffic Imbalance."""
    typer.echo(f"Running SRM Simulation (preset={preset}, seed={seed})...")
    res = run_srm_simulation(preset=preset, seed=seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    typer.secho(f"SRM Simulation completed. Scenarios evaluated: {len(res['matrix_results'])}. Saved to {p}", fg=typer.colors.GREEN)



@app.command("report")
def export_report(
    run: str = typer.Option(..., "--run", "-r", help="Directory containing run manifest report.json"),
    format: str = typer.Option("html", "--format", "-f", help="'html', 'markdown', 'csv', or 'json'"),
):
    """
    Exports an existing run manifest to the requested format.
    """
    p_run = Path(run)
    json_file = p_run / "report.json" if p_run.is_dir() else p_run
    if not json_file.exists():
        typer.secho(f"Report manifest not found at {json_file}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = ExperimentReportManifest(**data)

    fmt = format.lower()
    if fmt == "html":
        out = p_run / "exported_report.html"
        export_report_html(manifest, out)
    elif fmt == "markdown" or fmt == "md":
        out = p_run / "exported_report.md"
        export_report_markdown(manifest, out)
    elif fmt == "csv":
        out = p_run / "exported_report.csv"
        export_report_csv(manifest, out)
    else:
        out = p_run / "exported_report.json"
        export_report_json(manifest, out)

    typer.secho(f"Exported {fmt.upper()} report to {out}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
