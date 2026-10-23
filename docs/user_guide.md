# ExPO — User & Operator Guide

This guide describes how to install, configure, operate, and reproduce analyses using ExPO via the CLI and Streamlit web dashboard.

---

## 1. Quick Installation

```bash
git clone https://github.com/expo/expo.git
cd ExPO
pip install -e .
```

Verify installation:
```bash
expo --help
```

---

## 2. Command-Line Workflows

### 2.1 One-Command Canonical Demo
Generates a controlled synthetic experiment (3,000 users, known treatment effect $\delta=2.0$, pre-covariate correlation $\rho=0.7$), executes the entire pipeline, and writes reports to `results/demo/`:
```bash
expo demo
```

### 2.2 Validate an Experiment Dataset
Validates canonical schemas, checks for duplicate user IDs, variant contamination, and non-finite values:
```bash
expo data validate --input data/experiment.csv --config configs/demo_experiment.yaml
```

### 2.3 Run Experiment Analysis
Performs fixed-horizon Welch t-test, CUPED covariate adjustment, ratio metrics, BH multiple testing correction, and sequential inference:
```bash
expo analyze --input data/experiment.parquet --config configs/demo_experiment.yaml --out results/my_run
```

### 2.4 Experiment Planning & Power Calculator
Calculates required sample size, MDE, and duration based on variance or conversion baseline:
```bash
expo design --config configs/design_continuous.yaml --out results/design.json
```

### 2.5 Run Monte Carlo Research Simulations
Execute any of the 5 simulation studies:
```bash
expo simulate peeking --preset smoke --seed 42 --out results/simulations/peeking.json
expo simulate cuped --preset smoke --seed 42 --out results/simulations/cuped.json
expo simulate ratio --preset smoke --seed 42 --out results/simulations/ratio.json
```

---

## 3. Streamlit Interactive Dashboard

Launch the web application:
```bash
streamlit run app/streamlit_app.py
```

### Navigating the Pages
- **Home:** Overview and quick "Load Canonical Demo Experiment" button.
- **Data Lab:** Upload CSV/Parquet, configure column mappings, view real-time SRM goodness-of-fit charts and validation errors.
- **Experiment Designer:** Interactive sliders for alpha, power, MDE, and sample size with power curve visualizations.
- **Analyze:** Treatment effect cards, interactive Forest Plots comparing Raw Welch vs CUPED Adjusted 95% CIs, and Benjamini-Hochberg secondary metric tables.
- **Sequential Lab:** Real-time trajectory charts for mSPRT log-evidence, anytime-valid p-values, and time-uniform confidence sequence envelopes $[L_n, U_n]$.
- **Simulation Lab:** Run Monte Carlo research studies with Wilson score intervals and Plotly charts.
- **Report & Reproducibility:** Download self-contained HTML reports, Markdown summaries, JSON manifests, and CSV tables.
