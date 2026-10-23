# ExPO — Project Architecture & Overview

## 1. Executive Summary
ExPO is a local-first, CPU-only experimentation analysis platform and research suite. It empowers data science teams, experimentation scientists, and product analytics engineers to perform trustworthy A/B test analysis, automated data quality verification, optimal variance reduction (CUPED/ANCOVA), ratio-metric inference via the Delta method, and scoped anytime-valid sequential inference.

## 2. Core Modules Architecture

```text
src/expo_engine/
├── __init__.py           # Package version (0.1.0)
├── config/               # Pydantic configuration schemas & metric specifications
│   ├── models.py
│   └── __init__.py
├── data/                 # Deterministic hashing, synthetic data generator, loaders & DuckDB
│   ├── assignment.py
│   ├── synthetic.py
│   ├── loader.py
│   ├── aggregator.py
│   └── __init__.py
├── checks/               # Schema validation, missingness, and SRM goodness-of-fit
│   ├── validation.py
│   ├── srm.py
│   └── __init__.py
├── statistics/           # Statistical engines (pure math, no UI dependencies)
│   ├── welch.py
│   ├── cuped.py
│   ├── ratio.py
│   ├── power.py
│   ├── multiplicity.py
│   └── sequential.py
├── simulation/           # Monte Carlo research suite & uncertainty estimation
│   ├── runners.py
│   ├── uncertainty.py
│   └── __init__.py
├── reporting/            # Manifest schemas and report exporters (JSON/CSV/MD/HTML)
│   ├── schemas.py
│   ├── exporter.py
│   └── __init__.py
├── services/             # Orchestration service layer invoked by CLI & UI
│   ├── analysis_service.py
│   └── __init__.py
└── cli/                  # Typer command-line interface
    ├── main.py
    └── __init__.py
```

## 3. Key Design Tenets
1. **Zero UI Duplication:** All statistical calculations reside in `src/expo_engine/statistics/` and are orchestrated through `ExperimentAnalysisService`. Streamlit pages and Typer CLI call this identical service layer.
2. **Statistical Honesty:** No automated "Ship with 100% confidence" claims. Results explicitly highlight assumptions, sample sizes, confidence intervals, and audit warnings.
3. **Reproducibility:** All synthetic generation uses explicit integer seeds and `numpy.random.Generator`. Every analysis run generates an auditable `ExperimentReportManifest` with execution metadata.
