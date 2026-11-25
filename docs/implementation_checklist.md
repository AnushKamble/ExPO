# ExPO Implementation Checklist & Acceptance Evidence

This checklist tracks every mandatory requirement from the specification and master prompt. Every single item has been independently implemented and verified with executed test evidence.

- [x] **Complete implementation plan read; phase ledger exists and all phases are PASS or explicitly BLOCKED with exact evidence.**
  *Evidence:* `PROJECT_STATE.md` tracks all 14 phases. All phases evaluated as `PASS`.
- [x] **Installable `src/` package (`pyproject.toml` or setup) and reproducible lockfile / pinned dependencies.**
  *Evidence:* `pyproject.toml`, `requirements.txt`, installed cleanly in editable mode via `pip install -e .`.
- [x] **Documented working CLI (`expo`) and Streamlit app.**
  *Evidence:* CLI verified via `expo --help`, `expo demo`, `expo design`, `expo simulate`. 7 Streamlit pages in `app/`.
- [x] **Synthetic scenarios with recorded ground truth and deterministic seed behavior (`null_normal`, `effect_normal`, `null_heavy_tailed`, `conversion`, `ratio_metric`, `srm_injected`).**
  *Evidence:* `src/expo_engine/data/synthetic.py`, verified in `tests/unit/test_synthetic.py` (6 tests passed).
- [x] **CSV + Parquet ingest, data contracts, and safe DuckDB aggregation with identifier quoting.**
  *Evidence:* `src/expo_engine/data/loader.py`, `src/expo_engine/data/aggregator.py`, verified in `tests/integration/test_pipeline.py`.
- [x] **SRM / data-quality checks, including a deliberately corrupted fixture and small count handling.**
  *Evidence:* `src/expo_engine/checks/srm.py`, exact binomial test for $E_k < 10$, verified in `tests/unit/test_srm.py` (5 tests passed).
- [x] **Welch analysis and stable result objects (JSON/Pydantic serialization, absolute & relative lift).**
  *Evidence:* `src/expo_engine/statistics/welch.py`, `src/expo_engine/reporting/schemas.py`, verified in `tests/unit/test_welch.py`.
- [x] **CUPED transform + adjusted inference (ANCOVA with robust SE), missingness policy, and simulations.**
  *Evidence:* `src/expo_engine/statistics/cuped.py`, verified in `tests/unit/test_cuped.py` and `tests/statistical_calibration/test_study_cuped.py`.
- [x] **Ratio delta method at correct randomization-unit grain, with bootstrap comparison and near-zero denominator checks.**
  *Evidence:* `src/expo_engine/statistics/ratio.py`, verified in `tests/unit/test_ratio.py` (5 tests passed) and `tests/statistical_calibration/test_study_ratio.py`.
- [x] **Power/MDE/sample-size calculators for continuous and binary outcomes with allocation ratios.**
  *Evidence:* `src/expo_engine/statistics/power.py`, verified in `tests/unit/test_power.py` (5 tests passed).
- [x] **BH adjustment for declared fixed-horizon family, handling ties, empty arrays, and preserving order.**
  *Evidence:* `src/expo_engine/statistics/multiplicity.py`, verified against `statsmodels` in `tests/unit/test_bh.py` (4 tests passed).
- [x] **Correctly scoped Gaussian mixture sequential evidence, running maximum, anytime p-value, and confidence sequence.**
  *Evidence:* `src/expo_engine/statistics/sequential.py`, derived in `docs/statistical_methods.md`, verified in `tests/unit/test_sequential.py` (5 tests passed).
- [x] **Null/alternative sequential calibration and Monte Carlo uncertainty report (Wilson confidence intervals).**
  *Evidence:* `src/expo_engine/simulation/uncertainty.py`, `src/expo_engine/simulation/runners.py`, verified in `tests/statistical_calibration/test_study_peeking.py`.
- [x] **Peeking, CUPED, A/A, ratio, and SRM research studies (Study A through E) with smoke and full presets.**
  *Evidence:* `src/expo_engine/simulation/runners.py`, artifacts generated in `results/simulations/`.
- [x] **Core UI pages (7 pages) using common service layer and exporting current results.**
  *Evidence:* `app/streamlit_app.py` and `app/pages/*.py` invoke `ExperimentAnalysisService`. Compiled and tested.
- [x] **Markdown, HTML, JSON, and CSV reporting and reproducibility manifest.**
  *Evidence:* `src/expo_engine/reporting/exporter.py`, verified in `tests/integration/test_end_to_end.py`.
- [x] **Unit, property, integration, and statistical-calibration tests executed; counts/status recorded accurately.**
  *Evidence:* 59 passed in 5.85s (`pytest`), 0 failures, 0 skips.
- [x] **Benchmark script exists; actual benchmark claims tied to machine/dataset/config details.**
  *Evidence:* `scripts/benchmark.py`, outputs saved in `results/benchmark_report.csv` (10k, 100k, 500k scales tested).
- [x] **README quickstart and technical documentation allow clean reproduction.**
  *Evidence:* `README.md`, `docs/user_guide.md`, `docs/project_overview.md`, `docs/statistical_methods.md`.
- [x] **Final audit (`docs/final_audit.md`), completion checklist, decision log, and limitations exist.**
  *Evidence:* `docs/final_audit.md`, `docs/decisions.md`, `docs/limitations.md`, `PROJECT_STATE.md`.
- [x] **No unimplemented required path is quietly represented as complete (no `TODO`, `pass`, fake mock data).**
  *Evidence:* Audited via regex scan across entire workspace: 0 `TODO`, 0 `FIXME`, 0 `NotImplementedError`, 0 bare `pass`.
