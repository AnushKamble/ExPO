"""
Reproducible performance benchmarking script for ExPO.
Measures execution runtime, throughput (rows/sec), and memory across data scales (10k, 100k, 500k rows).
"""

import platform
import sys
import time
from typing import Dict, Any
import numpy as np
import pandas as pd

from expo_engine.checks.srm import check_srm
from expo_engine.checks.validation import validate_experiment_data
from expo_engine.config.models import ExperimentConfig, MetricKind, MetricRole, MetricSpec
from expo_engine.data.synthetic import generate_effect_normal
from expo_engine.statistics.cuped import compute_cuped
from expo_engine.statistics.sequential import evaluate_sequential_monitoring
from expo_engine.statistics.welch import compute_welch_ttest


def benchmark_scale(n_rows: int, seed: int = 42) -> Dict[str, Any]:
    print(f"\n--- Benchmarking Scale: {n_rows:,} rows ---")

    # 1. Synthetic Data Generation
    t0 = time.perf_counter()
    df, meta = generate_effect_normal(n_users=n_rows, seed=seed, delta=1.5, rho=0.7)
    t_gen = time.perf_counter() - t0
    print(f"  [1] Synthetic Generation: {t_gen:.4f}s ({n_rows / t_gen:,.0f} rows/s)")

    # 2. Schema Validation & SRM Check
    config = ExperimentConfig(
        experiment_id="bench_exp",
        metrics=[
            MetricSpec(name="revenue", label="Revenue", kind=MetricKind.CONTINUOUS_MEAN, column="metric_value", pre_covariate="pre_metric")
        ],
    )
    t0 = time.perf_counter()
    val_res = validate_experiment_data(df, config)
    srm_res = check_srm(df, variant_col="variant", planned_ratio=config.planned_ratio)
    t_val = time.perf_counter() - t0
    print(f"  [2] Validation & SRM:     {t_val:.4f}s ({n_rows / t_val:,.0f} rows/s)")

    # 3. Fixed-Horizon Welch t-test
    c_vals = df.loc[df["variant"] == "control", "metric_value"].values
    t_vals = df.loc[df["variant"] == "treatment", "metric_value"].values
    t0 = time.perf_counter()
    w_res = compute_welch_ttest(c_vals, t_vals, metric_name="revenue")
    t_welch = time.perf_counter() - t0
    print(f"  [3] Welch Two-Sample:     {t_welch:.4f}s ({n_rows / t_welch:,.0f} rows/s)")

    # 4. CUPED & Robust ANCOVA
    t0 = time.perf_counter()
    c_res = compute_cuped(df, metric_col="metric_value", pre_covariate_col="pre_metric")
    t_cuped = time.perf_counter() - t0
    print(f"  [4] CUPED / Robust ANCOVA: {t_cuped:.4f}s ({n_rows / t_cuped:,.0f} rows/s)")

    # 5. Sequential Monitoring (30 looks)
    t0 = time.perf_counter()
    seq_res = evaluate_sequential_monitoring(
        treatment_stream=t_vals,
        control_stream=c_vals,
        known_variance=meta["known_difference_variance"],
        look_frequency=max(10, n_rows // 60),
    )
    t_seq = time.perf_counter() - t0
    print(f"  [5] Sequential Monitoring: {t_seq:.4f}s")

    total_time = t_gen + t_val + t_welch + t_cuped + t_seq
    print(f"  Total Pipeline Time:      {total_time:.4f}s")

    return {
        "n_rows": n_rows,
        "generation_time_sec": t_gen,
        "validation_time_sec": t_val,
        "welch_time_sec": t_welch,
        "cuped_time_sec": t_cuped,
        "sequential_time_sec": t_seq,
        "total_time_sec": total_time,
        "overall_throughput_rows_per_sec": n_rows / total_time,
    }


def main():
    print("==================================================")
    print("      ExPO Engine Performance & Benchmark Audit   ")
    print("==================================================")
    print(f"Platform:       {platform.system()} ({platform.release()})")
    print(f"Processor:      {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")

    results = []
    for scale in [10_000, 100_000, 500_000]:
        res = benchmark_scale(scale)
        results.append(res)

    print("\n--- Summary Performance Table ---")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    out_file = "results/benchmark_report.csv"
    summary_df.to_csv(out_file, index=False)
    print(f"\nSaved benchmark outputs to {out_file}")


if __name__ == "__main__":
    main()
