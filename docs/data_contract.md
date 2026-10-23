# ExPO Data Contracts & Schema Specification

This document defines the canonical schemas, input contracts, validation constraints, aggregation rules, and missingness policies across ExPO.

---

## 1. Randomization and Analysis Units

- **Unit of Randomization:** The independent experimental entity assigned to a variant. By default in ExPO: `user_id`.
- **Unit of Analysis:** The grain at which metric values and statistical inference are calculated. ExPO enforces that the unit of analysis strictly matches the unit of randomization (`user_id`).
- **Violation Guard:** Analyzing session-level or event-level data as independent observations when users were randomized produces artificially deflated standard errors and severe Type I error inflation. ExPO strictly aggregates events to the randomization unit before calculating test statistics.

---

## 2. Canonical User-Level Table (`UserExperimentData`)

The core fixed-horizon and sequential statistical engines operate on a canonical user-level table with one row per randomized entity.

### Schema Definition

| Column | Type | Nullable | Description / Rules |
|---|---|---|---|
| `experiment_id` | `VARCHAR` | No | Identifier for the experiment. Multiple experiment IDs in a single run are rejected unless explicitly grouped. |
| `user_id` | `VARCHAR` / `BIGINT` | No | Unique identifier for the analysis unit. Must be unique within the table (no duplicate users). |
| `variant` | `VARCHAR` | No | Assigned variant. For two-arm tests, must match configured control/treatment labels (e.g. `control`, `treatment`). |
| Primary Metric (`metric_value`) | `FLOAT` / `INTEGER` | No | Continuous outcome, binary conversion (0/1), or count. Must be finite (no NaN, Inf). |
| Pre-metric / Covariate (`pre_metric`) | `FLOAT` / `INTEGER` | Yes | Pre-treatment covariate measured *before* assignment/exposure. Missing values tracked via missingness policy. |
| Ratio Numerator (`numerator`) | `FLOAT` / `INTEGER` | Optional | User-level sum of numerator events for ratio metrics. |
| Ratio Denominator (`denominator`) | `FLOAT` / `INTEGER` | Optional | User-level sum of denominator events for ratio metrics. Must be non-negative; sum across arm must be > 0. |
| `assigned_at` | `TIMESTAMP` | Yes | Timestamp of randomization assignment. |
| `exposed_at` | `TIMESTAMP` | Yes | Timestamp of first treatment exposure. |

---

## 3. Event-Level Source Tables & Aggregation Pipeline

When ingesting raw events, ExPO uses DuckDB to perform safe, validated aggregation to the canonical user level.

### 3.1 Assignments Table (`assignments`)
- `experiment_id`: `VARCHAR`
- `user_id`: `VARCHAR` or `BIGINT`
- `variant`: `VARCHAR`
- `assigned_at`: `TIMESTAMP`

### 3.2 Events Table (`events`)
- `experiment_id`: `VARCHAR`
- `user_id`: `VARCHAR` or `BIGINT`
- `event_time`: `TIMESTAMP`
- `event_name`: `VARCHAR`
- `value`: `FLOAT` (optional numeric payload, e.g. transaction amount)

### 3.3 Aggregation Rules
1. **Deduplication:** Multiple assignments of the same user to differing variants are flagged as `VARIANT_CONTAMINATION` and rejected. Multiple assignments to the same variant are deduplicated to the earliest `assigned_at`.
2. **Time-Window Filtering:**
   - Post-treatment window: `event_time >= assigned_at` (or `exposed_at` if exposure window configured) and `event_time <= experiment_end`.
   - Pre-treatment window (CUPED covariate): `event_time < assigned_at` within a predeclared lookback window (e.g. 14 days).
3. **Safe SQL Generation:** Column names, table names, and metric expressions are strictly validated against an identifier allow-list and quoted via DuckDB parameterization. Arbitrary raw SQL strings from users are forbidden.

---

## 4. Input Validation Rules & Failure Modes

ExPO enforces strict fail-early validation. The table below lists all failure conditions and their respective error codes:

| Failure Mode | Trigger Condition | Severity | System Behavior |
|---|---|---|---|
| `SCHEMA_MISSING_COLUMNS` | Required canonical columns absent | FATAL | Abort run; return actionable missing column list |
| `DUPLICATE_USERS` | Multiple rows for same `user_id` | FATAL | Abort run; report duplicate user count and sample IDs |
| `VARIANT_CONTAMINATION` | Same `user_id` assigned to multiple variants | FATAL | Abort run; report contaminated users |
| `NON_FINITE_METRICS` | Metric contains `NaN`, `+Inf`, or `-Inf` | FATAL | Abort run; report row indices and column names |
| `INVALID_BINARY_VALUES` | Binary metric contains values other than 0, 1 | FATAL | Abort run; report invalid values |
| `NON_POSITIVE_DENOMINATOR` | Total ratio denominator for an arm $\le 0$ | FATAL | Abort run; ratio calculation mathematically impossible |
| `ZERO_VARIANCE_COVARIATE` | Pre-covariate variance = 0 (constant) | WARNING | Disable CUPED adjustment; fall back to unadjusted Welch |
| `SRM_DETECTED` | Chi-square p-value < 0.001 against planned split | WARNING/BLOCK | Display warning; flag causal conclusions as unreliable |
| `HIGH_MISSING_COVARIATES` | Missing covariate rate > 30% or imbalanced | WARNING | Run complete-case CUPED with warning + full unadjusted |

---

## 5. Missingness and Covariate Policy

- **Outcome Missingness:** Any analysis unit missing the primary outcome is excluded with an explicit record in the analysis manifest. Silent deletion is forbidden.
- **Covariate Missingness:**
  - Complete-case adjusted inference: ANCOVA/CUPED is computed strictly on units with non-null pre-experiment covariates.
  - Baseline comparison: The unadjusted Welch test is computed across all valid outcome rows to ensure transparency.
  - Differential missingness check: ExPO runs Fisher's exact / chi-square test on missingness rates between treatment and control. If missingness is differential ($p < 0.05$), an alert is triggered.
