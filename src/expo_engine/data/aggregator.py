"""
Safe DuckDB aggregation pipeline for event-level experimentation data.
Enforces strict identifier sanitization, parameterization, and canonical user-level transformation.
"""

from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd


def sanitize_identifier(ident: str) -> str:
    """
    Ensures SQL identifiers contain only valid alphanumeric characters and underscores.
    Prevents SQL injection via column or table names.
    """
    if not ident or not ident.isidentifier():
        raise ValueError(f"Invalid SQL identifier: '{ident}'. Must be a valid Python identifier.")
    return ident


def aggregate_events_to_users(
    con: duckdb.DuckDBPyConnection,
    assignments_table: str,
    events_table: str,
    user_id_col: str = "user_id",
    variant_col: str = "variant",
    assignment_time_col: str = "assigned_at",
    event_time_col: str = "event_time",
    post_event_names: Optional[List[str]] = None,
    pre_event_names: Optional[List[str]] = None,
    value_col: Optional[str] = "value",
    numerator_event: Optional[str] = None,
    denominator_event: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregates event-level records to canonical user-level metrics.
    All table and column names are validated and quoted.
    """
    # Sanitize all column and table identifiers
    a_tbl = sanitize_identifier(assignments_table)
    e_tbl = sanitize_identifier(events_table)
    u_col = sanitize_identifier(user_id_col)
    v_col = sanitize_identifier(variant_col)
    at_col = sanitize_identifier(assignment_time_col)
    et_col = sanitize_identifier(event_time_col)
    
    val_expr = f'COALESCE("{sanitize_identifier(value_col)}", 1.0)' if value_col else "1.0"

    # Verify duplicate assignment / variant contamination
    contamination_query = f"""
        SELECT "{u_col}", COUNT(DISTINCT "{v_col}") as variant_count
        FROM "{a_tbl}"
        GROUP BY "{u_col}"
        HAVING COUNT(DISTINCT "{v_col}") > 1
    """
    contaminated_users = con.execute(contamination_query).fetchall()
    if contaminated_users:
        raise ValueError(f"Variant contamination detected: {len(contaminated_users)} users assigned to multiple variants.")

    # Deduplicate assignments to earliest assignment timestamp
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _deduped_assignments AS
        SELECT "{u_col}", "{v_col}", MIN("{at_col}") as "{at_col}"
        FROM "{a_tbl}"
        GROUP BY "{u_col}", "{v_col}"
    """)

    # Build metric aggregation expressions
    metric_selects = []
    
    # Standard post-period metric
    if post_event_names:
        placeholders = ", ".join(["?"] * len(post_event_names))
        post_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" >= a."{at_col}" AND e.event_name IN ({placeholders}) 
                     THEN {val_expr} ELSE 0.0 END), 0.0) AS metric_value
        """
        metric_selects.append((post_expr, list(post_event_names)))
    else:
        post_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" >= a."{at_col}" 
                     THEN {val_expr} ELSE 0.0 END), 0.0) AS metric_value
        """
        metric_selects.append((post_expr, []))

    # Pre-period covariate (CUPED)
    if pre_event_names:
        placeholders_pre = ", ".join(["?"] * len(pre_event_names))
        pre_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" < a."{at_col}" AND e.event_name IN ({placeholders_pre}) 
                     THEN {val_expr} ELSE 0.0 END), 0.0) AS pre_metric
        """
        metric_selects.append((pre_expr, list(pre_event_names)))
    else:
        pre_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" < a."{at_col}" 
                     THEN {val_expr} ELSE 0.0 END), 0.0) AS pre_metric
        """
        metric_selects.append((pre_expr, []))

    # Ratio metric numerator / denominator if requested
    if numerator_event and denominator_event:
        num_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" >= a."{at_col}" AND e.event_name = ? 
                     THEN {val_expr} ELSE 0.0 END), 0.0) AS numerator
        """
        metric_selects.append((num_expr, [numerator_event]))
        denom_expr = f"""
            COALESCE(SUM(CASE WHEN e."{et_col}" >= a."{at_col}" AND e.event_name = ? 
                     THEN 1.0 ELSE 0.0 END), 0.0) AS denominator
        """
        metric_selects.append((denom_expr, [denominator_event]))

    # Combine SQL query safely
    select_clauses = ",\n    ".join([clause for clause, _ in metric_selects])
    params = []
    for _, p_list in metric_selects:
        params.extend(p_list)

    query = f"""
        SELECT 
            a."{u_col}" AS user_id,
            a."{v_col}" AS variant,
            {select_clauses}
        FROM _deduped_assignments a
        LEFT JOIN "{e_tbl}" e ON a."{u_col}" = e."{u_col}"
        GROUP BY a."{u_col}", a."{v_col}"
        ORDER BY a."{u_col}"
    """

    user_df = con.execute(query, params).df()

    # Auditing manifest
    total_assigned = con.execute("SELECT COUNT(*) FROM _deduped_assignments").fetchone()[0]
    users_with_events = con.execute(f'SELECT COUNT(DISTINCT "{u_col}") FROM "{e_tbl}"').fetchone()[0]

    manifest = {
        "total_assigned_users": int(total_assigned),
        "users_with_events": int(users_with_events),
        "aggregated_canonical_rows": len(user_df),
    }

    return user_df, manifest
