from pathlib import Path
import duckdb
import pandas as pd
import pytest
from expo_engine.data.aggregator import aggregate_events_to_users
from expo_engine.data.loader import load_dataset, register_dataframe, validate_file_path


def test_duckdb_event_aggregation(tmp_path: Path):
    con = duckdb.connect(":memory:")

    # Create assignments table: 3 users
    # User 1: control, assigned at 2026-01-01 10:00:00
    # User 2: treatment, assigned at 2026-01-01 10:00:00
    # User 3: treatment, assigned at 2026-01-01 10:00:00
    assignments_df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "variant": ["control", "treatment", "treatment"],
        "assigned_at": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 10:00:00", "2026-01-01 10:00:00"]),
    })
    register_dataframe(con, "assignments", assignments_df)

    # Create events table:
    # User 1 has pre-event (value=10) and post-event (value=25)
    # User 2 has post-event (value=40) and ratio numerator event
    # User 3 has no events (defaults to 0.0)
    events_df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u2"],
        "event_time": pd.to_datetime([
            "2026-01-01 08:00:00",  # pre
            "2026-01-01 12:00:00",  # post
            "2026-01-01 14:00:00",  # post
            "2026-01-01 15:00:00",  # click (numerator)
            "2026-01-01 15:00:00",  # impression (denominator)
        ]),
        "event_name": ["purchase", "purchase", "purchase", "click", "impression"],
        "value": [10.0, 25.0, 40.0, 1.0, 1.0],
    })
    register_dataframe(con, "events", events_df)

    user_df, manifest = aggregate_events_to_users(
        con=con,
        assignments_table="assignments",
        events_table="events",
        user_id_col="user_id",
        variant_col="variant",
        assignment_time_col="assigned_at",
        event_time_col="event_time",
        post_event_names=["purchase"],
        pre_event_names=["purchase"],
        value_col="value",
        numerator_event="click",
        denominator_event="impression",
    )

    assert manifest["total_assigned_users"] == 3
    assert len(user_df) == 3

    # Check user 1
    u1 = user_df[user_df["user_id"] == "u1"].iloc[0]
    assert u1["metric_value"] == 25.0
    assert u1["pre_metric"] == 10.0
    assert u1["numerator"] == 0.0
    assert u1["denominator"] == 0.0

    # Check user 2
    u2 = user_df[user_df["user_id"] == "u2"].iloc[0]
    assert u2["metric_value"] == 40.0
    assert u2["pre_metric"] == 0.0
    assert u2["numerator"] == 1.0
    assert u2["denominator"] == 1.0

    # Check user 3
    u3 = user_df[user_df["user_id"] == "u3"].iloc[0]
    assert u3["metric_value"] == 0.0
    assert u3["pre_metric"] == 0.0


def test_csv_and_parquet_loaders(tmp_path: Path):
    df_sample = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_file = tmp_path / "sample.csv"
    parquet_file = tmp_path / "sample.parquet"

    df_sample.to_csv(csv_file, index=False)
    df_sample.to_parquet(parquet_file, index=False)

    df_csv = load_dataset(csv_file)
    df_parquet = load_dataset(parquet_file)

    pd.testing.assert_frame_equal(df_csv, df_sample)
    pd.testing.assert_frame_equal(df_parquet, df_sample)
