import pandas as pd
import numpy as np
from expo_engine.data.assignment import assign_dataframe, assign_variant, hash_assignment


def test_hash_assignment_bounds_and_determinism():
    val1 = hash_assignment("user_123", "exp_1")
    val2 = hash_assignment("user_123", "exp_1")
    assert 0.0 <= val1 < 1.0
    assert val1 == val2

    # Different user or experiment produces different hash
    val3 = hash_assignment("user_124", "exp_1")
    val4 = hash_assignment("user_123", "exp_2")
    assert val1 != val3
    assert val1 != val4


def test_assignment_row_order_invariance():
    users = [f"user_{i}" for i in range(100)]
    df1 = pd.DataFrame({"user_id": users, "value": np.arange(100)})
    
    # Shuffle dataframe rows
    df2 = df1.sample(frac=1.0, random_state=123).reset_index(drop=True)
    
    assigned1 = assign_dataframe(df1, user_id_col="user_id", experiment_id="exp_test")
    assigned2 = assign_dataframe(df2, user_id_col="user_id", experiment_id="exp_test")
    
    # Merge on user_id to verify every user receives identical variant
    merged = pd.merge(assigned1, assigned2, on="user_id", suffixes=("_1", "_2"))
    assert (merged["variant_1"] == merged["variant_2"]).all()


def test_assignment_distribution():
    users = [f"u_{i:05d}" for i in range(5000)]
    df = pd.DataFrame({"user_id": users})
    assigned = assign_dataframe(
        df,
        user_id_col="user_id",
        experiment_id="exp_dist",
        variants=("control", "treatment"),
        weights=(0.5, 0.5),
    )
    counts = assigned["variant"].value_counts()
    share_treatment = counts["treatment"] / len(assigned)
    # 5000 users should be very close to 0.5 (within 3 std dev = 3*sqrt(0.25/5000) ~ 0.021)
    assert abs(share_treatment - 0.5) < 0.03
