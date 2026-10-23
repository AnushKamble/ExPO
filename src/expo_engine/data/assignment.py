"""
Deterministic assignment utilities using cryptographic SHA-256 hashing.
Ensures reproducible, row-order independent variant allocation for synthetic
data generation and local experimentation simulations.
"""

import hashlib
from typing import Dict, List, Sequence, Union
import numpy as np
import pandas as pd


def hash_assignment(
    user_id: Union[str, int],
    experiment_id: str,
    salt: str = "expo_salt_2026",
) -> float:
    """
    Computes a deterministic hash value in [0, 1) for a given user_id and experiment_id.
    Uses SHA-256 to guarantee cross-process consistency and order independence.
    """
    key = f"{salt}:{experiment_id}:{str(user_id)}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    # Take first 12 hex characters (48 bits of entropy)
    int_val = int(digest[:12], 16)
    max_val = 16**12
    return int_val / max_val


def assign_variant(
    user_id: Union[str, int],
    experiment_id: str,
    variants: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] = (0.5, 0.5),
    salt: str = "expo_salt_2026",
) -> str:
    """
    Assigns a single user to a variant based on deterministic hash value.
    """
    if len(variants) != len(weights):
        raise ValueError("variants and weights must have identical length")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive")
    
    norm_weights = [w / total_weight for w in weights]
    cum_weights = np.cumsum(norm_weights)

    val = hash_assignment(user_id, experiment_id, salt=salt)
    for variant, threshold in zip(variants, cum_weights):
        if val <= threshold:
            return variant
    return variants[-1]


def assign_dataframe(
    df: pd.DataFrame,
    user_id_col: str,
    experiment_id: str,
    variant_col: str = "variant",
    variants: Sequence[str] = ("control", "treatment"),
    weights: Sequence[float] = (0.5, 0.5),
    salt: str = "expo_salt_2026",
) -> pd.DataFrame:
    """
    Deterministically assigns a DataFrame of users to variants.
    Row order does not affect user assignment.
    """
    result = df.copy()
    user_ids = result[user_id_col].astype(str).tolist()
    assigned = [
        assign_variant(uid, experiment_id, variants=variants, weights=weights, salt=salt)
        for uid in user_ids
    ]
    result[variant_col] = assigned
    return result
