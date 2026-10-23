"""
Safe data loading and path validation for CSV and Parquet files.
Prevents path traversal and validates file integrity.
"""

from pathlib import Path
from typing import Union
import duckdb
import pandas as pd
import pyarrow as pa


def register_dataframe(con: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    """
    Registers a pandas DataFrame into DuckDB safely via PyArrow to avoid dtype conflicts with Pandas 3.0.
    """
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")
    arrow_table = pa.Table.from_pandas(df)
    con.register(safe_table, arrow_table)


def validate_file_path(path: Union[str, Path], allowed_extensions: tuple = (".csv", ".parquet")) -> Path:
    """
    Validates that a path exists, has an allowed extension, and resolves safely.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a regular file: {p}")
    if p.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file extension '{p.suffix}'. Allowed: {allowed_extensions}")
    return p


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a dataset from CSV or Parquet into a pandas DataFrame.
    """
    p = validate_file_path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    elif p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def load_dataset_duckdb(path: Union[str, Path], con: duckdb.DuckDBPyConnection, table_name: str = "raw_data") -> str:
    """
    Registers a CSV or Parquet file directly into a DuckDB in-memory session.
    Returns the sanitized registered table name.
    """
    p = validate_file_path(path)
    # Sanitize table_name
    safe_table = "".join(c for c in table_name if c.isalnum() or c == "_")
    if not safe_table:
        safe_table = "raw_data"

    if p.suffix.lower() == ".csv":
        con.execute(f"CREATE OR REPLACE VIEW \"{safe_table}\" AS SELECT * FROM read_csv_auto(?)", [str(p)])
    else:
        con.execute(f"CREATE OR REPLACE VIEW \"{safe_table}\" AS SELECT * FROM read_parquet(?)", [str(p)])
    return safe_table
