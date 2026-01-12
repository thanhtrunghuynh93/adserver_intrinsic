from typing import Callable, Any
import time
import pandas as pd
from typing import List
import json

def measure_runtime(func: Callable, *args, **kwargs) -> float:
    """
    Measure execution time of a function in seconds.
    """
    start = time.perf_counter()
    res = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Runtime: {elapsed:.4f} seconds")
    return res, elapsed

def extract_columns_from_json(
    df: pd.DataFrame,
    source_col: str,
    target_cols: List[str],
):
    """
    Extract selected fields from a JSON-string column into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    source_col : str
        Column containing JSON strings
    target_cols : List[str]
        Keys to extract from the JSON

    Returns
    -------
    pd.DataFrame
        DataFrame with extracted columns
    """

    def parse_row(x):
        try:
            data = json.loads(x) if pd.notna(x) else {}
            return pd.Series({col: data.get(col) for col in target_cols})
        except Exception:
            return pd.Series({col: None for col in target_cols})

    extracted = df[source_col].apply(parse_row)

    return pd.concat([df.drop(columns=[source_col]), extracted], axis=1)

