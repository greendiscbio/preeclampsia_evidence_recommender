"""
Feature preparation utilities for efficiency scoring.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def ensure_numeric_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    fill_missing_cols_with: float = 0.0,
) -> pd.DataFrame:
    """
    Ensure all requested feature columns exist and are numeric.

    Behavior
    --------
    - Missing columns are added using a constant fill value.
    - Non-numeric columns are coerced to numeric, invalid parsing becomes NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : List[str]
        Feature columns to extract.
    fill_missing_cols_with : float
        Value used to create missing columns.

    Returns
    -------
    pd.DataFrame
        Numeric feature dataframe aligned to feature_cols.
    """
    X = df.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = fill_missing_cols_with

    X = X[feature_cols].copy()

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X