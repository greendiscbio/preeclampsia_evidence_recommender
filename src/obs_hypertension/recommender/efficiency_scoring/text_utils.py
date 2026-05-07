"""
Text normalization helpers for efficiency scoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_text(value: Any) -> str:
    """
    Normalize text safely to lowercase string.

    Parameters
    ----------
    value : Any
        Raw value.

    Returns
    -------
    str
        Normalized text.
    """
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip().lower()


def is_yes(value: Any) -> bool:
    """
    Robust yes-like detector.

    Accepted positive values:
    - yes
    - y
    - true
    - 1
    """
    normalized = normalize_text(value)
    return normalized in {"yes", "y", "true", "1"}


def safe_lower_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series into lowercase stripped strings.

    Parameters
    ----------
    series : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Lowercased series.
    """
    return series.fillna("").astype(str).str.strip().str.lower()