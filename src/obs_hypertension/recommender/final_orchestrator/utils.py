"""
Utility helpers for final clinical recommendation orchestration.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    """
    Normalize text safely for robust comparison.
    """
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    return str(value).strip().lower()


def safe_row_value(row: pd.Series, col: str) -> str:
    """
    Safely extract a string value from a row column.
    """
    if col not in row.index or pd.isna(row[col]):
        return ""
    return str(row[col]).strip()


def build_treatment_signature(row: pd.Series) -> str:
    """
    Build a normalized treatment signature to detect repeated treatments.
    """
    parts = [
        normalize_text(safe_row_value(row, "drug_1")),
        normalize_text(safe_row_value(row, "route_1")),
        normalize_text(safe_row_value(row, "dose_1")),
        normalize_text(safe_row_value(row, "drug_2")),
        normalize_text(safe_row_value(row, "route_2")),
        normalize_text(safe_row_value(row, "dose_2")),
    ]
    return " || ".join(parts)