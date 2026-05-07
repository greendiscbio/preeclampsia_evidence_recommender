"""
Text normalization utilities for recommender vectorization.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, List

import pandas as pd

from src.obs_hypertension.recommender.vectorization.config import NULLISH


def strip_accents(text: str) -> str:
    """
    Remove accents from text.
    """
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def clean_text_one(value: Any) -> str:
    """
    Clean a single text value.

    Parameters
    ----------
    value : Any
        Raw input value.

    Returns
    -------
    str
        Cleaned text or empty string if nullish.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = ""
    else:
        text = str(value)

    text = strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in NULLISH:
        return ""

    return text


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Clean a pandas text series.

    Parameters
    ----------
    series : pd.Series
        Raw text series.

    Returns
    -------
    pd.Series
        Cleaned text series.
    """
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.map(strip_accents)
    cleaned = (
        cleaned.str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    cleaned = cleaned.replace(list(NULLISH), "")
    return cleaned


def ensure_columns(df: pd.DataFrame, cols: List[str], fill_value=0) -> pd.DataFrame:
    """
    Ensure the dataframe contains all requested columns.
    Missing columns are created with fill_value.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def safe_concat_text_cols(df: pd.DataFrame, cols: List[str], sep: str = " ") -> pd.Series:
    """
    Concatenate only existing columns into one text series.
    """
    existing_cols = [col for col in cols if col in df.columns]
    if not existing_cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[existing_cols].fillna("").astype(str).agg(sep.join, axis=1)