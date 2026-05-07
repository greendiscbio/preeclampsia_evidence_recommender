"""
Utility helpers for clinical review pipeline.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


def get_existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """
    Return only columns that exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : Iterable[str]
        Candidate columns.

    Returns
    -------
    List[str]
        Existing columns.
    """
    return [col for col in cols if col in df.columns]


def ensure_patient_id_column(
    df: pd.DataFrame,
    patient_id_col: str,
) -> pd.DataFrame:
    """
    Ensure the dataframe contains a patient identifier column.

    If missing, creates a sequential identifier based on row order.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    patient_id_col : str
        Name of patient id column.

    Returns
    -------
    pd.DataFrame
        Dataframe with guaranteed patient id column.
    """
    df_out = df.copy()

    if patient_id_col not in df_out.columns:
        df_out[patient_id_col] = list(range(len(df_out)))

    return df_out


def optionally_sample_dataframe(
    df: pd.DataFrame,
    sample_size: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Optionally sample a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sample_size : Optional[int]
        Number of rows to sample. If None, returns full dataframe.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled or original dataframe.

    Raises
    ------
    ValueError
        If sample_size is non-positive.
    """
    if sample_size is None:
        return df.copy()

    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    if sample_size >= len(df):
        return df.copy().reset_index(drop=True)

    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)