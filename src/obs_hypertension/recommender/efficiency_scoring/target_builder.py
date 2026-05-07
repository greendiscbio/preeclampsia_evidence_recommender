"""
Target definition logic for protocol efficiency scoring.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.obs_hypertension.recommender.efficiency_scoring.config import NULLISH
from src.obs_hypertension.recommender.efficiency_scoring.text_utils import (
    is_yes,
    normalize_text,
    safe_lower_series,
)


def define_efficiency_target(
    df: pd.DataFrame,
    sig_col: str = "statistical_significance_vs_control",
    is_comb_col: str = "is_combination",
    drug1_col: str = "drug_1__stdcat",
    drug2_col: str = "drug_2__stdcat",
    winner1_col: str = "winner_1__stdcat",
    winner2_col: str = "winner_2__stdcat",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Define protocol-level efficiency target.

    Logic
    -----
    A row is considered efficient if:
    - statistical_significance_vs_control is yes-like
    - mono-therapy: winner_1__stdcat == drug_1__stdcat
    - combination therapy:
        winner_1__stdcat == drug_1__stdcat
        and
        winner_2__stdcat == drug_2__stdcat

    Rows with missing/unknown significance are marked as ambiguous.

    Parameters
    ----------
    df : pd.DataFrame
        Input protocol dataframe.
    sig_col : str
        Significance column.
    is_comb_col : str
        Combination flag column.
    drug1_col : str
        First treatment drug column.
    drug2_col : str
        Second treatment drug column.
    winner1_col : str
        First winner drug column.
    winner2_col : str
        Second winner drug column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        - dataframe copy with Is_Efficient column
        - ambiguous mask for missing/unknown significance
    """
    df_out = df.copy()

    sig_series = df_out.get(sig_col, pd.Series([None] * len(df_out), index=df_out.index))
    sig_normalized = sig_series.map(normalize_text)
    ambiguous_mask = sig_normalized.isin(NULLISH)
    sig_yes = sig_series.map(is_yes)

    drug1 = safe_lower_series(df_out.get(drug1_col, pd.Series([""] * len(df_out), index=df_out.index)))
    drug2 = safe_lower_series(df_out.get(drug2_col, pd.Series([""] * len(df_out), index=df_out.index)))
    winner1 = safe_lower_series(df_out.get(winner1_col, pd.Series([""] * len(df_out), index=df_out.index)))
    winner2 = safe_lower_series(df_out.get(winner2_col, pd.Series([""] * len(df_out), index=df_out.index)))

    is_combination = df_out.get(is_comb_col, 0)
    is_combination = pd.to_numeric(is_combination, errors="coerce").fillna(0).astype(int)

    mono_mask = (
        (is_combination == 0)
        & sig_yes
        & (winner1 != "")
        & (drug1 != "")
        & (winner1 == drug1)
    )

    combo_mask = (
        (is_combination == 1)
        & sig_yes
        & (winner1 != "")
        & (drug1 != "")
        & (winner1 == drug1)
        & (winner2 != "")
        & (drug2 != "")
        & (winner2 == drug2)
    )

    df_out["Is_Efficient"] = 0
    df_out.loc[mono_mask | combo_mask, "Is_Efficient"] = 1

    return df_out, ambiguous_mask