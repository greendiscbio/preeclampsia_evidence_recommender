"""
Post-processing utilities for synthetic population.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.obs_hypertension.synthetic_population.config import (
    AGE_OH,
    DBP_OH,
    DEFAULT_OUTPUT_COLUMNS,
    GA_OH,
    SBP_OH,
)


AGE_VALUE_MAP: Dict[str, int] = {
    "age_18_20": 19,
    "age_21_23": 22,
    "age_24_26": 25,
    "age_27_29": 28,
    "age_30_32": 31,
    "age_33_35": 34,
    "age_36_38": 37,
    "age_39_41": 40,
    "age_42_44": 43,
}

GA_VALUE_MAP: Dict[str, int] = {
    "<20_weeks": 18,
    "20_33_weeks": 28,
    ">33_weeks": 36,
}

SBP_VALUE_MAP: Dict[str, int] = {
    "<119_SBP": 115,
    "120_129_SBP": 125,
    "130_139_SBP": 135,
    "140_160_SBP": 150,
    ">161_SBP": 170,
}

DBP_VALUE_MAP: Dict[str, int] = {
    "<85_DBP": 80,
    "86_90_DBP": 88,
    "91_105_DBP": 98,
    ">106_DBP": 110,
}


def recover_selected_category(df: pd.DataFrame, one_hot_cols: List[str], output_col: str) -> pd.DataFrame:
    """
    Recover the selected category from a one-hot encoded group.
    """
    df_out = df.copy()
    df_out[output_col] = df_out[one_hot_cols].idxmax(axis=1)
    return df_out


def add_numeric_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add representative numeric proxy values for age, gestational age, SBP and DBP.
    """
    df_out = df.copy()

    df_out = recover_selected_category(df_out, AGE_OH, "Age_category")
    df_out = recover_selected_category(df_out, GA_OH, "GA_category")
    df_out = recover_selected_category(df_out, SBP_OH, "SBP_category")
    df_out = recover_selected_category(df_out, DBP_OH, "DBP_category")

    df_out["Age_value"] = df_out["Age_category"].map(AGE_VALUE_MAP)
    df_out["GA_value"] = df_out["GA_category"].map(GA_VALUE_MAP)
    df_out["SBP_value"] = df_out["SBP_category"].map(SBP_VALUE_MAP)
    df_out["DBP_value"] = df_out["DBP_category"].map(DBP_VALUE_MAP)

    return df_out


def reorder_population_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder dataframe columns to a standard export schema.
    """
    existing_cols = [col for col in DEFAULT_OUTPUT_COLUMNS if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]
    return df[existing_cols + remaining_cols]


def postprocess_population(
    df: pd.DataFrame,
    include_numeric_proxies: bool = True,
    include_one_hot_columns: bool = True,
) -> pd.DataFrame:
    """
    Apply postprocessing pipeline to synthetic population dataframe.
    """
    df_out = df.copy()

    if include_numeric_proxies:
        df_out = add_numeric_proxy_columns(df_out)

    if not include_one_hot_columns:
        cols_to_drop = AGE_OH + GA_OH + SBP_OH + DBP_OH
        df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    df_out = reorder_population_columns(df_out)

    return df_out