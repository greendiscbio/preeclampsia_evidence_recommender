"""
Builders for patient review and clinical recommendation tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.obs_hypertension.recommender.clinical_review.config import ClinicalReviewConfig
from src.obs_hypertension.recommender.clinical_review.utils import (
    ensure_patient_id_column,
    get_existing_cols,
)


def build_patient_review_df(
    synthetic_population: pd.DataFrame,
    config: ClinicalReviewConfig = ClinicalReviewConfig(),
) -> pd.DataFrame:
    """
    Build patient profile review table for clinical validation.

    Parameters
    ----------
    synthetic_population : pd.DataFrame
        Synthetic patient population.
    config : ClinicalReviewConfig
        Display configuration.

    Returns
    -------
    pd.DataFrame
        Patient review dataframe.
    """
    df = ensure_patient_id_column(
        df=synthetic_population,
        patient_id_col=config.patient_id_col,
    )

    out = pd.DataFrame(index=df.index)

    for col in config.patient_display_cols:
        out[col] = df[col] if col in df.columns else np.nan

    return out.reset_index(drop=True)


def build_clinical_recommendations_df(
    recs_df: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    config: ClinicalReviewConfig = ClinicalReviewConfig(),
) -> pd.DataFrame:
    """
    Build final recommendation table for clinical validation.

    Notes
    -----
    This function assumes recs_df already contains the correct treatment columns.
    Therefore, no merge with df_database is needed.

    Parameters
    ----------
    recs_df : pd.DataFrame
        Final or filtered recommendations dataframe.
    synthetic_population : pd.DataFrame
        Synthetic patient population.
    config : ClinicalReviewConfig
        Display configuration.

    Returns
    -------
    pd.DataFrame
        Clinical recommendations dataframe.
    """
    if recs_df is None or recs_df.empty:
        return pd.DataFrame()

    recs = recs_df.copy()
    syn = ensure_patient_id_column(
        df=synthetic_population,
        patient_id_col=config.patient_id_col,
    )

    patient_cols = get_existing_cols(syn, config.patient_display_cols)
    syn_view = syn[patient_cols].copy()

    merged = recs.merge(
        syn_view,
        how="left",
        on=config.patient_id_col,
    )

    final_columns = (
        list(config.patient_display_cols)
        + [col for col in config.treatment_display_cols if col in merged.columns]
        + [col for col in config.score_cols if col in merged.columns]
    )

    out = merged[final_columns].copy()

    if "final_score" in out.columns:
        out["final_score"] = (
            pd.to_numeric(out["final_score"], errors="coerce")
            .round(config.score_round)
        )

    if config.patient_id_col in out.columns and "rank" in out.columns:
        out = out.sort_values(
            by=[config.patient_id_col, "rank"],
            ascending=[True, True],
        )

    return out.reset_index(drop=True)