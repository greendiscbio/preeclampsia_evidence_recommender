"""
Main builder for protocol-level risk scoring.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.obs_hypertension.recommender.risk_scoring.config import (
    DEFAULT_MAX_OUTCOMES_PER_TYPE,
    DEFAULT_ST_BATCH_SIZE,
)
from src.obs_hypertension.recommender.risk_scoring.risk_core import compute_protocol_risk_score
from src.obs_hypertension.recommender.risk_scoring.tier_builder import (
    build_risk_tier_reference_embeddings,
)


def _validate_risk_inputs(
    df_database: pd.DataFrame,
    df_bbdd_vec: pd.DataFrame,
) -> None:
    """
    Validate risk-scoring inputs.
    """
    if len(df_database) != len(df_bbdd_vec):
        raise ValueError(
            "df_database and df_bbdd_vec must have the same number of rows and aligned indices."
        )

    required_vector_cols = [
        "outcome_name_vectors_maternal",
        "outcome_name_vectors_fetal",
        "outcome_name_vectors_anomaly",
    ]

    missing_cols = [col for col in required_vector_cols if col not in df_bbdd_vec.columns]
    if missing_cols:
        raise ValueError(
            f"df_bbdd_vec is missing required outcome vector columns: {missing_cols}"
        )


def _extract_outcome_vector_payload(
    vec_row: pd.Series,
) -> Dict[str, Dict[str, list]]:
    """
    Extract nested outcome vector dictionaries from one vectorized protocol row.
    """
    return {
        "maternal": vec_row.get("outcome_name_vectors_maternal", {}) or {},
        "fetal": vec_row.get("outcome_name_vectors_fetal", {}) or {},
        "anomaly": vec_row.get("outcome_name_vectors_anomaly", {}) or {},
    }


def compute_protocol_risk_scores(
    df_database: pd.DataFrame,
    df_bbdd_vec: pd.DataFrame,
    st_model: SentenceTransformer,
    max_outcomes_per_type: int = DEFAULT_MAX_OUTCOMES_PER_TYPE,
    st_batch_size: int = DEFAULT_ST_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute one protocol-level risk score per database row.

    Parameters
    ----------
    df_database : pd.DataFrame
        Original protocol dataframe.
    df_bbdd_vec : pd.DataFrame
        Vectorized protocol dataframe containing outcome_name_vectors_* columns.
    st_model : SentenceTransformer
        Embedding model used to build risk tier centroids.
    max_outcomes_per_type : int
        Maximum number of outcomes inspected per family.
    st_batch_size : int
        Batch size for tier phrase encoding.
    show_progress : bool
        Whether to show progress bar.

    Returns
    -------
    np.ndarray
        Risk score vector of shape (K,) aligned with database rows.
    """
    _validate_risk_inputs(
        df_database=df_database,
        df_bbdd_vec=df_bbdd_vec,
    )

    tier_refs = build_risk_tier_reference_embeddings(
        st_model=st_model,
        batch_size=st_batch_size,
        show_progress=False,
    )

    scores = np.zeros(len(df_database), dtype="float32")

    iterator = (
        tqdm(range(len(df_database)), desc="Computing protocol risk")
        if show_progress
        else range(len(df_database))
    )

    for idx in iterator:
        row = df_database.iloc[idx]
        vec_row = df_bbdd_vec.iloc[idx]

        outcome_vectors_row = _extract_outcome_vector_payload(vec_row)

        scores[idx] = compute_protocol_risk_score(
            row=row,
            tier_refs=tier_refs,
            outcome_vectors_row=outcome_vectors_row,
            max_outcomes_per_type=max_outcomes_per_type,
        )

    return scores