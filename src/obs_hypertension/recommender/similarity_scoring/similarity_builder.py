"""
Main similarity scoring builder.

This module computes:
- sim_struct
- sim_notes
- final similarity score

between population and protocol rows in the database.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.similarity_scoring.config import (
    DEFAULT_DB_ID_COL,
    DEFAULT_MAX_ANOMALY_SLOTS,
    DEFAULT_PATIENT_ID_COL,
    DEFAULT_PATIENT_NOTES_VEC_COL,
    DEFAULT_PROTOCOL_CLINICAL_VEC_COL,
    DEFAULT_STRUCTURED_VEC_COL,
    DEFAULT_W_NOTES,
    DEFAULT_W_STRUCT,
)
from src.obs_hypertension.recommender.similarity_scoring.protocol_builder import (
    build_protocol_notes_matrix,
)
from src.obs_hypertension.recommender.similarity_scoring.vector_utils import (
    as_float_matrix_from_list_column,
    cosine_sim_matrix,
)


def _validate_required_columns(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    structured_vec_col: str,
    patient_notes_vec_col: str,
    protocol_clinical_vec_col: str,
) -> None:
    """
    Validate required vector columns before similarity computation.
    """
    for column in (structured_vec_col, patient_notes_vec_col):
        if column not in synthetic_population.columns:
            raise ValueError(f"synthetic_population missing required column: '{column}'")

    for column in (structured_vec_col, protocol_clinical_vec_col):
        if column not in df_database.columns:
            raise ValueError(f"df_database missing required column: '{column}'")


def _validate_similarity_weights(w_struct: float, w_notes: float) -> None:
    """
    Validate similarity score weights.
    """
    if w_struct < 0 or w_notes < 0:
        raise ValueError("Similarity weights must be non-negative.")

    weight_sum = w_struct + w_notes
    if weight_sum <= 0:
        raise ValueError("At least one similarity weight must be positive.")


def _collect_identifiers(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    patient_id_col: str,
    db_id_col: Optional[str],
) -> Dict[str, Optional[list]]:
    """
    Collect patient and protocol identifiers.
    """
    patient_ids = (
        synthetic_population[patient_id_col].tolist()
        if patient_id_col in synthetic_population.columns
        else list(range(len(synthetic_population)))
    )

    protocol_ids = (
        df_database[db_id_col].tolist()
        if (db_id_col is not None and db_id_col in df_database.columns)
        else None
    )

    return {
        "patient_ids": patient_ids,
        "protocol_ids": protocol_ids,
    }


def compute_similarity_matrices_for_population(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    st_model: SentenceTransformer,
    patient_id_col: str = DEFAULT_PATIENT_ID_COL,
    db_id_col: Optional[str] = DEFAULT_DB_ID_COL,
    structured_vec_col: str = DEFAULT_STRUCTURED_VEC_COL,
    patient_notes_vec_col: str = DEFAULT_PATIENT_NOTES_VEC_COL,
    protocol_clinical_vec_col: str = DEFAULT_PROTOCOL_CLINICAL_VEC_COL,
    max_anomaly_slots: int = DEFAULT_MAX_ANOMALY_SLOTS,
    w_struct: float = DEFAULT_W_STRUCT,
    w_notes: float = DEFAULT_W_NOTES,
    st_batch_size: int = 64,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
) -> Dict[str, object]:
    """
    Compute similarity matrices between synthetic patients and protocol database.

    Similarity components
    ---------------------
    sim_structured = cosine(patient_structured, protocol_structured)
    sim_notes      = cosine(patient_notes, protocol_notes)

    final_similarity = w_struct * sim_structured + w_notes * sim_notes

    Parameters
    ----------
    synthetic_population : pd.DataFrame
        Synthetic patient dataframe.
    df_database : pd.DataFrame
        Protocol database dataframe.
    st_model : SentenceTransformer
        Embedding model used for anomaly note augmentation.
    patient_id_col : str
        Patient identifier column.
    db_id_col : Optional[str]
        Protocol identifier column.
    structured_vec_col : str
        Structured vector column shared by both datasets.
    patient_notes_vec_col : str
        Synthetic patient notes vector column.
    protocol_clinical_vec_col : str
        Protocol clinical note vector column.
    max_anomaly_slots : int
        Maximum anomaly slots to inspect for protocol augmentation.
    w_struct : float
        Weight for structured similarity.
    w_notes : float
        Weight for notes similarity.
    st_batch_size : int
        Batch size for anomaly name embeddings.
    normalize_embeddings : bool
        Whether to normalize anomaly embeddings.
    show_progress : bool
        Whether to print progress messages.

    Returns
    -------
    Dict[str, object]
        Dictionary containing:
        - similarity   : np.ndarray (P, K)
        - sim_struct   : np.ndarray (P, K)
        - sim_notes    : np.ndarray (P, K)
        - patient_ids  : list
        - protocol_ids : Optional[list]
    """
    _validate_required_columns(
        synthetic_population=synthetic_population,
        df_database=df_database,
        structured_vec_col=structured_vec_col,
        patient_notes_vec_col=patient_notes_vec_col,
        protocol_clinical_vec_col=protocol_clinical_vec_col,
    )

    _validate_similarity_weights(
        w_struct=w_struct,
        w_notes=w_notes,
    )

    if show_progress:
        print("🔹 Building protocol matrices (structured + notes)...")

    db_struct_matrix = as_float_matrix_from_list_column(
        df_database[structured_vec_col]
    )

    db_notes_matrix = build_protocol_notes_matrix(
        df_db=df_database,
        st_model=st_model,
        clinical_notes_vec_col=protocol_clinical_vec_col,
        max_anomaly_slots=max_anomaly_slots,
        st_batch_size=st_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress=show_progress,
    )

    if show_progress:
        print("🔹 Building patient matrices (structured + notes)...")

    syn_struct_matrix = as_float_matrix_from_list_column(
        synthetic_population[structured_vec_col],
        expected_dim=db_struct_matrix.shape[1],
    )

    syn_notes_matrix = as_float_matrix_from_list_column(
        synthetic_population[patient_notes_vec_col],
        expected_dim=db_notes_matrix.shape[1],
    )

    if show_progress:
        print("🔹 Computing cosine similarity matrices...")

    sim_struct = cosine_sim_matrix(syn_struct_matrix, db_struct_matrix)
    sim_notes = cosine_sim_matrix(syn_notes_matrix, db_notes_matrix)

    similarity = (w_struct * sim_struct) + (w_notes * sim_notes)

    identifiers = _collect_identifiers(
        synthetic_population=synthetic_population,
        df_database=df_database,
        patient_id_col=patient_id_col,
        db_id_col=db_id_col,
    )

    return {
        "similarity": similarity.astype("float32"),
        "sim_struct": sim_struct.astype("float32"),
        "sim_notes": sim_notes.astype("float32"),
        "patient_ids": identifiers["patient_ids"],
        "protocol_ids": identifiers["protocol_ids"],
    }