"""
Protocol note matrix builder for similarity scoring.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.similarity_scoring.anomaly_utils import (
    extract_profile_anomaly_names_for_row,
)
from src.obs_hypertension.recommender.similarity_scoring.vector_utils import (
    as_float_matrix_from_list_column,
    l2_normalize_rows,
)
from src.obs_hypertension.recommender.vectorization.embedding_utils import encode_texts


def build_protocol_notes_matrix(
    df_db: pd.DataFrame,
    st_model: SentenceTransformer,
    clinical_notes_vec_col: str = "clinical_notes_vector",
    max_anomaly_slots: int = 3,
    st_batch_size: int = 64,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Build protocol notes matrix for similarity scoring.

    Logic
    -----
    protocol_notes_vector =
        clinical_notes_vector
        + mean(embeddings(relevant anomaly_outcome_name_i))

    Relevant anomaly names are added only when:
    - anomaly_outcome_type_i == "profile"
    - anomaly_outcome_value_i is relevant
    - anomaly_outcome_p_value_i is relevant

    Parameters
    ----------
    df_db : pd.DataFrame
        Database dataframe.
    st_model : SentenceTransformer
        Embedding model.
    clinical_notes_vec_col : str
        Column containing protocol clinical note vectors.
    max_anomaly_slots : int
        Maximum number of anomaly slots to inspect.
    st_batch_size : int
        Batch size for embedding anomaly names.
    normalize_embeddings : bool
        Whether to normalize anomaly embeddings.
    show_progress : bool
        Whether to print progress messages.

    Returns
    -------
    np.ndarray
        Protocol notes matrix of shape (K, D).
    """
    base_matrix = as_float_matrix_from_list_column(df_db[clinical_notes_vec_col])
    embedding_dim = st_model.get_sentence_embedding_dimension()

    per_row_names: List[List[str]] = []
    unique_names = set()

    for _, row in df_db.iterrows():
        row_names = extract_profile_anomaly_names_for_row(
            row=row,
            max_k=max_anomaly_slots,
        )
        per_row_names.append(row_names)
        for name in row_names:
            unique_names.add(name)

    unique_names = sorted(unique_names)

    if show_progress:
        print(
            f"🔹 Protocol notes augmentation: "
            f"{len(unique_names)} unique relevant profile-anomaly names"
        )

    cache: Dict[str, np.ndarray] = {}

    if unique_names:
        vectors = encode_texts(
            model=st_model,
            texts=unique_names,
            batch_size=st_batch_size,
            normalize=normalize_embeddings,
            show_progress_bar=show_progress,
        )
        cache = {text: vectors[idx] for idx, text in enumerate(unique_names)}

    notes_matrix = base_matrix.copy()

    for row_idx, names in enumerate(per_row_names):
        if not names:
            continue

        row_vectors = [cache[name] for name in names if name in cache]
        if not row_vectors:
            continue

        addition = np.mean(np.vstack(row_vectors), axis=0).astype("float32")

        if addition.shape[0] != embedding_dim:
            continue

        notes_matrix[row_idx, :] = notes_matrix[row_idx, :] + addition

    notes_matrix = l2_normalize_rows(notes_matrix).astype("float32")
    return notes_matrix