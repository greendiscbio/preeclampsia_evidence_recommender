"""
Embedding utilities for recommender vectorization.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.vectorization.text_utils import clean_text_series


def load_sentence_transformer(
    model_name: str,
    model_dir: Optional[str] = None,
    show_progress: bool = True,
) -> SentenceTransformer:
    """
    Load SentenceTransformer either from local directory or model name.
    """
    if model_dir:
        if show_progress:
            print(f"Loading SentenceTransformer from local directory: {model_dir}")
        return SentenceTransformer(model_dir)

    if show_progress:
        print(f"Initializing SentenceTransformer from model hub: {model_name}")
    return SentenceTransformer(model_name)


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 normalize embeddings row-wise.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + eps
    return matrix / norms


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts into embeddings.
    """
    if len(texts) == 0:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    if normalize:
        embeddings = l2_normalize(embeddings).astype("float32")

    return embeddings


def embed_text_column_per_row(
    df: pd.DataFrame,
    model: SentenceTransformer,
    source_col: str,
    output_vec_col: str,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Embed one text column into one vector column per row.
    Empty rows become zero-vectors.
    Missing source column also becomes zero-vectors.
    """
    df_out = df.copy()
    dim = model.get_sentence_embedding_dimension()
    zero_vector = np.zeros((dim,), dtype="float32")

    if source_col not in df_out.columns:
        df_out[output_vec_col] = [zero_vector.tolist()] * len(df_out)
        return df_out

    temp_clean_col = f"__{source_col}_clean__"
    df_out[temp_clean_col] = clean_text_series(df_out[source_col])

    mask_has_text = df_out[temp_clean_col].str.len() > 0
    vectors = np.zeros((len(df_out), dim), dtype="float32")

    if mask_has_text.any():
        row_idx = np.where(mask_has_text.values)[0]
        texts = df_out.loc[mask_has_text, temp_clean_col].tolist()

        embeddings = encode_texts(
            model=model,
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress,
        )
        vectors[row_idx] = embeddings

    df_out[output_vec_col] = [vec.tolist() for vec in vectors]
    df_out.drop(columns=[temp_clean_col], inplace=True, errors="ignore")

    return df_out