"""
Main vectorization module for recommender model.

This module builds:
- structured_profile_vector
- clinical_notes_vector
- synthetic anomaly_1_vector
- synthetic general_notes_vector
- outcome_name_vectors_maternal
- outcome_name_vectors_fetal
- outcome_name_vectors_anomaly
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.vectorization.config import (
    DEFAULT_DB_CLINICAL_TEXT_COLS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAVE_PREFIX,
    DEFAULT_ST_MODEL_NAME,
    DEFAULT_SYN_ANOMALY_1_COL,
    DEFAULT_SYN_CLINICAL_TEXT_COL,
)
from src.obs_hypertension.recommender.vectorization.embedding_utils import (
    embed_text_column_per_row,
    encode_texts,
    load_sentence_transformer,
)
from src.obs_hypertension.recommender.vectorization.outcome_vectorizer import (
    detect_outcome_name_cols,
    embed_outcomes_per_column,
)
from src.obs_hypertension.recommender.vectorization.text_utils import (
    clean_text_series,
    ensure_columns,
    safe_concat_text_cols,
)


def _build_structured_vectors(
    df: pd.DataFrame,
    structured_cols: List[str],
) -> pd.DataFrame:
    """
    Build structured_profile_vector from ordered structured columns.
    """
    df_out = df.copy()
    df_out = ensure_columns(df_out, structured_cols, fill_value=0)

    structured_matrix = df_out[structured_cols].astype("float32").to_numpy()
    df_out["structured_profile_vector"] = [row.tolist() for row in structured_matrix]

    return df_out


def _build_db_clinical_notes_vector(
    df: pd.DataFrame,
    model: SentenceTransformer,
    db_clinical_text_cols: List[str],
    batch_size: int,
    normalize_embeddings: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """
    Build clinical_notes_vector for database dataframe from selected DB text columns.
    """
    df_out = df.copy()

    df_out["_clinical_notes_raw"] = safe_concat_text_cols(df_out, list(db_clinical_text_cols), sep=" ")
    df_out["_clinical_notes_clean"] = clean_text_series(df_out["_clinical_notes_raw"])

    embeddings = encode_texts(
        model=model,
        texts=df_out["_clinical_notes_clean"].tolist(),
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress_bar=show_progress,
    )
    df_out["clinical_notes_vector"] = [vec.tolist() for vec in embeddings]

    return df_out


def _build_syn_clinical_notes_vector(
    df: pd.DataFrame,
    model: SentenceTransformer,
    syn_clinical_text_col: str,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """
    Build clinical_notes_vector for synthetic population.
    """
    df_out = df.copy()

    if syn_clinical_text_col in df_out.columns:
        df_out["_clinical_notes_raw"] = df_out[syn_clinical_text_col].fillna("").astype(str)
    else:
        df_out["_clinical_notes_raw"] = ""

    df_out["_clinical_notes_clean"] = clean_text_series(df_out["_clinical_notes_raw"])

    embeddings = encode_texts(
        model=model,
        texts=df_out["_clinical_notes_clean"].tolist(),
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress_bar=show_progress,
    )
    df_out["clinical_notes_vector"] = [vec.tolist() for vec in embeddings]

    return df_out


def _build_syn_general_notes_vector(
    df: pd.DataFrame,
    model: SentenceTransformer,
    syn_anomaly_1_col: str,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """
    Build general_notes_vector from synthetic clinical notes + anomaly_1.
    """
    df_out = df.copy()

    anomaly_text = (
        df_out[syn_anomaly_1_col].fillna("").astype(str)
        if syn_anomaly_1_col in df_out.columns
        else ""
    )

    df_out["_general_notes_raw"] = (
        df_out["_clinical_notes_raw"].fillna("").astype(str)
        + " "
        + anomaly_text
    )
    df_out["_general_notes_clean"] = clean_text_series(df_out["_general_notes_raw"])

    embeddings = encode_texts(
        model=model,
        texts=df_out["_general_notes_clean"].tolist(),
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress_bar=show_progress,
    )
    df_out["general_notes_vector"] = [vec.tolist() for vec in embeddings]

    return df_out


def _add_outcome_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
    normalize_embeddings: bool,
    show_progress: bool,
) -> pd.DataFrame:
    """
    Add outcome embeddings per column for maternal, fetal and anomaly outcomes.
    """
    df_out = df.copy()

    maternal_cols = detect_outcome_name_cols(df_out, "maternal_outcome_name_")
    fetal_cols = detect_outcome_name_cols(df_out, "fetal_outcome_name_")
    anomaly_cols = detect_outcome_name_cols(df_out, "anomaly_outcome_name_")

    if show_progress:
        print("  - Maternal outcome columns:", maternal_cols)
        print("  - Fetal outcome columns:", fetal_cols)
        print("  - Anomaly outcome columns:", anomaly_cols)

    df_out["outcome_name_vectors_maternal"] = embed_outcomes_per_column(
        df=df_out,
        model=model,
        outcome_cols=maternal_cols,
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress=show_progress,
    ) if maternal_cols else [{} for _ in range(len(df_out))]

    df_out["outcome_name_vectors_fetal"] = embed_outcomes_per_column(
        df=df_out,
        model=model,
        outcome_cols=fetal_cols,
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress=show_progress,
    ) if fetal_cols else [{} for _ in range(len(df_out))]

    df_out["outcome_name_vectors_anomaly"] = embed_outcomes_per_column(
        df=df_out,
        model=model,
        outcome_cols=anomaly_cols,
        batch_size=batch_size,
        normalize=normalize_embeddings,
        show_progress=show_progress,
    ) if anomaly_cols else [{} for _ in range(len(df_out))]

    return df_out


def _save_vectorizer_artifacts(
    model: SentenceTransformer,
    structured_cols: List[str],
    db_clinical_text_cols: List[str],
    syn_clinical_text_col: str,
    syn_anomaly_1_col: str,
    embed_outcomes: bool,
    df_db: pd.DataFrame,
    st_model_name: str,
    output_dir: str,
    save_prefix: str,
    requested_model_dir: Optional[str] = None,
    show_progress: bool = True,
) -> None:
    """
    Save sentence transformer and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    st_save_dir = requested_model_dir or os.path.join(
        output_dir,
        f"{save_prefix}_sentence_transformer",
    )
    model.save(st_save_dir)

    metadata: Dict[str, Any] = {
        "structured_cols": structured_cols,
        "db_clinical_text_cols": list(db_clinical_text_cols),
        "syn_clinical_text_col": syn_clinical_text_col,
        "syn_anomaly_1_col": syn_anomaly_1_col,
        "embed_outcomes": embed_outcomes,
        "maternal_outcome_cols": detect_outcome_name_cols(df_db, "maternal_outcome_name_") if embed_outcomes else [],
        "fetal_outcome_cols": detect_outcome_name_cols(df_db, "fetal_outcome_name_") if embed_outcomes else [],
        "anomaly_outcome_cols": detect_outcome_name_cols(df_db, "anomaly_outcome_name_") if embed_outcomes else [],
        "st_model_name": st_model_name,
        "st_model_dir": st_save_dir,
    }

    meta_path = os.path.join(output_dir, f"{save_prefix}_vectorizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    if show_progress:
        print(f"💾 Saved SentenceTransformer to: {st_save_dir}")
        print(f"💾 Saved vectorizer metadata to: {meta_path}")


def _cleanup_temp_columns(df: pd.DataFrame, temp_cols: List[str]) -> pd.DataFrame:
    """
    Remove temporary columns if present.
    """
    df_out = df.copy()
    df_out.drop(columns=[col for col in temp_cols if col in df_out.columns], inplace=True, errors="ignore")
    return df_out


def vectorize_recommender_inputs(
    df_database: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    structured_cols: List[str],
    st_model_name: str = DEFAULT_ST_MODEL_NAME,
    st_model_dir: Optional[str] = None,
    st_batch_size: int = 64,
    normalize_embeddings: bool = True,
    db_clinical_text_cols: Tuple[str, ...] = DEFAULT_DB_CLINICAL_TEXT_COLS,
    syn_clinical_text_col: str = DEFAULT_SYN_CLINICAL_TEXT_COL,
    syn_anomaly_1_col: str = DEFAULT_SYN_ANOMALY_1_COL,
    embed_outcomes: bool = True,
    save_model: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_prefix: str = DEFAULT_SAVE_PREFIX,
    copy_dfs: bool = True,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, SentenceTransformer]:
    """
    Main vectorization function for recommender inputs.

    Parameters
    ----------
    df_database : pd.DataFrame
        Standardized arm-level database.
    synthetic_population : pd.DataFrame
        Synthetic patient population.
    structured_cols : List[str]
        Ordered structured columns used to build structured vectors.
    st_model_name : str
        SentenceTransformer model name.
    st_model_dir : Optional[str]
        Optional local directory of a pre-saved SentenceTransformer model.
    st_batch_size : int
        Embedding batch size.
    normalize_embeddings : bool
        Whether to L2-normalize embeddings.
    db_clinical_text_cols : Tuple[str, ...]
        DB columns used to build clinical text.
    syn_clinical_text_col : str
        Synthetic clinical notes column.
    syn_anomaly_1_col : str
        Synthetic anomaly text column.
    embed_outcomes : bool
        Whether to embed DB outcome name columns.
    save_model : bool
        Whether to persist model and metadata.
    output_dir : str
        Directory where vectorizer artifacts will be saved.
    save_prefix : str
        Prefix used for artifact naming.
    copy_dfs : bool
        Whether to work on dataframe copies.
    show_progress : bool
        Whether to print progress messages.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, SentenceTransformer]
        Vectorized DB dataframe, vectorized synthetic dataframe, and loaded model.
    """
    df_db = df_database.copy() if copy_dfs else df_database
    df_syn = synthetic_population.copy() if copy_dfs else synthetic_population

    if show_progress:
        print("🔹 Starting recommender vectorization pipeline...")

    model = load_sentence_transformer(
        model_name=st_model_name,
        model_dir=st_model_dir,
        show_progress=show_progress,
    )

    if show_progress:
        print("🔹 Building structured_profile_vector for DB and synthetic population...")

    df_db = _build_structured_vectors(df_db, structured_cols)
    df_syn = _build_structured_vectors(df_syn, structured_cols)

    if show_progress:
        print("🔹 Encoding clinical notes for DB...")
    df_db = _build_db_clinical_notes_vector(
        df=df_db,
        model=model,
        db_clinical_text_cols=list(db_clinical_text_cols),
        batch_size=st_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress=show_progress,
    )

    if show_progress:
        print("🔹 Encoding clinical notes for synthetic population...")
    df_syn = _build_syn_clinical_notes_vector(
        df=df_syn,
        model=model,
        syn_clinical_text_col=syn_clinical_text_col,
        batch_size=st_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress=show_progress,
    )

    if show_progress:
        print("🔹 Encoding synthetic anomaly_1_vector...")
    df_syn = embed_text_column_per_row(
        df=df_syn,
        model=model,
        source_col=syn_anomaly_1_col,
        output_vec_col="anomaly_1_vector",
        batch_size=st_batch_size,
        normalize=normalize_embeddings,
        show_progress=show_progress,
    )

    if show_progress:
        print("🔹 Encoding synthetic general_notes_vector...")
    df_syn = _build_syn_general_notes_vector(
        df=df_syn,
        model=model,
        syn_anomaly_1_col=syn_anomaly_1_col,
        batch_size=st_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress=show_progress,
    )

    if embed_outcomes:
        if show_progress:
            print("🔹 Encoding DB outcome names per column...")
        df_db = _add_outcome_embeddings(
            df=df_db,
            model=model,
            batch_size=st_batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress=show_progress,
        )

    if save_model:
        _save_vectorizer_artifacts(
            model=model,
            structured_cols=structured_cols,
            db_clinical_text_cols=list(db_clinical_text_cols),
            syn_clinical_text_col=syn_clinical_text_col,
            syn_anomaly_1_col=syn_anomaly_1_col,
            embed_outcomes=embed_outcomes,
            df_db=df_db,
            st_model_name=st_model_name,
            output_dir=output_dir,
            save_prefix=save_prefix,
            requested_model_dir=st_model_dir,
            show_progress=show_progress,
        )

    df_db = _cleanup_temp_columns(
        df_db,
        temp_cols=[
            "_clinical_notes_raw",
            "_clinical_notes_clean",
        ],
    )

    df_syn = _cleanup_temp_columns(
        df_syn,
        temp_cols=[
            "_clinical_notes_raw",
            "_clinical_notes_clean",
            "_general_notes_raw",
            "_general_notes_clean",
        ],
    )

    if show_progress:
        print("✅ Vectorization complete.")
        print("DB vectors created: structured_profile_vector, clinical_notes_vector, outcome_name_vectors_*")
        print("Synthetic vectors created: structured_profile_vector, clinical_notes_vector, anomaly_1_vector, general_notes_vector")

    return df_db, df_syn, model