"""
Outcome-specific vectorization utilities.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.vectorization.embedding_utils import encode_texts
from src.obs_hypertension.recommender.vectorization.text_utils import clean_text_one


def detect_outcome_name_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Detect ordered outcome name columns like:
    - maternal_outcome_name_1
    - fetal_outcome_name_2
    - anomaly_outcome_name_3
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
    cols = [col for col in df.columns if pattern.match(col)]

    def suffix_as_int(col_name: str) -> int:
        match = re.search(r"(\d+)$", col_name)
        return int(match.group(1)) if match else 999999

    return sorted(cols, key=suffix_as_int)


def embed_outcomes_per_column(
    df: pd.DataFrame,
    model: SentenceTransformer,
    outcome_cols: List[str],
    batch_size: int = 64,
    normalize: bool = True,
    show_progress: bool = True,
) -> List[Dict[str, List[float]]]:
    """
    Embed outcome names independently per column.
    Nullish outcome names are assigned zero-vectors.
    """
    dim = model.get_sentence_embedding_dimension()
    zero_vector = np.zeros((dim,), dtype="float32")

    if not outcome_cols:
        return [{} for _ in range(len(df))]

    unique_texts = set()
    per_row_clean_texts = []

    outcome_df = df[outcome_cols].copy()

    for _, row in outcome_df.iterrows():
        current_row_clean = []
        for col in outcome_cols:
            cleaned_text = clean_text_one(row.get(col, ""))
            current_row_clean.append(cleaned_text)
            if cleaned_text:
                unique_texts.add(cleaned_text)
        per_row_clean_texts.append(current_row_clean)

    unique_texts = sorted(unique_texts)

    if show_progress:
        print(f"Outcome columns: {len(outcome_cols)} | unique cleaned texts: {len(unique_texts)}")

    if unique_texts:
        unique_vectors = encode_texts(
            model=model,
            texts=unique_texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress,
        )
        cache = {text: unique_vectors[i] for i, text in enumerate(unique_texts)}
    else:
        cache = {}

    output = []
    for row_clean in per_row_clean_texts:
        row_dict = {}
        for col, text in zip(outcome_cols, row_clean):
            row_dict[col] = zero_vector.tolist() if not text else cache[text].tolist()
        output.append(row_dict)

    return output