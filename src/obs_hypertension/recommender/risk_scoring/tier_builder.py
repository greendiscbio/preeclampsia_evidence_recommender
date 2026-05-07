"""
Tier centroid builder for protocol risk scoring.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.obs_hypertension.recommender.risk_scoring.config import RISK_TIERS
from src.obs_hypertension.recommender.risk_scoring.vector_utils import l2_normalize_vec


def build_risk_tier_reference_embeddings(
    st_model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build reference embeddings for risk tiers from phrase centroids.

    Each phrase embedding is L2-normalized by the encoder.
    The final centroid is also L2-normalized after averaging.
    """
    references: Dict[str, Dict[str, np.ndarray]] = {}

    for tier_name, tier_cfg in RISK_TIERS.items():
        phrase_vectors = st_model.encode(
            tier_cfg["phrases"],
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        centroid = phrase_vectors.mean(axis=0).astype("float32")
        centroid = l2_normalize_vec(centroid)

        references[tier_name] = {
            "weight": float(tier_cfg["weight"]),
            "centroid": centroid,
        }

    return references