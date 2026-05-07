"""
Core risk scoring logic for protocol-level severity aggregation.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from src.obs_hypertension.recommender.risk_scoring.config import OUTCOME_TYPE_WEIGHTS
from src.obs_hypertension.recommender.risk_scoring.text_filters import is_relevant_risk_outcome
from src.obs_hypertension.recommender.risk_scoring.vector_utils import (
    as_numpy_vector,
    cosine_sim_unit,
    l2_normalize_vec,
)


def compute_outcome_severity_score(
    outcome_vector,
    tier_refs: Dict[str, Dict],
) -> float:
    """
    Score one outcome vector against all risk tier centroids.

    Severity is computed as:
        max_tier( cosine(outcome, tier_centroid) * tier_weight )
    """
    vector = as_numpy_vector(outcome_vector)
    if vector is None:
        return 0.0

    vector_unit = l2_normalize_vec(vector)

    best_score = 0.0
    for tier_cfg in tier_refs.values():
        similarity = cosine_sim_unit(vector_unit, tier_cfg["centroid"])
        weighted_score = similarity * float(tier_cfg["weight"])
        best_score = max(best_score, weighted_score)

    return float(best_score)


def compute_protocol_risk_score(
    row: pd.Series,
    tier_refs: Dict[str, Dict],
    outcome_vectors_row: Dict[str, Dict[str, list]],
    max_outcomes_per_type: int = 2,
) -> float:
    """
    Aggregate protocol-level risk from relevant maternal, fetal and anomaly outcomes.
    """
    total_risk = 0.0

    for family, family_weight in OUTCOME_TYPE_WEIGHTS.items():
        vectors_for_family = outcome_vectors_row.get(family, {}) or {}

        for idx in range(1, max_outcomes_per_type + 1):
            outcome_type = row.get(f"{family}_outcome_type_{idx}")
            outcome_value = row.get(f"{family}_outcome_value_{idx}")
            outcome_p_value = row.get(f"{family}_outcome_p_value_{idx}")

            if not is_relevant_risk_outcome(
                outcome_type=outcome_type,
                outcome_value=outcome_value,
                outcome_p_value=outcome_p_value,
            ):
                continue

            col_name = f"{family}_outcome_name_{idx}"
            outcome_vector = vectors_for_family.get(col_name)

            if outcome_vector is None:
                continue

            severity = compute_outcome_severity_score(
                outcome_vector=outcome_vector,
                tier_refs=tier_refs,
            )
            total_risk += severity * float(family_weight)

    return float(total_risk)