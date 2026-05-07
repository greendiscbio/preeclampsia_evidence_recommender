"""
Text filtering helpers for protocol risk scoring.
"""

from __future__ import annotations

import numpy as np

from src.obs_hypertension.recommender.risk_scoring.config import (
    NON_SIGNIFICANT_MARKERS,
    NULLISH,
    RISK_TYPE_ALIASES,
)


def normalize_text(value) -> str:
    """
    Normalize text safely for rule-based checks.
    """
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip().lower()


def is_nullish_text(value) -> bool:
    """
    Check whether a text value is nullish.
    """
    normalized = normalize_text(value)
    return normalized in NULLISH or normalized == ""


def contains_non_significant_marker(value) -> bool:
    """
    Check whether a text contains a non-significance marker.
    """
    normalized = normalize_text(value)

    if not normalized:
        return True

    return any(marker in normalized for marker in NON_SIGNIFICANT_MARKERS)


def is_relevant_risk_outcome(
    outcome_type,
    outcome_value,
    outcome_p_value,
) -> bool:
    """
    Keep only outcomes that:
    - have a risk-compatible type
    - have relevant value text
    - have relevant p-value text
    """
    normalized_type = normalize_text(outcome_type)

    if normalized_type not in RISK_TYPE_ALIASES:
        return False

    if is_nullish_text(outcome_value) or contains_non_significant_marker(outcome_value):
        return False

    if is_nullish_text(outcome_p_value) or contains_non_significant_marker(outcome_p_value):
        return False

    return True