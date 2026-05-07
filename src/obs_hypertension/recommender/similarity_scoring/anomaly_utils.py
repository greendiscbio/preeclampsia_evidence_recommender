"""
Anomaly-related helpers for protocol notes augmentation.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.obs_hypertension.recommender.similarity_scoring.config import (
    NON_SIGNIFICANT_MARKERS,
    NULLISH,
)


def normalize_text(value) -> str:
    """
    Normalize text safely for rule-based checks.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


def is_nullish_text(value) -> bool:
    """
    Check whether text is nullish.
    """
    normalized = normalize_text(value)
    return (normalized in NULLISH) or (normalized == "")


def contains_non_significant_marker(value) -> bool:
    """
    Check whether text indicates a non-significant result.
    """
    normalized = normalize_text(value)

    if not normalized:
        return True

    for marker in NON_SIGNIFICANT_MARKERS:
        if marker in normalized:
            return True

    return False


def extract_profile_anomaly_names_for_row(
    row: pd.Series,
    max_k: int = 3,
    type_prefix: str = "anomaly_outcome_type_",
    name_prefix: str = "anomaly_outcome_name_",
    value_prefix: str = "anomaly_outcome_value_",
    pval_prefix: str = "anomaly_outcome_p_value_",
) -> List[str]:
    """
    Extract relevant anomaly outcome names for profile-type anomalies.

    Inclusion rules
    ---------------
    anomaly_outcome_name_i is included only if:
    - anomaly_outcome_type_i == "profile"
    - anomaly_outcome_value_i is relevant
    - anomaly_outcome_p_value_i is relevant

    Parameters
    ----------
    row : pd.Series
        Protocol row.
    max_k : int
        Maximum number of anomaly slots to inspect.

    Returns
    -------
    List[str]
        Relevant anomaly names.
    """
    names: List[str] = []

    for idx in range(1, max_k + 1):
        anomaly_type = row.get(f"{type_prefix}{idx}", None)
        if normalize_text(anomaly_type) != "profile":
            continue

        anomaly_value = row.get(f"{value_prefix}{idx}", None)
        anomaly_pvalue = row.get(f"{pval_prefix}{idx}", None)

        if is_nullish_text(anomaly_value) or contains_non_significant_marker(anomaly_value):
            continue

        if is_nullish_text(anomaly_pvalue) or contains_non_significant_marker(anomaly_pvalue):
            continue

        anomaly_name = row.get(f"{name_prefix}{idx}", None)
        if not is_nullish_text(anomaly_name):
            names.append(str(anomaly_name).strip())

    return names