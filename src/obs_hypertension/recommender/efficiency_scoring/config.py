"""
Configuration objects and constants for protocol efficiency scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


NULLISH = {
    "",
    "not specified",
    "unspecified",
    "na",
    "n/a",
    "none",
    "unknown",
    "nan",
    "null",
    "missing",
}


@dataclass
class EfficiencyScorerConfig:
    """
    Configuration for protocol-level efficiency scoring.

    Parameters
    ----------
    feature_cols : List[str]
        Numeric / OHE feature columns used to train the RandomForest model.
    random_state : int
        Random seed.
    min_train_rows : int
        Minimum number of non-ambiguous rows required to train.
    n_estimators : int
        Number of trees in RandomForest.
    max_depth : Optional[int]
        Max tree depth.
    min_samples_leaf : int
        Minimum samples per leaf.
    n_jobs : int
        Parallel jobs for RandomForest.
    class_weight : str
        Class weighting strategy.
    default_score_if_insufficient : float
        Default efficiency score used when training signal is insufficient.
    """
    feature_cols: List[str]
    random_state: int = 42
    min_train_rows: int = 25
    n_estimators: int = 400
    max_depth: Optional[int] = 12
    min_samples_leaf: int = 4
    n_jobs: int = -1
    class_weight: str = "balanced"
    default_score_if_insufficient: float = 0.50