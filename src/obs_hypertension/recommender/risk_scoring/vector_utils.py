"""
Vector utilities for risk scoring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def l2_normalize_vec(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize a 1D vector.
    """
    norm = float(np.linalg.norm(vector)) + eps
    return (vector / norm).astype("float32")


def as_numpy_vector(value, dtype: str = "float32") -> Optional[np.ndarray]:
    """
    Convert list or numpy array into a 1D numpy vector.
    Returns None if invalid.
    """
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    array = np.asarray(value, dtype=dtype)

    if array.ndim != 1:
        array = array.reshape(-1)

    if array.size == 0:
        return None

    return array


def cosine_sim_unit(vec_a_unit: np.ndarray, vec_b_unit: np.ndarray) -> float:
    """
    Dot product between two already L2-normalized vectors.
    """
    return float(vec_a_unit @ vec_b_unit)