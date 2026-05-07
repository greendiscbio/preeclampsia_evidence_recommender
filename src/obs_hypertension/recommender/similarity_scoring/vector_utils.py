"""
Vector utilities for similarity scoring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def as_numpy_vector(value, dtype=np.float32) -> np.ndarray:
    """
    Convert list | tuple | np.ndarray into a 1D numpy array.

    Parameters
    ----------
    value : Any
        Vector-like input.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    np.ndarray
        1D vector.

    Raises
    ------
    ValueError
        If value is None or not 1D.
    TypeError
        If value type is invalid.
    """
    if value is None:
        raise ValueError("Vector is None")

    if isinstance(value, np.ndarray):
        vec = value.astype(dtype, copy=False)
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=dtype)
    else:
        raise TypeError(f"Invalid vector type: {type(value)}")

    if vec.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {vec.shape}")

    return vec


def as_float_matrix_from_list_column(
    series: pd.Series,
    expected_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Convert a Series of list[float] vectors into a 2D float32 matrix.

    Invalid or missing rows are replaced with zero-vectors.

    Parameters
    ----------
    series : pd.Series
        Series containing vector-like rows.
    expected_dim : Optional[int]
        Expected vector dimension. If None, inferred from first valid row.

    Returns
    -------
    np.ndarray
        Matrix of shape (N, D).
    """
    n_rows = len(series)

    if n_rows == 0:
        dim = expected_dim or 1
        return np.zeros((0, dim), dtype="float32")

    if expected_dim is None:
        for value in series.values:
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                expected_dim = len(value)
                break

    dim = expected_dim or 1
    matrix = np.zeros((n_rows, dim), dtype="float32")

    for idx, value in enumerate(series.values):
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)) and len(value) == dim:
            matrix[idx, :] = np.asarray(value, dtype="float32")

    return matrix


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize matrix row-wise.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (N, D).
    eps : float
        Numerical stability constant.

    Returns
    -------
    np.ndarray
        Normalized matrix.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + eps
    return matrix / norms


def cosine_sim_matrix(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of matrix_a and matrix_b.

    Parameters
    ----------
    matrix_a : np.ndarray
        Shape (N, D)
    matrix_b : np.ndarray
        Shape (M, D)

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (N, M)
    """
    a_norm = l2_normalize_rows(matrix_a)
    b_norm = l2_normalize_rows(matrix_b)
    return a_norm @ b_norm.T