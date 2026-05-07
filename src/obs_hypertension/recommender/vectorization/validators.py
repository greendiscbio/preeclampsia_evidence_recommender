"""
Validation utilities for recommender vectorization module.

Purpose
-------
Validate schema consistency before and after vectorization.

Checks included
---------------
1. Structured columns existence in DB and cohort dataset
2. Outcome column numbering consistency
3. Cohort schema compatibility with DB structured schema
4. Vector column dimensional consistency after vectorization
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ============================================================
# Generic helpers
# ============================================================

def _missing_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """
    Return missing columns from a dataframe.
    """
    return [col for col in columns if col not in df.columns]


def _extract_numeric_suffix(column_name: str) -> Optional[int]:
    """
    Extract trailing numeric suffix from a column name.

    Example
    -------
    maternal_outcome_name_3 -> 3
    """
    match = re.search(r"(\d+)$", column_name)
    return int(match.group(1)) if match else None


def _find_prefixed_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Find columns matching prefix + integer suffix.

    Example
    -------
    prefix='maternal_outcome_name_' finds:
    maternal_outcome_name_1, maternal_outcome_name_2, ...
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
    matched = [col for col in df.columns if pattern.match(col)]
    return sorted(
        matched,
        key=lambda col: _extract_numeric_suffix(col) if _extract_numeric_suffix(col) is not None else 999999
    )


def _check_consecutive_suffixes(columns: Sequence[str], prefix: str) -> Tuple[bool, List[int], List[int]]:
    """
    Check whether numeric suffixes are consecutive starting from 1.

    Returns
    -------
    Tuple[bool, List[int], List[int]]
        (
            is_valid,
            observed_suffixes,
            expected_suffixes
        )
    """
    if not columns:
        return True, [], []

    observed = [_extract_numeric_suffix(col) for col in columns]
    observed = [x for x in observed if x is not None]
    expected = list(range(1, len(observed) + 1))
    is_valid = observed == expected
    return is_valid, observed, expected


def _get_vector_length(value) -> Optional[int]:
    """
    Safely infer the dimensionality of a vector-like object.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return len(value)

    return None


# ============================================================
# Pre-vectorization validators
# ============================================================

def validate_structured_columns_exist(
    df_database: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    structured_cols: Sequence[str],
    raise_error: bool = True,
) -> Dict[str, List[str]]:
    """
    Validate that all structured columns exist in both DB and synthetic datasets.

    Parameters
    ----------
    df_database : pd.DataFrame
        Database dataframe.
    synthetic_population : pd.DataFrame
        Synthetic population dataframe.
    structured_cols : Sequence[str]
        Required structured columns.
    raise_error : bool, default=True
        Whether to raise ValueError when validation fails.

    Returns
    -------
    Dict[str, List[str]]
        Missing columns grouped by dataframe.
    """
    db_missing = _missing_columns(df_database, structured_cols)
    syn_missing = _missing_columns(synthetic_population, structured_cols)

    result = {
        "db_missing_structured_cols": db_missing,
        "synthetic_missing_structured_cols": syn_missing,
    }

    if raise_error and (db_missing or syn_missing):
        message_parts = []
        if db_missing:
            message_parts.append(
                f"Database is missing structured columns: {db_missing}"
            )
        if syn_missing:
            message_parts.append(
                f"Synthetic population is missing structured columns: {syn_missing}"
            )
        raise ValueError(" | ".join(message_parts))

    return result


def validate_outcome_column_numbering(
    df_database: pd.DataFrame,
    prefixes: Sequence[str] = (
        "maternal_outcome_name_",
        "fetal_outcome_name_",
        "anomaly_outcome_name_",
    ),
    raise_error: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Validate that outcome columns are consecutively numbered.

    Example valid patterns
    ----------------------
    maternal_outcome_name_1, maternal_outcome_name_2, maternal_outcome_name_3

    Example invalid patterns
    ------------------------
    maternal_outcome_name_1, maternal_outcome_name_3

    Parameters
    ----------
    df_database : pd.DataFrame
        Database dataframe.
    prefixes : Sequence[str]
        Prefixes to validate.
    raise_error : bool, default=True
        Whether to raise ValueError if invalid numbering is found.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Validation details per prefix.
    """
    results: Dict[str, Dict[str, object]] = {}
    errors: List[str] = []

    for prefix in prefixes:
        cols = _find_prefixed_columns(df_database, prefix)
        is_valid, observed, expected = _check_consecutive_suffixes(cols, prefix)

        results[prefix] = {
            "columns": cols,
            "is_valid": is_valid,
            "observed_suffixes": observed,
            "expected_suffixes": expected,
        }

        if not is_valid:
            errors.append(
                f"Invalid numbering for prefix '{prefix}'. "
                f"Observed suffixes: {observed}. Expected: {expected}."
            )

    if raise_error and errors:
        raise ValueError(" | ".join(errors))

    return results


def validate_synthetic_schema_compatibility(
    df_database: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    structured_cols: Sequence[str],
    required_syn_text_cols: Sequence[str] = ("Clinical_notes", "Anomaly_1", "Maternal_Diagnosis"),
    raise_error: bool = True,
) -> Dict[str, object]:
    """
    Validate that the synthetic population is compatible with the expected recommender schema.

    Checks
    ------
    - structured columns exist in both datasets
    - synthetic population includes required text/clinical columns
    - database contains at least the structured schema used for vectorization

    Parameters
    ----------
    df_database : pd.DataFrame
        Database dataframe.
    synthetic_population : pd.DataFrame
        Synthetic population dataframe.
    structured_cols : Sequence[str]
        Structured columns required for both datasets.
    required_syn_text_cols : Sequence[str]
        Required synthetic-specific non-structured columns.
    raise_error : bool, default=True
        Whether to raise ValueError if validation fails.

    Returns
    -------
    Dict[str, object]
        Validation summary.
    """
    structured_check = validate_structured_columns_exist(
        df_database=df_database,
        synthetic_population=synthetic_population,
        structured_cols=structured_cols,
        raise_error=False,
    )

    syn_missing_required = _missing_columns(synthetic_population, required_syn_text_cols)

    result = {
        "structured_check": structured_check,
        "synthetic_missing_required_cols": syn_missing_required,
        "is_compatible": (
            len(structured_check["db_missing_structured_cols"]) == 0
            and len(structured_check["synthetic_missing_structured_cols"]) == 0
            and len(syn_missing_required) == 0
        ),
    }

    if raise_error and not result["is_compatible"]:
        message_parts = []

        if structured_check["db_missing_structured_cols"]:
            message_parts.append(
                f"Database missing structured columns: {structured_check['db_missing_structured_cols']}"
            )
        if structured_check["synthetic_missing_structured_cols"]:
            message_parts.append(
                f"Synthetic population missing structured columns: {structured_check['synthetic_missing_structured_cols']}"
            )
        if syn_missing_required:
            message_parts.append(
                f"Synthetic population missing required columns: {syn_missing_required}"
            )

        raise ValueError(" | ".join(message_parts))

    return result


def run_pre_vectorization_validations(
    df_database: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    structured_cols: Sequence[str],
    outcome_prefixes: Sequence[str] = (
        "maternal_outcome_name_",
        "fetal_outcome_name_",
        "anomaly_outcome_name_",
    ),
    required_syn_text_cols: Sequence[str] = ("Clinical_notes", "Anomaly_1", "Maternal_Diagnosis"),
    raise_error: bool = True,
) -> Dict[str, object]:
    """
    Run all pre-vectorization validations.

    Returns
    -------
    Dict[str, object]
        Complete validation report.
    """
    report = {
        "structured_columns": validate_structured_columns_exist(
            df_database=df_database,
            synthetic_population=synthetic_population,
            structured_cols=structured_cols,
            raise_error=False,
        ),
        "outcome_numbering": validate_outcome_column_numbering(
            df_database=df_database,
            prefixes=outcome_prefixes,
            raise_error=False,
        ),
        "synthetic_schema": validate_synthetic_schema_compatibility(
            df_database=df_database,
            synthetic_population=synthetic_population,
            structured_cols=structured_cols,
            required_syn_text_cols=required_syn_text_cols,
            raise_error=False,
        ),
    }

    errors: List[str] = []

    if (
        report["structured_columns"]["db_missing_structured_cols"]
        or report["structured_columns"]["synthetic_missing_structured_cols"]
    ):
        errors.append("Structured columns validation failed.")

    invalid_outcome_prefixes = [
        prefix
        for prefix, info in report["outcome_numbering"].items()
        if not info["is_valid"]
    ]
    if invalid_outcome_prefixes:
        errors.append(
            f"Outcome numbering validation failed for prefixes: {invalid_outcome_prefixes}."
        )

    if not report["synthetic_schema"]["is_compatible"]:
        errors.append("Synthetic schema compatibility validation failed.")

    report["is_valid"] = len(errors) == 0
    report["errors"] = errors

    if raise_error and errors:
        raise ValueError(" | ".join(errors))

    return report


# ============================================================
# Post-vectorization validators
# ============================================================

def validate_vector_dimensions(
    df: pd.DataFrame,
    vector_columns: Sequence[str],
    dataframe_name: str = "dataframe",
    raise_error: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Validate that each vector column has consistent dimensionality across rows.

    Parameters
    ----------
    df : pd.DataFrame
        Vectorized dataframe.
    vector_columns : Sequence[str]
        Columns expected to contain vectors.
    dataframe_name : str
        Label used in error messages.
    raise_error : bool, default=True
        Whether to raise ValueError on failure.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Validation summary per vector column.
    """
    results: Dict[str, Dict[str, object]] = {}
    errors: List[str] = []

    for col in vector_columns:
        if col not in df.columns:
            results[col] = {
                "exists": False,
                "unique_dimensions": [],
                "is_valid": False,
            }
            errors.append(f"Column '{col}' is missing in {dataframe_name}.")
            continue

        lengths = df[col].map(_get_vector_length)
        non_null_lengths = sorted(set(length for length in lengths if length is not None))

        is_valid = len(non_null_lengths) == 1

        results[col] = {
            "exists": True,
            "unique_dimensions": non_null_lengths,
            "is_valid": is_valid,
        }

        if not is_valid:
            errors.append(
                f"Column '{col}' in {dataframe_name} has inconsistent vector dimensions: {non_null_lengths}"
            )

    if raise_error and errors:
        raise ValueError(" | ".join(errors))

    return results


def validate_db_and_synthetic_vector_compatibility(
    df_database_vectorized: pd.DataFrame,
    df_synthetic_vectorized: pd.DataFrame,
    db_vector_columns: Sequence[str] = (
        "structured_profile_vector",
        "clinical_notes_vector",
    ),
    synthetic_vector_columns: Sequence[str] = (
        "structured_profile_vector",
        "clinical_notes_vector",
        "anomaly_1_vector",
        "general_notes_vector",
    ),
    raise_error: bool = True,
) -> Dict[str, object]:
    """
    Validate post-vectorization consistency for DB and synthetic data.

    Checks
    ------
    - each vector column exists
    - each vector column has internally consistent dimensions
    - shared vector columns between DB and synthetic have same dimension

    Parameters
    ----------
    df_database_vectorized : pd.DataFrame
        Vectorized database dataframe.
    df_synthetic_vectorized : pd.DataFrame
        Vectorized synthetic dataframe.
    db_vector_columns : Sequence[str]
        Vector columns expected in DB.
    synthetic_vector_columns : Sequence[str]
        Vector columns expected in synthetic population.
    raise_error : bool, default=True
        Whether to raise ValueError if validation fails.

    Returns
    -------
    Dict[str, object]
        Validation summary.
    """
    db_report = validate_vector_dimensions(
        df=df_database_vectorized,
        vector_columns=db_vector_columns,
        dataframe_name="df_database_vectorized",
        raise_error=False,
    )

    syn_report = validate_vector_dimensions(
        df=df_synthetic_vectorized,
        vector_columns=synthetic_vector_columns,
        dataframe_name="df_synthetic_vectorized",
        raise_error=False,
    )

    shared_dimension_checks = {}
    errors: List[str] = []

    shared_cols = set(db_vector_columns).intersection(set(synthetic_vector_columns))
    for col in shared_cols:
        db_dims = db_report.get(col, {}).get("unique_dimensions", [])
        syn_dims = syn_report.get(col, {}).get("unique_dimensions", [])

        same_dimension = (
            len(db_dims) == 1
            and len(syn_dims) == 1
            and db_dims[0] == syn_dims[0]
        )

        shared_dimension_checks[col] = {
            "db_dimensions": db_dims,
            "synthetic_dimensions": syn_dims,
            "same_dimension": same_dimension,
        }

        if not same_dimension:
            errors.append(
                f"Shared vector column '{col}' has incompatible dimensions. "
                f"DB: {db_dims} | Synthetic: {syn_dims}"
            )

    db_invalid = [col for col, info in db_report.items() if not info["is_valid"]]
    syn_invalid = [col for col, info in syn_report.items() if not info["is_valid"]]

    if db_invalid:
        errors.append(f"Invalid DB vector columns: {db_invalid}")
    if syn_invalid:
        errors.append(f"Invalid synthetic vector columns: {syn_invalid}")

    result = {
        "db_vector_report": db_report,
        "synthetic_vector_report": syn_report,
        "shared_dimension_checks": shared_dimension_checks,
        "is_valid": len(errors) == 0,
        "errors": errors,
    }

    if raise_error and errors:
        raise ValueError(" | ".join(errors))

    return result


def run_post_vectorization_validations(
    df_database_vectorized: pd.DataFrame,
    df_synthetic_vectorized: pd.DataFrame,
    raise_error: bool = True,
) -> Dict[str, object]:
    """
    Run all post-vectorization validations.

    Returns
    -------
    Dict[str, object]
        Complete post-vectorization validation report.
    """
    return validate_db_and_synthetic_vector_compatibility(
        df_database_vectorized=df_database_vectorized,
        df_synthetic_vectorized=df_synthetic_vectorized,
        raise_error=raise_error,
    )