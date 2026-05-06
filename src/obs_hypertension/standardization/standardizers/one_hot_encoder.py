"""
ONE HOT ENCODER MODULE

Purpose
-------
Standardize selected categorical columns into reduced canonical categories
and apply one-hot encoding for downstream modeling.

Current scope
-------------
- Maternal diagnosis
- Drug categories
- Route of administration

Expected use
------------
This module is intended to be called after:
1. arm-level dataset construction
2. therapy standardization
3. cohort standardization
4. outcome standardization

Main public function
--------------------
one_hot_encode_standardized_columns(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Constants
# ============================================================

CAT_NULLISH = {
    "not specified",
    "not_specified",
    "unspecified",
    "na",
    "n/a",
    "none",
    "unknown",
    "",
    "nan",
    "null",
}

DEFAULT_DRUG_COLS: Tuple[str, ...] = ("drug_1", "drug_2", "winner_1", "winner_2")
DEFAULT_ROUTE_COLS: Tuple[str, ...] = ("route_1", "route_2")
DEFAULT_DB_DIAGNOSIS_COL: str = "maternal_clinical_diagnosis"


# ============================================================
# Maternal diagnosis standardization
# ============================================================

_DX_RULES: List[Tuple[str, re.Pattern]] = [
    ("Severe Preeclampsia/Eclampsia", re.compile(r"\beclampsia\b", re.I)),
    ("Severe Preeclampsia/Eclampsia", re.compile(r"\bsevere\s*pe\b", re.I)),
    ("Severe Preeclampsia/Eclampsia", re.compile(r"\bpreeclampsia\s*eclampsia\b", re.I)),
    ("Non-severe Preeclampsia", re.compile(r"\bpreeclampsia\b|\bpre[-\s]?eclampsia\b|\bpe\b", re.I)),
    (
        "Severe Gestational Hypertension",
        re.compile(
            r"\bgestational\s+hypertension\b.*\bsevere\b|\bsevere\b.*\bgestational\s+hypertension\b",
            re.I,
        ),
    ),
    ("Non-severe Gestational Hypertension", re.compile(r"\bgestational\s+hypertension\b", re.I)),
    ("Chronic Hypertension", re.compile(r"\bchronic\s+hypertension\b|\bchronic\s+htn\b", re.I)),
    (
        "Antenatal/Peripartum Hypertension",
        re.compile(r"\bhypertensive\s+crises?\b|\bhypertensive\s+emergenc(y|ies)\b", re.I),
    ),
    (
        "Antenatal/Peripartum Hypertension",
        re.compile(r"\bhdcp\b|\bhypertensive\s+disorder\s+complicating\s+pregnancy\b", re.I),
    ),
    ("Antenatal/Peripartum Hypertension", re.compile(r"\bpih\b|\bpregnancy\s+induced\s+hypertension\b", re.I)),
    (
        "Antenatal/Peripartum Hypertension",
        re.compile(r"\bhypertensive\s+disorders?\s+of\s+pregnancy\b|\bhypertensive\s+disorder\b", re.I),
    ),
    ("Normotensive", re.compile(r"\bnormotension\b|\bnormotensive\b", re.I)),
]


def _normalize_text(value: Any) -> str:
    """
    Normalize generic text safely.

    Parameters
    ----------
    value : Any
        Input value.

    Returns
    -------
    str
        Lowercased and whitespace-normalized text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def standardize_maternal_dx(value: Any) -> str:
    """
    Standardize maternal diagnosis into a reduced canonical taxonomy.

    Parameters
    ----------
    value : Any
        Raw diagnosis value.

    Returns
    -------
    str
        Standardized diagnosis category.
    """
    text = _normalize_text(value)

    if not text or text in CAT_NULLISH:
        return "not_specified"

    # Exclude values that are clearly not maternal diagnosis categories
    if re.search(r"\bheart disease\b|\btype\s*1\s*diabetes\b", text, flags=re.I):
        return "not_specified"
    if re.search(r"\bfetal growth restriction\b|\bumbilical\b", text, flags=re.I):
        return "not_specified"

    for label, pattern in _DX_RULES:
        if pattern.search(text):
            return label

    return "not_specified"


# ============================================================
# Drug category standardization
# ============================================================

def _clean_drug_text(value: Any) -> str:
    """
    Clean drug-related text for robust pattern matching.

    Parameters
    ----------
    value : Any
        Raw drug value.

    Returns
    -------
    str
        Cleaned text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    text = str(value).strip().lower()
    text = text.replace("β", "beta")
    text = re.sub(r"[^\w\s\-\+\/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def standardize_drug_category(value: Any) -> str:
    """
    Map raw drug names/treatments into reduced standardized categories.

    Parameters
    ----------
    value : Any
        Raw drug value.

    Returns
    -------
    str
        Standardized drug category.
    """
    text = _clean_drug_text(value)

    if not text or text in CAT_NULLISH:
        return "Not specified"

    if re.search(r"\bmgso4\b|\bmagnesium\s+sul(ph)?ate\b|\bmagnesium\b", text, flags=re.I):
        return "Magnesium sulfate"

    if re.search(r"\blabet(alol|olol)\b|\blabetolol\b", text, flags=re.I):
        return "Labetalol"

    if re.search(r"\bnifedipine\b|\bnifidepine\b", text, flags=re.I):
        return "Nifedipine"

    if re.search(r"\bmethyldopa\b|\balphamethydopa\b|\bdopegyt\b", text, flags=re.I):
        return "Methyldopa"

    if re.search(r"\bhydralazine\b", text, flags=re.I):
        return "Hydralazine"

    if re.search(
        r"\b(beta[-\s]?blockers?|bbs?)\b|\bmetoprolol\b|\batenolol\b|\bpropranolol\b|\bother\s+beta[-\s]?blockers\b",
        text,
        flags=re.I,
    ):
        return "Betablockers"

    if re.search(r"\bphenytoin\b|\bdiazepam\b", text, flags=re.I):
        return "Anticonvulsant"

    if re.search(r"\bropivacaine\b|\bbupivacaine\b|\bfentanyl\b|\btramadol\b|\bhyperbaric\b", text, flags=re.I):
        return "Anesthetic/pain relief"

    if re.search(r"\bnitroglycer(in|ine)\b|\bphentolamine\b|\bphenylephrine\b|\bnicardipine\b", text, flags=re.I):
        return "Vasodilators"

    if re.search(r"\bvitamin\s*d\b|\bvd\b|\bvitamin\s*c\b|\bvitamin\s*e\b", text, flags=re.I):
        return "Vitamins"

    if re.search(
        r"\btraditional\s+chinese\s+medicine\b|\bdecoction\b|\brecipe\b|\bdanshen\b|\bpinggan\b|\byiqi\b|\byangshen\b",
        text,
        flags=re.I,
    ):
        return "Traditional Chinese Medicine"

    if re.search(
        r"\bresveratrol\b|\ballopurinol\b|\bphytosterol\b|\bl[-\s]?citrulline\b|\bplacebo\b|\bglucose\b|\bstandard\s+treatment\b|\btotal\s+antihypertensive\b|\bsimilar\s+antihypertensive\b|\bcentral\s+a\s+agonists\b",
        text,
        flags=re.I,
    ):
        return "Other/Adjunct"

    return "Other/Adjunct"


# ============================================================
# Route standardization
# ============================================================

def _clean_route_text(value: Any) -> str:
    """
    Clean route-related text for robust pattern matching.

    Parameters
    ----------
    value : Any
        Raw route value.

    Returns
    -------
    str
        Cleaned text.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    text = str(value).strip().lower()
    text = text.replace("\\", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def standardize_route_category(value: Any) -> str:
    """
    Standardize route of administration into reduced categories.

    Categories
    ----------
    - IV
    - Oral
    - IV+Oral
    - Other
    - Not specified

    Parameters
    ----------
    value : Any
        Raw route value.

    Returns
    -------
    str
        Standardized route category.
    """
    text = _clean_route_text(value)

    if not text or text in CAT_NULLISH:
        return "Not specified"

    if re.search(r"\biv\s*\+\s*oral\b|\biv\+oral\b|\biv\s*and\s*oral\b", text, flags=re.I):
        return "IV+Oral"

    if re.fullmatch(r"oral", text) or re.search(r"\boral\b", text, flags=re.I):
        return "Oral"

    if re.search(r"\biv\b|\bi/v\b|\bintravenous\b", text, flags=re.I):
        return "IV"

    if re.search(r"\bim\b|\bi/m\b|\bintramuscular\b", text, flags=re.I):
        return "IV"

    if re.search(r"\bim\s*\+\s*iv\b|\biv\s*\+\s*im\b", text, flags=re.I):
        return "IV"

    if re.search(r"\binjectable\b|\binjected\b", text, flags=re.I):
        return "IV"

    if re.search(r"5%\s*glucose|dissolved|dropped", text, flags=re.I):
        return "IV"

    return "Other"


# ============================================================
# OHE helpers
# ============================================================

def _build_one_hot_encoder(series: pd.Series, feature_name: str) -> OneHotEncoder:
    """
    Fit a OneHotEncoder on a single pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input categorical series.
    feature_name : str
        Name used during fitting.

    Returns
    -------
    OneHotEncoder
        Fitted encoder.
    """
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int8,
    )
    encoder.fit(series.to_frame(name=feature_name))
    return encoder


def _transform_series_to_ohe_df(
    encoder: OneHotEncoder,
    series: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """
    Transform a single series using a fitted OneHotEncoder and return a DataFrame.

    Parameters
    ----------
    encoder : OneHotEncoder
        Fitted sklearn encoder.
    series : pd.Series
        Input categorical series.
    prefix : str
        Prefix for resulting one-hot columns.

    Returns
    -------
    pd.DataFrame
        One-hot encoded dataframe aligned to series index.
    """
    feature_name = encoder.feature_names_in_[0]
    transformed = encoder.transform(series.to_frame(name=feature_name))
    categories = list(encoder.categories_[0])
    columns = [f"{prefix}__{category}" for category in categories]

    return pd.DataFrame(transformed.astype(np.int8), columns=columns, index=series.index)


# ============================================================
# Main transformation function
# ============================================================

def one_hot_encode_standardized_columns(
    df: pd.DataFrame,
    drug_cols: Tuple[str, ...] = DEFAULT_DRUG_COLS,
    route_cols: Tuple[str, ...] = DEFAULT_ROUTE_COLS,
    diagnosis_col: str = DEFAULT_DB_DIAGNOSIS_COL,
    drop_original_cols: bool = False,
    return_artifacts: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardize selected categorical columns and apply one-hot encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input standardized dataframe.
    drug_cols : Tuple[str, ...], default=("drug_1", "drug_2", "winner_1", "winner_2")
        Drug-related columns to standardize and encode.
    route_cols : Tuple[str, ...], default=("route_1", "route_2")
        Route-related columns to standardize and encode.
    diagnosis_col : str, default="maternal_clinical_diagnosis"
        Maternal diagnosis column to standardize and encode.
    drop_original_cols : bool, default=False
        Whether to drop original raw columns after encoding.
    return_artifacts : bool, default=False
        Whether to return fitted encoders and metadata.

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict[str, Any]]
        Transformed dataframe, optionally with fitted encoders and metadata.
    """
    df_out = df.copy()

    # --------------------------------------------------------
    # Ensure expected columns exist
    # --------------------------------------------------------
    for column in list(drug_cols) + list(route_cols) + [diagnosis_col]:
        if column not in df_out.columns:
            df_out[column] = np.nan

    # --------------------------------------------------------
    # Standardize drug columns
    # --------------------------------------------------------
    drug_std_cols: List[str] = []
    for column in drug_cols:
        std_col = f"{column}__stdcat"
        df_out[std_col] = df_out[column].map(standardize_drug_category)
        drug_std_cols.append(std_col)

    # --------------------------------------------------------
    # Standardize route columns
    # --------------------------------------------------------
    route_std_cols: List[str] = []
    for column in route_cols:
        std_col = f"{column}__stdcat"
        df_out[std_col] = df_out[column].map(standardize_route_category)
        route_std_cols.append(std_col)

    # --------------------------------------------------------
    # Standardize diagnosis column
    # --------------------------------------------------------
    diagnosis_std_col = f"{diagnosis_col}__stdcat"
    df_out[diagnosis_std_col] = df_out[diagnosis_col].map(standardize_maternal_dx)

    # --------------------------------------------------------
    # Fit encoders
    # --------------------------------------------------------
    all_drug_values = pd.concat([df_out[col] for col in drug_std_cols], axis=0, ignore_index=True)
    drug_encoder = _build_one_hot_encoder(all_drug_values, "__drug_stdcat__")

    all_route_values = pd.concat([df_out[col] for col in route_std_cols], axis=0, ignore_index=True)
    route_encoder = _build_one_hot_encoder(all_route_values, "__route_stdcat__")

    diagnosis_encoder = _build_one_hot_encoder(df_out[diagnosis_std_col], diagnosis_std_col)

    # --------------------------------------------------------
    # Transform drug columns
    # --------------------------------------------------------
    encoded_parts: List[pd.DataFrame] = [df_out]

    for std_col in drug_std_cols:
        encoded_parts.append(
            _transform_series_to_ohe_df(
                encoder=drug_encoder,
                series=df_out[std_col],
                prefix=std_col,
            )
        )

    # --------------------------------------------------------
    # Transform route columns
    # --------------------------------------------------------
    for std_col in route_std_cols:
        encoded_parts.append(
            _transform_series_to_ohe_df(
                encoder=route_encoder,
                series=df_out[std_col],
                prefix=std_col,
            )
        )

    # --------------------------------------------------------
    # Transform diagnosis column
    # --------------------------------------------------------
    encoded_parts.append(
        _transform_series_to_ohe_df(
            encoder=diagnosis_encoder,
            series=df_out[diagnosis_std_col],
            prefix=diagnosis_std_col,
        )
    )

    df_encoded = pd.concat(encoded_parts, axis=1)

    # --------------------------------------------------------
    # Optional cleanup
    # --------------------------------------------------------
    if drop_original_cols:
        df_encoded = df_encoded.drop(
            columns=list(drug_cols) + list(route_cols) + [diagnosis_col],
            errors="ignore",
        )

    artifacts = {
        "encoders": {
            "drug_encoder": drug_encoder,
            "route_encoder": route_encoder,
            "diagnosis_encoder": diagnosis_encoder,
        },
        "metadata": {
            "drug_original_cols": list(drug_cols),
            "route_original_cols": list(route_cols),
            "diagnosis_original_col": diagnosis_col,
            "drug_std_cols": drug_std_cols,
            "route_std_cols": route_std_cols,
            "diagnosis_std_col": diagnosis_std_col,
            "drug_categories": list(drug_encoder.categories_[0]),
            "route_categories": list(route_encoder.categories_[0]),
            "diagnosis_categories": list(diagnosis_encoder.categories_[0]),
        },
    }

    if return_artifacts:
        return df_encoded, artifacts

    return df_encoded