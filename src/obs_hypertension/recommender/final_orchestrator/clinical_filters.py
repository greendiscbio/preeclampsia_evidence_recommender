"""
Clinical filtering helpers for final recommendation orchestration.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.obs_hypertension.recommender.final_orchestrator.config import FinalClinicalFilterConfig
from src.obs_hypertension.recommender.final_orchestrator.utils import (
    build_treatment_signature,
    normalize_text,
)


def build_patient_hypertension_status_map(
    synthetic_population: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> Dict[str, bool]:
    """
    Build patient -> hypertensive status map from synthetic population.
    """
    available_dbp_cols = [
        col for col in config.hypertensive_dbp_cols
        if col in synthetic_population.columns
    ]

    if not available_dbp_cols:
        raise ValueError(
            "No hypertensive DBP columns were found in synthetic_population."
        )

    if config.diagnosis_col in synthetic_population.columns:
        is_hypertensive = (
            (synthetic_population[available_dbp_cols].sum(axis=1) > 0)
            & (synthetic_population[config.diagnosis_col] != "Normotensive")
        )
    else:
        is_hypertensive = synthetic_population[available_dbp_cols].sum(axis=1) > 0

    return dict(
        zip(
            synthetic_population[config.patient_id_col],
            is_hypertensive,
        )
    )


def build_patient_comorbidity_map(
    synthetic_population: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> Dict[str, str]:
    """
    Build patient -> comorbidity map from synthetic population.
    """
    if config.comorbidity_col in synthetic_population.columns:
        return dict(
            zip(
                synthetic_population[config.patient_id_col],
                synthetic_population[config.comorbidity_col].fillna("None").astype(str),
            )
        )

    return {
        pid: "None"
        for pid in synthetic_population[config.patient_id_col].unique()
    }


def is_recommendation_contraindicated(
    row: pd.Series,
    patient_comorbidity: str,
    config: FinalClinicalFilterConfig,
) -> bool:
    """
    Check whether a recommendation is contraindicated for a patient comorbidity.
    """
    patient_comorbidity_norm = normalize_text(patient_comorbidity)

    for drug_col in ["drug_1", "drug_2"]:
        drug_name = normalize_text(row.get(drug_col, ""))
        if drug_name in config.contraindicated_drugs:
            if patient_comorbidity_norm in config.contraindicated_drugs[drug_name]:
                return True

    return False


def build_normotensive_output(
    group: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> pd.DataFrame:
    """
    Build final recommendation row for normotensive patients.
    """
    first_row = group.iloc[[0]].copy()

    cols_to_clear = [
        config.final_score_col,
        "similarity_score",
        "drug_1",
        "route_1",
        "dose_1",
        "drug_2",
        "route_2",
        "dose_2",
    ]

    for col in cols_to_clear:
        if col in first_row.columns:
            first_row[col] = np.nan

    first_row["clinical_alert"] = (
        "Esta paciente muestra rangos de presión normales (<85_DBP). "
        "No requiere tratamiento antihipertensivo."
    )

    return first_row


def build_all_contraindicated_output(
    group: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build fallback output when all recommendations are contraindicated.
    """
    first_row = group.iloc[[0]].copy()
    first_row["clinical_alert"] = (
        "Alerta: Todas las opciones preferentes están contraindicadas por comorbilidad."
    )
    return first_row


def deduplicate_recommendations(
    recommendations: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> pd.DataFrame:
    """
    Deduplicate recommendations according to configured strategy.
    """
    df = recommendations.copy()

    if config.deduplicate_by == "treatment":
        df["treatment_signature"] = df.apply(build_treatment_signature, axis=1)
        df = (
            df.sort_values(config.final_score_col, ascending=False)
            .drop_duplicates(subset=["treatment_signature"], keep="first")
        )

    elif config.deduplicate_by == "protocol_id":
        if config.protocol_id_col in df.columns:
            df = (
                df.sort_values(config.final_score_col, ascending=False)
                .drop_duplicates(subset=[config.protocol_id_col], keep="first")
            )

    return df


def apply_patient_clinical_logic(
    group: pd.DataFrame,
    patient_status_map: Dict[str, bool],
    patient_comorbidity_map: Dict[str, str],
    config: FinalClinicalFilterConfig,
) -> pd.DataFrame:
    """
    Apply patient-specific final clinical filtering logic.
    """
    patient_id = group.name
    is_hypertensive = patient_status_map.get(patient_id, False)

    if config.final_score_col in group.columns:
        group = group.sort_values(config.final_score_col, ascending=False).copy()
    else:
        group = group.copy()

    if not is_hypertensive:
        return build_normotensive_output(
            group=group,
            config=config,
        )

    patient_comorbidity = patient_comorbidity_map.get(patient_id, "None")

    valid_recommendations = group[
        ~group.apply(
            lambda row: is_recommendation_contraindicated(
                row=row,
                patient_comorbidity=patient_comorbidity,
                config=config,
            ),
            axis=1,
        )
    ].copy()

    if valid_recommendations.empty:
        return build_all_contraindicated_output(group)

    valid_recommendations = deduplicate_recommendations(
        recommendations=valid_recommendations,
        config=config,
    )

    valid_recommendations = (
        valid_recommendations
        .sort_values(config.final_score_col, ascending=False)
        .head(config.top_n)
        .copy()
    )

    valid_recommendations["clinical_alert"] = "Tratamiento validado clínicamente."

    return valid_recommendations