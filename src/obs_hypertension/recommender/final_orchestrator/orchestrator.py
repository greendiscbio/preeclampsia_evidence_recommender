"""
Final orchestrator for clinical filtering of ranked recommendations.
"""

from __future__ import annotations

import pandas as pd

from src.obs_hypertension.recommender.final_orchestrator.clinical_filters import (
    apply_patient_clinical_logic,
    build_patient_comorbidity_map,
    build_patient_hypertension_status_map,
)
from src.obs_hypertension.recommender.final_orchestrator.config import FinalClinicalFilterConfig


def _validate_final_orchestrator_inputs(
    recs_raw_df: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> None:
    """
    Validate required inputs for final clinical orchestration.
    """
    if recs_raw_df.empty:
        raise ValueError("recs_raw_df is empty.")

    if synthetic_population.empty:
        raise ValueError("synthetic_population is empty.")

    if config.patient_id_col not in recs_raw_df.columns:
        raise ValueError(
            f"recs_raw_df is missing required patient id column: '{config.patient_id_col}'"
        )

    if config.patient_id_col not in synthetic_population.columns:
        raise ValueError(
            f"synthetic_population is missing required patient id column: '{config.patient_id_col}'"
        )

    if config.top_n <= 0:
        raise ValueError("config.top_n must be greater than 0.")

    if config.deduplicate_by not in {"treatment", "protocol_id", None}:
        raise ValueError(
            "config.deduplicate_by must be one of: 'treatment', 'protocol_id', None"
        )


def apply_final_clinical_filters(
    recs_raw_df: pd.DataFrame,
    synthetic_population: pd.DataFrame,
    config: FinalClinicalFilterConfig,
) -> pd.DataFrame:
    """
    Apply final clinical filtering to ranked recommendations.

    Workflow
    --------
    1. Infer hypertensive status per patient
    2. Read patient comorbidities
    3. Apply patient-level clinical logic:
       - normotensive patients -> no antihypertensive recommendation
       - hypertensive patients -> remove contraindicated drugs
       - deduplicate repeated treatments/protocols
       - keep top_n final recommendations

    Parameters
    ----------
    recs_raw_df : pd.DataFrame
        Ranked recommendations before clinical filtering.
    synthetic_population : pd.DataFrame
        Synthetic patient population with diagnosis and comorbidity data.
    config : FinalClinicalFilterConfig
        Filtering configuration.

    Returns
    -------
    pd.DataFrame
        Final clinically-filtered recommendations.
    """
    _validate_final_orchestrator_inputs(
        recs_raw_df=recs_raw_df,
        synthetic_population=synthetic_population,
        config=config,
    )

    patient_status_map = build_patient_hypertension_status_map(
        synthetic_population=synthetic_population,
        config=config,
    )

    patient_comorbidity_map = build_patient_comorbidity_map(
        synthetic_population=synthetic_population,
        config=config,
    )

    recs_final_df = (
        recs_raw_df
        .groupby(config.patient_id_col, group_keys=False)
        .apply(
            lambda group: apply_patient_clinical_logic(
                group=group,
                patient_status_map=patient_status_map,
                patient_comorbidity_map=patient_comorbidity_map,
                config=config,
            )
        )
        .reset_index(drop=True)
    )

    return recs_final_df