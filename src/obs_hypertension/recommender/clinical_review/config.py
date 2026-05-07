"""
Configuration objects for clinical review outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ClinicalReviewConfig:
    """
    Configuration for patient profile display and clinical recommendation tables.

    Parameters
    ----------
    patient_id_col : str
        Patient identifier column.
    patient_display_cols : Tuple[str, ...]
        Columns to show in patient review table.
    treatment_display_cols : Tuple[str, ...]
        Columns to show in recommendation table for treatment details.
    score_cols : Tuple[str, ...]
        Columns to show in recommendation table for ranking / score info.
    score_round : int
        Number of decimals for score rounding.
    """
    patient_id_col: str = "patient_id"

    patient_display_cols: Tuple[str, ...] = (
        "patient_id",
        "maternal_age_years",
        "gestational_age_weeks",
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "Maternal_Diagnosis",
        "Clinical_notes",
        "Anomaly_1",
    )

    treatment_display_cols: Tuple[str, ...] = (
        "drug_1",
        "route_1",
        "dose_1",
        "drug_2",
        "route_2",
        "dose_2",
    )

    score_cols: Tuple[str, ...] = (
        "final_score",
        "rank",
        "clinical_alert",
    )

    score_round: int = 4