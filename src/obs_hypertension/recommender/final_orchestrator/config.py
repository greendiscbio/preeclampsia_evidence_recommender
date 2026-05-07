"""
Configuration objects and constants for final clinical recommendation filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


DEFAULT_HYPERTENSIVE_DBP_COLS = [
    "86_90_DBP",
    "91_105_DBP",
    ">106_DBP",
]

DEFAULT_CONTRAINDICATED_DRUGS = {
    "labetalol": {"asthma", "history of heart disease", "av block"},
    "nifedipine": {"history of heart disease"},
    "methyldopa": {"chronic kidney disease", "depression"},
}


@dataclass
class FinalClinicalFilterConfig:
    """
    Configuration for final clinical filtering and recommendation cleanup.

    Parameters
    ----------
    top_n : int
        Maximum number of final recommendations per patient.
    patient_id_col : str
        Patient identifier column.
    diagnosis_col : str
        Diagnosis column in synthetic population.
    comorbidity_col : str
        Comorbidity column in synthetic population.
    final_score_col : str
        Final score column used for ranking.
    protocol_id_col : str
        Protocol identifier column in recommendations dataframe.
    deduplicate_by : Optional[str]
        Deduplication mode:
        - "treatment"
        - "protocol_id"
        - None
    hypertensive_dbp_cols : List[str]
        DBP columns used to infer hypertensive status.
    contraindicated_drugs : Dict[str, Set[str]]
        Drug-comorbidity contraindication map.
    """
    top_n: int = 3
    patient_id_col: str = "patient_id"
    diagnosis_col: str = "Maternal_Diagnosis"
    comorbidity_col: str = "Comorbility"
    final_score_col: str = "final_score"
    protocol_id_col: str = "corpusid"
    deduplicate_by: Optional[str] = "protocol_id"
    hypertensive_dbp_cols: List[str] = field(default_factory=lambda: DEFAULT_HYPERTENSIVE_DBP_COLS.copy())
    contraindicated_drugs: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            drug: set(values)
            for drug, values in DEFAULT_CONTRAINDICATED_DRUGS.items()
        }
    )