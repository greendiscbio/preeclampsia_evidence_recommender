"""
Central default configuration for recommender_model.

Purpose
-------
Store reusable defaults for:
- structured feature columns
- efficiency feature columns
- default paths
- default score weights
- model defaults

This avoids repeating long CLI arguments on every execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATABASE_PATH = PROJECT_ROOT / "ProcessedOutputs" / "arm_dataset_standardized.csv"
DEFAULT_SYNTHETIC_PATH = PROJECT_ROOT / "synthetic_population_generator" / "outputs" / "synthetic_population.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "recommender_model" / "outputs" / "default_run"
DEFAULT_VECTORIZER_ARTIFACTS_DIR = DEFAULT_OUTPUT_DIR / "vectorizer_artifacts"


# ============================================================
# Structured columns
# ============================================================

STRUCTURED_COLS: List[str] = [
    # Age
    "age_18_20",
    "age_21_23",
    "age_24_26",
    "age_27_29",
    "age_30_32",
    "age_33_35",
    "age_36_38",
    "age_39_41",
    "age_42_44",

    # Gestational age
    "<20_weeks",
    "20_33_weeks",
    ">33_weeks",

    # Systolic blood pressure
    "<119_SBP",
    "120_129_SBP",
    "130_139_SBP",
    "140_160_SBP",
    ">161_SBP",

    # Diastolic blood pressure
    "<85_DBP",
    "86_90_DBP",
    "91_105_DBP",
    ">106_DBP",
]


# ============================================================
# Efficiency model feature columns
# ============================================================
# Base version:
# - starts from structured columns
# - adds a small set of useful numeric / treatment structure columns
# - you can expand this list later with OHE treatment columns
#   after checking which columns are consistently present in your final dataset.
# ============================================================

#EFFICIENCY_FEATURE_COLS: List[str] = STRUCTURED_COLS + [
#    "is_combination",
#]

# expanded example:
EFFICIENCY_FEATURE_COLS_MONO: List[str] = STRUCTURED_COLS + [

    # Diagnosis OHE examples
    "maternal_clinical_diagnosis__stdcat__Severe Preeclampsia/Eclampsia",
    "maternal_clinical_diagnosis__stdcat__Non-severe Preeclampsia",
    "maternal_clinical_diagnosis__stdcat__Severe Gestational Hypertension",
    "maternal_clinical_diagnosis__stdcat__Non-severe Gestational Hypertension",
    "maternal_clinical_diagnosis__stdcat__Antenatal/Peripartum Hypertension",
    "maternal_clinical_diagnosis__stdcat__Chronic Hypertension",
    "maternal_clinical_diagnosis__stdcat__Normotensive",
    "maternal_clinical_diagnosis__stdcat__not_specified",

    # Drug OHE examples for first treatment
    "drug_1__stdcat__Labetalol",
    "drug_1__stdcat__Nifedipine",
    "drug_1__stdcat__Methyldopa",
    "drug_1__stdcat__Hydralazine",
    "drug_1__stdcat__Betablockers",
    "drug_1__stdcat__Magnesium sulfate",
    "drug_1__stdcat__Vasodilators",
    "drug_1__stdcat__Other/Adjunct",
    "drug_1__stdcat__Not specified",

    # Route OHE examples
    "route_1__stdcat__IV",
    "route_1__stdcat__Oral",
    "route_1__stdcat__IV+Oral",
    "route_1__stdcat__Other",
    "route_1__stdcat__Not specified",
]

EFFICIENCY_FEATURE_COLS_COMBO: List[str] = STRUCTURED_COLS + [

    # Diagnosis OHE examples
    "maternal_clinical_diagnosis__stdcat__Severe Preeclampsia/Eclampsia",
    "maternal_clinical_diagnosis__stdcat__Non-severe Preeclampsia",
    "maternal_clinical_diagnosis__stdcat__Severe Gestational Hypertension",
    "maternal_clinical_diagnosis__stdcat__Non-severe Gestational Hypertension",
    "maternal_clinical_diagnosis__stdcat__Antenatal/Peripartum Hypertension",
    "maternal_clinical_diagnosis__stdcat__Chronic Hypertension",
    "maternal_clinical_diagnosis__stdcat__Normotensive",
    "maternal_clinical_diagnosis__stdcat__not_specified",

    # Drug OHE examples for first treatment
    "drug_1__stdcat__Labetalol",
    "drug_1__stdcat__Nifedipine",
    "drug_1__stdcat__Methyldopa",
    "drug_1__stdcat__Hydralazine",
    "drug_1__stdcat__Betablockers",
    "drug_1__stdcat__Magnesium sulfate",
    "drug_1__stdcat__Vasodilators",
    "drug_1__stdcat__Other/Adjunct",
    "drug_1__stdcat__Not specified",

    # Route OHE examples
    "route_1__stdcat__IV",
    "route_1__stdcat__Oral",
    "route_1__stdcat__IV+Oral",
    "route_1__stdcat__Other",
    "route_1__stdcat__Not specified",

    # Drug OHE examples for first treatment
    "drug_2__stdcat__Labetalol",
    "drug_2__stdcat__Nifedipine",
    "drug_2__stdcat__Methyldopa",
    "drug_2__stdcat__Hydralazine",
    "drug_2__stdcat__Betablockers",
    "drug_2__stdcat__Magnesium sulfate",
    "drug_2__stdcat__Vasodilators",
    "drug_2__stdcat__Other/Adjunct",
    "drug_2__stdcat__Not specified",

    # Route OHE examples
    "route_2__stdcat__IV",
    "route_2__stdcat__Oral",
    "route_2__stdcat__IV+Oral",
    "route_2__stdcat__Other",
    "route_2__stdcat__Not specified",
]



# ============================================================
# Default score weights
# ============================================================

DEFAULT_FINAL_SCORE_WEIGHTS: Dict[str, float] = {
    "similarity_weight": 0.50,
    "efficiency_weight": 0.30,
    "safety_weight": 0.20,
}

DEFAULT_SIMILARITY_WEIGHTS: Dict[str, float] = {
    "w_struct": 0.60,
    "w_notes": 0.40,
}


# ============================================================
# Default model / processing parameters
# ============================================================

DEFAULT_ST_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
DEFAULT_ST_BATCH_SIZE = 64
DEFAULT_NORMALIZE_EMBEDDINGS = True

DEFAULT_TOP_N = 3
DEFAULT_PRECLINICAL_TOP_K = 20

DEFAULT_RISK_MAX_OUTCOMES_PER_TYPE = 2
DEFAULT_SIMILARITY_MAX_ANOMALY_SLOTS = 5

DEFAULT_EFFICIENCY_PARAMS: Dict[str, object] = {
    "random_state": 42,
    "min_train_rows": 25,
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 4,
    "n_jobs": -1,
    "class_weight": "balanced",
    "default_score_if_insufficient": 0.50,
}


# ============================================================
# Final clinical filtering defaults
# ============================================================

DEFAULT_PATIENT_ID_COL = "patient_id"
DEFAULT_PROTOCOL_ID_COL = "corpusid"
DEFAULT_DIAGNOSIS_COL = "Maternal_Diagnosis"
DEFAULT_COMORBIDITY_COL = "Comorbility"
DEFAULT_FINAL_SCORE_COL = "final_score"
DEFAULT_DEDUPLICATE_BY = "protocol_id"


# ============================================================
# Review display defaults
# ============================================================

DEFAULT_SCORE_ROUND = 4