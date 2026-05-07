"""
Configuration defaults for recommender vectorization module.
"""

DEFAULT_ST_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
DEFAULT_OUTPUT_DIR = "recommender_model/artifacts/vectorizer"
DEFAULT_SAVE_PREFIX = "hypertension_rec_v1"

NULLISH = {
    "not specified",
    "unspecified",
    "na",
    "n/a",
    "none",
    "unknown",
    "",
    "nan",
    "null",
}

DEFAULT_DB_CLINICAL_TEXT_COLS = ("maternal_clinical_diagnosis__stdcat",)
DEFAULT_SYN_CLINICAL_TEXT_COL = "Clinical_notes"
DEFAULT_SYN_ANOMALY_1_COL = "Anomaly_1"