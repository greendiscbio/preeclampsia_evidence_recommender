"""
Configuration constants for similarity scoring module.
"""

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

NON_SIGNIFICANT_MARKERS = {
    "not significant",
    "no significant",
    "no statistically significant",
    "statistically insignificant",
    "insignificant",
    "no statistical difference",
    "no statistically significant difference",
    "not statistically significant difference",
    "no difference",
    "no differences",
    "similar",
}

DEFAULT_PATIENT_ID_COL = "patient_id"
DEFAULT_DB_ID_COL = "corpusid"

DEFAULT_STRUCTURED_VEC_COL = "structured_profile_vector"
DEFAULT_PATIENT_NOTES_VEC_COL = "general_notes_vector"
DEFAULT_PROTOCOL_CLINICAL_VEC_COL = "clinical_notes_vector"

DEFAULT_MAX_ANOMALY_SLOTS = 3

DEFAULT_W_STRUCT = 0.7
DEFAULT_W_NOTES = 0.3