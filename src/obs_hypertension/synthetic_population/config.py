"""
Configuration constants for synthetic population generation.
"""

AGE_OH = [
    "age_18_20",
    "age_21_23",
    "age_24_26",
    "age_27_29",
    "age_30_32",
    "age_33_35",
    "age_36_38",
    "age_39_41",
    "age_42_44",
]

GA_OH = [
    "<20_weeks",
    "20_33_weeks",
    ">33_weeks",
]

SBP_OH = [
    "<119_SBP",
    "120_129_SBP",
    "130_139_SBP",
    "140_160_SBP",
    ">161_SBP",
]

DBP_OH = [
    "<85_DBP",
    "86_90_DBP",
    "91_105_DBP",
    ">106_DBP",
]

STRUCTURED_COLS = AGE_OH + GA_OH + SBP_OH + DBP_OH

DIAGNOSIS_CATEGORIES = [
    "Severe Preeclampsia/Eclampsia",
    "Non-severe Preeclampsia",
    "Severe Gestational Hypertension",
    "Non-severe Gestational Hypertension",
    "Antenatal/Peripartum Hypertension",
    "Chronic Hypertension",
    "Normotensive",
]

COMORBIDITIES_CATEGORIES = [
    "None",
    "gestational diabetes",
    "chronic kidney disease",
    "asthma",
    "AV block",
    "lupus/antiphospholipid syndrome",
    "depression",
    "history of heart disease",
]

AGE_PROBS = [
    0.05,
    0.07,
    0.12,
    0.16,
    0.18,
    0.16,
    0.12,
    0.09,
    0.05,
]

GA_PROBS = [
    0.08,
    0.35,
    0.57,
]

DEFAULT_OUTPUT_COLUMNS = (
    ["patient_id"]
    + STRUCTURED_COLS
    + [
        "Maternal_Diagnosis",
        "Clinical_notes",
        "Anomaly_1",
        "Comorbility",
        "Proteinuria",
        "SBP_value",
        "DBP_value",
        "Age_value",
        "GA_value",
    ]
)