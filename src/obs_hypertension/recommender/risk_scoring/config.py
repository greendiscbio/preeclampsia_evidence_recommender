"""
Configuration constants for protocol risk scoring.
"""

RISK_TIERS = {
    "catastrophic": {
        "weight": 3.0,
        "phrases": [
            "maternal death",
            "fetal death",
            "stillbirth",
            "organ failure",
            "stroke",
            "intracranial hemorrhage",
            "HELLP syndrome",
            "eclampsia",
            "severe preeclampsia",
            "neonatal mortality",
            "intrauterine death",
        ],
    },
    "severe": {
        "weight": 2.0,
        "phrases": [
            "ICU admission",
            "seizure",
            "placental abruption",
            "preterm birth",
            "fetal growth restriction",
            "neonatal ICU admission",
            "respiratory distress syndrome",
            "serious maternal complications",
        ],
    },
    "moderate": {
        "weight": 1.0,
        "phrases": [
            "hypotension",
            "tachycardia",
            "headache",
            "low birth weight",
            "small for gestational age",
            "adverse effects",
            "side effects",
        ],
    },
}

OUTCOME_TYPE_WEIGHTS = {
    "maternal": 1.0,
    "fetal": 0.9,
    "anomaly": 0.7,
}

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
    "missing",
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

RISK_TYPE_ALIASES = {
    "risk",
    "adverse",
    "safety",
    "harm",
}

DEFAULT_MAX_OUTCOMES_PER_TYPE = 2
DEFAULT_ST_BATCH_SIZE = 64