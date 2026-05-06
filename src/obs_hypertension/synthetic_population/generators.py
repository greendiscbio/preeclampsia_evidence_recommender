"""
Core logic for synthetic patient generation.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pandas as pd

from src.obs_hypertension.synthetic_population.config import (
    AGE_OH,
    AGE_PROBS,
    COMORBIDITIES_CATEGORIES,
    DBP_OH,
    GA_OH,
    GA_PROBS,
    SBP_OH,
)


def seed_everything(seed: int = 42) -> None:
    """
    Seed all internal random generators used in this module.
    """
    random.seed(seed)


def choose_one(options: List[str], weights: List[float]) -> str:
    """
    Randomly choose one element from a weighted categorical list.
    """
    return random.choices(options, weights=weights, k=1)[0]


def one_hot(group: List[str], chosen: str) -> Dict[str, int]:
    """
    Convert a selected category into one-hot encoded dictionary.
    """
    return {item: int(item == chosen) for item in group}


def choose_diagnosis_by_ga(ga: str) -> str:
    """
    Choose maternal diagnosis conditioned on gestational age category.
    """
    if ga == "<20_weeks":
        probs = {
            "Chronic Hypertension": 0.45,
            "Normotensive": 0.45,
            "Antenatal/Peripartum Hypertension": 0.07,
            "Non-severe Gestational Hypertension": 0.02,
            "Non-severe Preeclampsia": 0.01,
        }
    elif ga == "20_33_weeks":
        probs = {
            "Non-severe Gestational Hypertension": 0.30,
            "Non-severe Preeclampsia": 0.18,
            "Chronic Hypertension": 0.18,
            "Normotensive": 0.20,
            "Severe Gestational Hypertension": 0.08,
            "Severe Preeclampsia/Eclampsia": 0.04,
            "Antenatal/Peripartum Hypertension": 0.02,
        }
    else:
        probs = {
            "Severe Preeclampsia/Eclampsia": 0.15,
            "Severe Gestational Hypertension": 0.12,
            "Non-severe Gestational Hypertension": 0.28,
            "Non-severe Preeclampsia": 0.18,
            "Chronic Hypertension": 0.12,
            "Normotensive": 0.12,
            "Antenatal/Peripartum Hypertension": 0.03,
        }

    categories = list(probs.keys())
    weights = list(probs.values())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_bp_for_diagnosis(dx: str) -> Tuple[str, str, bool]:
    """
    Generate SBP/DBP category and proteinuria based on diagnosis.
    """
    if dx == "Normotensive":
        sbp = random.choice(["<119_SBP", "120_129_SBP", "130_139_SBP"])
        dbp = random.choice(["<85_DBP", "86_90_DBP"])
        proteinuria = False

    elif dx == "Non-severe Gestational Hypertension":
        sbp = "140_160_SBP"
        dbp = random.choice(["86_90_DBP", "91_105_DBP"])
        proteinuria = False

    elif dx == "Severe Gestational Hypertension":
        sbp = ">161_SBP"
        dbp = ">106_DBP"
        proteinuria = False

    elif dx == "Non-severe Preeclampsia":
        sbp = "140_160_SBP"
        dbp = random.choice(["91_105_DBP", "86_90_DBP"])
        proteinuria = True

    elif dx == "Severe Preeclampsia/Eclampsia":
        sbp = ">161_SBP"
        dbp = ">106_DBP"
        proteinuria = True

    elif dx == "Chronic Hypertension":
        sbp = random.choice(["130_139_SBP", "140_160_SBP"])
        dbp = random.choice(["86_90_DBP", "91_105_DBP"])
        proteinuria = random.random() < 0.25

    else:
        sbp = random.choice(["130_139_SBP", "140_160_SBP"])
        dbp = random.choice(["86_90_DBP", "91_105_DBP"])
        proteinuria = False

    return sbp, dbp, proteinuria


def choose_comorbidity(dx: str) -> str:
    """
    Choose one comorbidity conditioned on diagnosis.
    """
    base = {
        "None": 0.45,
        "chronic hypertension": 0.08,
        "gestational diabetes": 0.12,
        "chronic kidney disease": 0.05,
        "asthma": 0.08,
        "AV block": 0.02,
        "lupus/antiphospholipid syndrome": 0.05,
        "depression": 0.07,
        "history of heart disease": 0.08,
    }

    if "Preeclampsia" in dx:
        base["lupus/antiphospholipid syndrome"] += 0.08
        base["chronic kidney disease"] += 0.05
        base["gestational diabetes"] += 0.05
        base["None"] -= 0.10

    if "Chronic Hypertension" in dx:
        base["chronic hypertension"] += 0.25
        base["history of heart disease"] += 0.10
        base["None"] -= 0.15

    categories = list(base.keys())
    weights = list(base.values())
    return random.choices(categories, weights=weights, k=1)[0]


def generate_clinical_note(
    dx: str,
    sbp: str,
    dbp: str,
    proteinuria: bool,
    comorbidity: str,
) -> Tuple[str, str]:
    """
    Generate synthetic narrative clinical note and anomaly text.
    """
    proteinuria_text = "Proteinuria detected." if proteinuria else "No proteinuria detected."

    note = (
        f"Pregnant patient with blood pressure category SBP {sbp.replace('_SBP', '')} "
        f"and DBP {dbp.replace('_DBP', '')}. "
        f"Clinical presentation compatible with {dx}. "
        f"{proteinuria_text}"
    )

    anomaly = comorbidity if comorbidity != "None" else "No relevant comorbidities reported."

    return note, anomaly


def generate_patient_row() -> Dict[str, object]:
    """
    Generate one synthetic patient row.
    """
    age = choose_one(AGE_OH, AGE_PROBS)
    ga = choose_one(GA_OH, GA_PROBS)
    dx = choose_diagnosis_by_ga(ga)
    sbp, dbp, proteinuria = generate_bp_for_diagnosis(dx)
    comorbidity = choose_comorbidity(dx)
    note, anomaly = generate_clinical_note(dx, sbp, dbp, proteinuria, comorbidity)

    row: Dict[str, object] = {}
    row.update(one_hot(AGE_OH, age))
    row.update(one_hot(GA_OH, ga))
    row.update(one_hot(SBP_OH, sbp))
    row.update(one_hot(DBP_OH, dbp))

    row["Maternal_Diagnosis"] = dx
    row["Clinical_notes"] = note
    row["Anomaly_1"] = anomaly
    row["Comorbility"] = comorbidity
    row["Proteinuria"] = int(proteinuria)

    return row


def generate_population_base(
    n: int = 50,
    seed: int = 42,
    patient_prefix: str = "PT",
) -> pd.DataFrame:
    """
    Generate a synthetic population dataframe.
    """
    seed_everything(seed)

    rows = []
    for i in range(n):
        row = generate_patient_row()
        row["patient_id"] = f"{patient_prefix}{i + 1:05d}"
        rows.append(row)

    return pd.DataFrame(rows)