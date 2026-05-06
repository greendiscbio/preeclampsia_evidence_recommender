from src.obs_hypertension.standardization.config.config import (
    MAX_THERAPIES,
    MAX_OUTCOMES,
    MISSING_VALUE
)

from src.obs_hypertension.standardization.config.utils import safe_get



def flatten_therapy(therapy: dict, idx: int):

    return {

        f"drug_{idx}": safe_get(therapy.get("drug")),

        f"route_{idx}": safe_get(therapy.get("route")),

        f"dose_{idx}": safe_get(therapy.get("dose")),

        f"frequency_{idx}": safe_get(therapy.get("frequency")),

        f"duration_{idx}": safe_get(therapy.get("duration")),
    }



def flatten_all_therapies(therapies: list):

    row = {}

    for i in range(MAX_THERAPIES):

        if i < len(therapies):

            row.update(flatten_therapy(therapies[i], i + 1))

        else:

            row.update(flatten_therapy({}, i + 1))

    return row



def flatten_outcomes(outcomes: list, prefix: str):

    row = {}

    for i in range(MAX_OUTCOMES):

        if i < len(outcomes):

            o = outcomes[i]

            row.update({

                f"{prefix}_outcome_name_{i+1}": safe_get(o.get("name")),
                f"{prefix}_outcome_metric_{i+1}": safe_get(o.get("metric")),
                f"{prefix}_outcome_value_{i+1}": safe_get(o.get("value")),
                f"{prefix}_outcome_unit_{i+1}": safe_get(o.get("unit")),
                f"{prefix}_outcome_type_{i+1}": safe_get(o.get("outcome_type")),
                f"{prefix}_outcome_comparison_{i+1}": safe_get(o.get("comparison")),
                f"{prefix}_outcome_p_value_{i+1}": safe_get(o.get("p_value")),

            })

        else:

            row.update({

                f"{prefix}_outcome_name_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_metric_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_value_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_unit_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_type_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_comparison_{i+1}": MISSING_VALUE,
                f"{prefix}_outcome_p_value_{i+1}": MISSING_VALUE,

            })

    return row