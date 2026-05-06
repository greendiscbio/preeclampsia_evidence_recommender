import json

from src.obs_hypertension.standardization.config.config import MISSING_VALUE


def safe_json_loads(text):

    try:
        return json.loads(text)
    except Exception:
        return {}



def safe_get(value):

    if value is None or value == "":
        return MISSING_VALUE

    return value