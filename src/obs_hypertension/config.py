"""Repository-wide configuration utilities."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Repository root:
# preeclampsia_evidence_recommender/
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Load an optional local .env file.
load_dotenv(REPOSITORY_ROOT / ".env")


def get_env_path(variable_name: str, default=None):
    """
    Return an environment variable as a Path.

    Parameters
    ----------
    variable_name:
        Name of the environment variable.
    default:
        Optional fallback path.

    Returns
    -------
    pathlib.Path or None
    """
    value = os.getenv(variable_name)

    if value:
        return Path(value).expanduser()

    if default is not None:
        return Path(default).expanduser()

    return None
