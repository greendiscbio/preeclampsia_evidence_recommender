"""Path configuration for the screening pipeline."""

from src.obs_hypertension.config import REPOSITORY_ROOT
from src.obs_hypertension.filtering.config.paths import (
    OUTPUT_BEFORE80,
    OUTPUT_AFTER80,
)


# ---------------------------------------------------------------------
# Inputs produced by the filtering pipeline
# ---------------------------------------------------------------------

INPUT_BEFORE80 = (
    OUTPUT_BEFORE80
    / "from0_to80_regex_density_biobert_thr_P95_v2"
    / "stage3_semantic"
)

INPUT_AFTER80 = (
    OUTPUT_AFTER80
    / "from81_to269_regex_density_biobert_thr_P95_v2"
    / "stage3_semantic"
)


# ---------------------------------------------------------------------
# Screening outputs
# ---------------------------------------------------------------------

SCREENING_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "screening"

OUTPUT_ROOT = SCREENING_OUTPUT_ROOT
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
