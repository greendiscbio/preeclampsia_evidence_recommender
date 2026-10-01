"""Path configuration for the extraction pipeline."""

from src.obs_hypertension.config import REPOSITORY_ROOT


EXTRACTION_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "extraction"

INPUT_CSV = (
    REPOSITORY_ROOT
    / "outputs"
    / "screening"
    / "llm_kept.csv"
)

OUTPUT_CSV = EXTRACTION_OUTPUT_ROOT / "standardized_evidence.csv"

EXTRACTION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
