"""Path configuration for the standardization pipeline."""

from src.obs_hypertension.config import REPOSITORY_ROOT


STANDARDIZATION_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "outputs" / "standardization"
)

INPUT_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "extraction"
    / "standardized_evidence.csv"
)

ARM_OUTPUT = (
    STANDARDIZATION_OUTPUT_ROOT
    / "arm_level_dataset.csv"
)

FINAL_OUTPUT = (
    STANDARDIZATION_OUTPUT_ROOT
    / "arm_dataset_standardized.csv"
)

STANDARDIZATION_OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)
