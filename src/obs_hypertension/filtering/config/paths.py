"""Path configuration for the literature-filtering pipeline."""

from src.obs_hypertension.config import REPOSITORY_ROOT, get_env_path


# ---------------------------------------------------------------------
# External data
# ---------------------------------------------------------------------

DATA_0_80 = get_env_path("S2ORC_DATA_0_80")
DATA_81_269 = get_env_path("S2ORC_DATA_81_269")

# ---------------------------------------------------------------------
# Repository-managed outputs
# ---------------------------------------------------------------------

PIPELINE_ROOT = REPOSITORY_ROOT / "outputs" / "filtering"

OUTPUT_ROOT = PIPELINE_ROOT
OUTPUT_BEFORE80 = OUTPUT_ROOT / "before80"
OUTPUT_AFTER80 = OUTPUT_ROOT / "after80"

for path in (OUTPUT_ROOT, OUTPUT_BEFORE80, OUTPUT_AFTER80):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# External/local Spark runtime storage
# ---------------------------------------------------------------------

SPARK_CHECKPOINT_DIR = get_env_path(
    "SPARK_CHECKPOINT_DIR",
    REPOSITORY_ROOT / ".cache" / "spark_checkpoints",
)


def require_path(path, variable_name):
    """Raise a clear error when a required external path is not configured."""
    if path is None:
        raise RuntimeError(
            f"{variable_name} is not configured. "
            f"Set it in the environment or in a local .env file."
        )

    if not path.exists():
        raise FileNotFoundError(
            f"{variable_name} does not exist: {path}"
        )

    return path


def print_paths_summary():
    print("Repository root:", REPOSITORY_ROOT)
    print("Data 0-80:", DATA_0_80)
    print("Data 81-269:", DATA_81_269)
    print("Output before80:", OUTPUT_BEFORE80)
    print("Output after80:", OUTPUT_AFTER80)
    print("Spark checkpoint:", SPARK_CHECKPOINT_DIR)
