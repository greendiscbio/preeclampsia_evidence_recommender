from pathlib import Path

PROJECT_ROOT = Path("/home/juandiegoarevalo/hypertension_project")

INPUT_PARQUET = (
    PROJECT_ROOT /
    "filtering_high_recall_semantic/outputs/after80/from81_to269"
)

OUTPUT_ROOT = PROJECT_ROOT / "llm_inclusion_exclusion/outputs"

ACCEPTED_DIR = OUTPUT_ROOT / "accepted"
REJECTED_DIR = OUTPUT_ROOT / "rejected"
LOG_DIR = OUTPUT_ROOT / "logs"

for p in [ACCEPTED_DIR, REJECTED_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
