from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = PROJECT_ROOT / "llm_inclusion_exclusion/outputs/accepted/llm_filtered_all.csv"

OUTPUT_DIR = PROJECT_ROOT / "llm_parameter_extraction/outputs"
RAW_JSON_DIR = OUTPUT_DIR / "raw_json"
CSV_DIR = OUTPUT_DIR / "extracted_csv"

for p in [RAW_JSON_DIR, CSV_DIR]:
    p.mkdir(parents=True, exist_ok=True)
