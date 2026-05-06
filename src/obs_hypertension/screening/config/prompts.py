from pathlib import Path

# Ruta absoluta basada en ESTE archivo
BASE_DIR = Path(__file__).resolve().parent

PROMPT_PATH = BASE_DIR / "inclusion_exclusion_prompt.txt"

if not PROMPT_PATH.exists():
    raise FileNotFoundError(
        f"Prompt file not found at {PROMPT_PATH}"
    )

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

