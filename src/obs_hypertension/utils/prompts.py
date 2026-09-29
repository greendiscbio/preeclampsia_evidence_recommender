"""Central loader for externally versioned LLM prompts.

Prompt files live at repository root under ``prompts/`` so that the exact
text sent to the model is directly inspectable without navigating Python
configuration modules. ``read_text`` is intentionally used without
``strip()`` or other normalization: whitespace is part of the prompt and
must be preserved exactly for reproducibility.
"""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = REPOSITORY_ROOT / "prompts"


def load_prompt(filename: str) -> str:
    """Load one prompt verbatim as UTF-8 text."""
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


SCREENING_SYSTEM_PROMPT = load_prompt("screening_system_prompt.md")
SCREENING_USER_PROMPT_TEMPLATE = load_prompt("screening_user_prompt.md")
STANDARDIZATION_SYSTEM_PROMPT = load_prompt("standardization_system_prompt.md")
STANDARDIZATION_USER_PROMPT_TEMPLATE = load_prompt("standardization_user_prompt.md")
