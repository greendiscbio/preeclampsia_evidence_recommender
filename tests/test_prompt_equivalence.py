"""Regression tests for prompt externalization.

These tests protect the scientific behavior of the pipeline while prompt
storage is moved out of Python modules. Equality is intentionally strict:
any whitespace or Unicode difference fails the test.
"""

import json

from src.obs_hypertension.extraction.config.standardization_prompt import (
    SYSTEM_PROMPT as LEGACY_STANDARDIZATION_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE as LEGACY_STANDARDIZATION_USER_PROMPT_TEMPLATE,
)
from src.obs_hypertension.screening.config.inclusion_exclusion_prompt import (
    SYSTEM_PROMPT as LEGACY_SCREENING_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE as LEGACY_SCREENING_USER_PROMPT_TEMPLATE,
)
from src.obs_hypertension.utils.prompts import (
    SCREENING_SYSTEM_PROMPT,
    SCREENING_USER_PROMPT_TEMPLATE,
    STANDARDIZATION_SYSTEM_PROMPT,
    STANDARDIZATION_USER_PROMPT_TEMPLATE,
)


def test_prompt_files_are_character_for_character_equivalent_to_legacy():
    assert SCREENING_SYSTEM_PROMPT == LEGACY_SCREENING_SYSTEM_PROMPT
    assert SCREENING_USER_PROMPT_TEMPLATE == LEGACY_SCREENING_USER_PROMPT_TEMPLATE
    assert STANDARDIZATION_SYSTEM_PROMPT == LEGACY_STANDARDIZATION_SYSTEM_PROMPT
    assert STANDARDIZATION_USER_PROMPT_TEMPLATE == LEGACY_STANDARDIZATION_USER_PROMPT_TEMPLATE


def test_screening_final_message_is_unchanged():
    paper_text = "Synthetic test paper. BP 170/110 mmHg."
    legacy = LEGACY_SCREENING_USER_PROMPT_TEMPLATE.replace("{paper_text}", paper_text)
    externalized = SCREENING_USER_PROMPT_TEMPLATE.replace("{paper_text}", paper_text)
    assert externalized == legacy


def test_standardization_final_message_is_unchanged():
    raw_json = {"study": "synthetic", "dose": "10 mg", "result": "BP controlled"}
    serialized = json.dumps(raw_json, ensure_ascii=False)
    legacy = LEGACY_STANDARDIZATION_USER_PROMPT_TEMPLATE.replace("__RAW_JSON__", serialized)
    externalized = STANDARDIZATION_USER_PROMPT_TEMPLATE.replace("__RAW_JSON__", serialized)
    assert externalized == legacy
