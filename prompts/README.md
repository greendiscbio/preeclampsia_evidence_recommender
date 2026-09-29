# LLM prompts and execution metadata

This directory exposes the exact prompt templates used by the two LLM-assisted stages of the study. The runtime loader reads these files verbatim; no whitespace normalization or `.strip()` operation is applied.

## Screening / eligibility and raw evidence extraction

- System prompt: `screening_system_prompt.md`
- User prompt template: `screening_user_prompt.md`
- Runtime consumer: `src/obs_hypertension/screening/llm/inclusion_filter.py`
- API/model identifier: `gpt-4o`
- Temperature: `0`
- Response format: JSON object
- Maximum output tokens: `4000`
- Request timeout: `120` seconds
- Maximum retries: `3`
- Retry sleep: starts at `2.0` seconds and increases linearly by attempt
- Inter-request sleep configured by the screening pipeline: `0.5` seconds
- Runtime placeholder: `{paper_text}`

The repository configuration uses the OpenAI API model identifier `gpt-4o`; a dated model snapshot was not pinned in the original experiment. This is reported explicitly rather than retrospectively assigning a checkpoint that was not recorded.

## Evidence standardization

- System prompt: `standardization_system_prompt.md`
- User prompt template: `standardization_user_prompt.md`
- Runtime consumer: `src/obs_hypertension/extraction/llm/gpt_standardizer.py`
- API/model identifier: `gpt-4o`
- Temperature: `0`
- Response format: JSON object
- Maximum output tokens: not explicitly set in the original standardization API call (provider default applied)
- Maximum retries: `3`
- Retry sleep after failed calls: `1.5` seconds
- Runtime placeholder: `__RAW_JSON__`

As above, the original implementation recorded the API model identifier `gpt-4o` but did not pin a dated snapshot. No unrecorded checkpoint or token limit has been added retrospectively.

## Reproducibility note

The prompt externalization is a storage/refactoring change only. The prompt text and runtime substitution logic are preserved. Regression tests in `tests/test_prompt_equivalence.py` compare the externalized prompt text character-for-character with the legacy Python constants and also compare the final messages after representative runtime substitution.
