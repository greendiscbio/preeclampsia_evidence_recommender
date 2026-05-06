import json
import re
import time

from openai import OpenAI

from src.obs_hypertension.extraction.config.openai import (
    OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE
)

client = OpenAI(api_key=OPENAI_API_KEY)



def _extract_json(content: str):

    if not content:
        raise ValueError("Empty LLM response")

    txt = content.strip()

    # remove markdown fences
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)

    # extract JSON object
    match = re.search(r"\{.*\}\s*$", txt, flags=re.S)

    if not match:
        raise ValueError("No JSON found in LLM response")

    candidate = match.group(0)

    return json.loads(candidate)



def query_standardizer(raw_json: dict, max_retries=3):

    from src.obs_hypertension.extraction.config.standardization_prompt import (
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE
    )

    # user_prompt = USER_PROMPT_TEMPLATE.format(
    #     raw_json=json.dumps(raw_json, ensure_ascii=False)
    # )

    user_prompt = USER_PROMPT_TEMPLATE.replace(
    "__RAW_JSON__",
    json.dumps(raw_json, ensure_ascii=False)
)


    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                temperature=TEMPERATURE,

                response_format={"type": "json_object"},  # 

                messages=[

                    {"role": "system", "content": SYSTEM_PROMPT},

                    {"role": "user", "content": user_prompt}

                ]

            )

            content = response.choices[0].message.content

            return _extract_json(content)

        except Exception as e:

            print(f"Retry {attempt+1} error:", e)

            time.sleep(1.5)

    raise RuntimeError("Standardizer failed after retries")

