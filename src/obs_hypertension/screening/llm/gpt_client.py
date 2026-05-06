import json
import re
import time
from typing import Dict, Any

from openai import OpenAI
from src.obs_hypertension.screening.config.llm import (
    OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def query_gpt(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """
    Production-grade GPT query with guaranteed valid JSON output.
    """

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                temperature=TEMPERATURE,

                response_format={"type": "json_object"},

                max_tokens=max_tokens,

                timeout=120,

                messages=[

                    {"role": "system", "content": system_prompt},

                    {"role": "user", "content": user_prompt},

                ],

            )

            content = response.choices[0].message.content

            return json.loads(content)


        except Exception as e:

            print(f"[WARN] GPT attempt {attempt+1} failed: {str(e)}")

            if attempt == max_retries - 1:

                return {
                    "keep": False,
                    "error": f"LLM error after retries: {str(e)}"
                }

            time.sleep(retry_sleep * (attempt + 1))