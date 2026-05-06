import json
import re
import time
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from tqdm import tqdm

from src.obs_hypertension.screening.llm.gpt_client import query_gpt
from src.obs_hypertension.screening.config.llm import REQUEST_SLEEP
from src.obs_hypertension.screening.config.inclusion_exclusion_prompt import (SYSTEM_PROMPT, 
                                                                         USER_PROMPT_TEMPLATE)

# ----------------------------
# Text cleaning helpers
# ----------------------------

_CUTOFF_PATTERNS = [
    r"\bAcknowledg(e)?ments?\b",
    r"\bReferences\b",
    r"\bBibliography\b",
    r"\bFunding\b",
    r"\bConflicts of Interest\b",
]

def strip_tail_sections(text: str) -> str:
    """Cut text at the first occurrence of common tail sections."""
    if not text:
        return text
    t = re.sub(r"\s+", " ", text).strip()
    earliest = None
    for pat in _CUTOFF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            earliest = m.start() if earliest is None else min(earliest, m.start())
    return t[:earliest].strip() if earliest is not None else t


# ----------------------------
# Output normalization helpers
# ----------------------------

def ns(val: Any) -> str:
    if val is None:
        return "not specified"
    s = str(val).strip()
    return s if s else "not specified"

def ensure_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]

def normalize_keep_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes the LLM output to a predictable schema.
    """
    if not isinstance(data, dict):
        return {"keep": False, "reason": "Model output was not a JSON object."}

    if "keep" not in data:
        return {"keep": False, "reason": "Missing 'keep' field in model output."}

    keep = bool(data.get("keep", False))

    if not keep:
        return {
            "keep": False,
            "reason": ns(data.get("reason", "not specified")),
        }

    cohort = data.get("cohort_summary") if isinstance(data.get("cohort_summary"), dict) else {}

    out: Dict[str, Any] = {
        "keep": True,
        "title": ns(data.get("title")),
        "authors": ns(data.get("authors")),
        "doi": ns(data.get("doi")),
        "year": ns(data.get("year")),
        "study_design": ns(data.get("study_design")),
        "comparative_groups": [],
        "cohort_summary": {
            "cohort_description": ns(cohort.get("cohort_description")),
            "maternal_diagnosis": ns(cohort.get("maternal_diagnosis")),
            "pressure_info": ns(cohort.get("pressure_info")),
            "gestational_age_info": ns(cohort.get("gestational_age_info")),
            "maternal_age_info": ns(cohort.get("maternal_age_info")),
            "sample_size_per_group": ns(cohort.get("sample_size_per_group")),
            "stratification_sample_per_group": ns(cohort.get("stratification_sample_per_group")),
        },
        "treatment_description": [],
        "outcome_description": [],
        "quantitative_evidence": [],
        "evidence_snippets": [str(x).strip() for x in ensure_list(data.get("evidence_snippets")) if str(x).strip()],
    }

    for g in ensure_list(data.get("comparative_groups")):
        if isinstance(g, dict):
            out["comparative_groups"].append({
                "group_label": ns(g.get("group_label")),
                "comparison_against": ns(g.get("comparison_against")),
            })

    for t in ensure_list(data.get("treatment_description")):
        if isinstance(t, dict):
            out["treatment_description"].append({
                "group_label": ns(t.get("group_label")),
                "drug_used": ns(t.get("drug_used")),
                "route_of_administration": ns(t.get("route_of_administration")),
                "dosage": ns(t.get("dosage")),
                "treatment_schedule": ns(t.get("treatment_schedule")),
            })

    for o in ensure_list(data.get("outcome_description")):
        if isinstance(o, dict):
            out["outcome_description"].append({
                "maternal_outcomes": ns(o.get("maternal_outcomes")),
                "fetal_outcomes": ns(o.get("fetal_outcomes")),
                "anomalies_outcomes": ns(o.get("anomalies_outcomes")),
                "main_findings": ns(o.get("main_findings")),
            })

    for e in ensure_list(data.get("quantitative_evidence")):
        if isinstance(e, dict):
            out["quantitative_evidence"].append({
                "metric": ns(e.get("metric")),
                "value": ns(e.get("value")),
                "comparison": ns(e.get("comparison")),
                "statistical_significance": ns(e.get("statistical_significance")),
            })

    return out


# ----------------------------
# Main public function (modular API)
# ----------------------------

def apply_llm_filter_2(
    df: pd.DataFrame,
    max_docs: Optional[int] = None,
    save_raw_jsonl: bool = False,
    raw_jsonl_path: Optional[str] = None,
    text_col: str = "text",
    id_col: str = "corpusid",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply LLM screening + evidence extraction.

    Returns:
      kept_df     : accepted papers with extracted fields
      rejected_df : rejected papers with reason
      errors_df   : parsing/API errors

    Notes:
      - Stores nested lists/dicts as JSON strings for CSV friendliness.
      - Uses tail-cut to reduce references noise.
    """
    kept_rows, rejected_rows, error_rows = [], [], []
    df_iter = df.head(max_docs) if max_docs else df

    raw_fh = None
    if save_raw_jsonl:
        path = raw_jsonl_path or "llm_screening_raw.jsonl"
        raw_fh = open(path, "w", encoding="utf-8")

    for _, row in tqdm(df_iter.iterrows(), total=len(df_iter)):
    # for i, row in tqdm(df_iter.iterrows(), total=len(df_iter)):
        #if i % 5 == 0 and i > 0:
        corpusid = row.get(id_col)
        text_raw = str(row.get(text_col, "")).replace("\n", " ")
        text = strip_tail_sections(text_raw)

        #user_prompt = USER_PROMPT_TEMPLATE.format(paper_text=text)
        user_prompt = USER_PROMPT_TEMPLATE.replace("{paper_text}", text)

        try:
            #data = query_gpt(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
            raw_data  = query_gpt(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

            if raw_fh is not None:
                raw_fh.write(json.dumps({"corpusid": corpusid, "llm_output": raw_data}, ensure_ascii=False) + "\n")

            data = normalize_keep_payload(raw_data)
            if data["keep"]:
                kept_rows.append({
                    "corpusid": corpusid,
                    "title": data.get("title"),
                    "authors": data.get("authors"),
                    "year": data.get("year"),
                    "doi": data.get("doi"),
                    "study_design": data.get("study_design"),
                    "comparative_groups": json.dumps(data.get("comparative_groups", []), ensure_ascii=False),
                    "cohort_summary": json.dumps(data.get("cohort_summary", {}), ensure_ascii=False),
                    "treatment_description": json.dumps(data.get("treatment_description", []), ensure_ascii=False),
                    "outcome_description": json.dumps(data.get("outcome_description", []), ensure_ascii=False),
                    "quantitative_evidence": json.dumps(data.get("quantitative_evidence", []), ensure_ascii=False),
                    "evidence_snippets": json.dumps(data.get("evidence_snippets", []), ensure_ascii=False),
                    "raw_llm_json": json.dumps(raw_data, ensure_ascii=False),
                    "text": text,
                })
            else:
                rejected_rows.append({
                    "corpusid": corpusid,
                    "reason": data.get("reason", "not specified"),
                    "text": text,
                })

        except Exception as e:
            error_rows.append({
                "corpusid": corpusid,
                "error": str(e),
                "raw_llm_json": json.dumps(raw_data, ensure_ascii=False) if 'raw_data' in locals() else None,
                "text": text[:2000],
            })

        time.sleep(REQUEST_SLEEP)

    if raw_fh is not None:
        raw_fh.close()

    return pd.DataFrame(kept_rows), pd.DataFrame(rejected_rows), pd.DataFrame(error_rows)
