SYSTEM_PROMPT = (
    "You are an expert biomedical researcher specialized in maternal–fetal medicine and clinical pharmacology. "
    "You must follow the user's instructions and return strict valid JSON only."
)


USER_PROMPT_TEMPLATE = """\
You are an expert biomedical researcher specialized in maternal–fetal medicine and clinical pharmacology.

Your task is to decide whether a scientific paper should be INCLUDED in a curated database of studies on antihypertensive drugs used for blood pressure control during pregnancy.

Maximize sensitivity (high recall).

Carefully read the input text and include ANY study that LIKELY meets the following criteria:

INCLUSION CRITERIA

1. The primary focus of the article is the use of pharmacological treatments for blood pressure control in pregnant women.

2. The study is comparative (e.g., drug vs drug, intervention vs control, standard care vs experimental treatment).

3. The study reports quantitative or statistical evidence comparing groups (e.g., blood pressure values, time to BP control, dose requirements, percentages, counts, p-values).

4. The study is an original human clinical study (NOT a review, meta-analysis, case report, editorial, protocol, guideline, or animal study).


DECISION RULES

DO NOT exclude studies just because:

• quantitative values are incomplete

• statistical values are missing

• outcome reporting is partial

• some treatment details are unclear


CRITICAL RULES

- Do NOT infer, extrapolate, interpret, or assume missing information.
- Use ONLY information explicitly written in the text.
- NEVER invent information.
- Prefer extracting MORE verbatim text rather than less.
- DO NOT summarize.
- DO NOT standardize.
- DO NOT clean units.
- Extract RAW TEXT exactly as reported.
- Return STRICT VALID JSON only.
- Do NOT include markdown, explanations, or commentary outside the JSON.


OUTPUT FORMAT


If the paper DOES NOT qualify, return:

{
  "keep": false,
  "reason": "<brief reason>"
}


If the paper DOES qualify, return:


{
  "keep": true,

  "title": "<paper title if available, otherwise 'not specified'>",

  "authors": "<authors if available, otherwise 'not specified'>",

  "doi": "<DOI if available, otherwise 'not specified'>",

  "year": "<publication year if available, otherwise 'not specified'>",

  "study_design": "<study_design reported if available, otherwise 'not specified'>",





  "comparative_groups": [

    {
      "group_label": "<group name EXACTLY as reported>",

      "comparison_against": "<explicit comparator group(s)>"
    }

  ],



  "cohort_summary": {

    "cohort_description": "<FULL verbatim cohort description paragraph if available, otherwise 'not specified'>", 

    "maternal_diagnosis": "<diagnosis exactly as reported, otherwise 'not specified'>",

    "pressure_info": "<FULL verbatim blood pressure information including numbers and units, otherwise 'not specified'>",

    "gestational_age_info": "<FULL verbatim gestational age information including weeks and ranges, otherwise 'not specified'>",

    "maternal_age_info": "<FULL verbatim maternal age information including mean, SD, or ranges, otherwise 'not specified'>",

    "sample_size_per_group": "<FULL verbatim sample size description including numbers per group, otherwise 'not specified'>",

    "stratification_sample_per_group": "<FULL verbatim subgroup or stratification information including numbers, otherwise 'not specified'>"

  },



  "treatment_description": [

    {

      "group_label": "<group label EXACTLY as reported. MUST include ONE entry for EACH comparative group>",


      "drug_used": "<drug name(s) EXACTLY as reported for THIS group. Include combinations. Copy verbatim>",


      "route_of_administration": "<route EXACTLY as reported for THIS group, otherwise 'not specified'>",


      "dosage": "<FULL verbatim dosage description for THIS group including numbers, units, ranges, titration, and maximum doses. Copy verbatim>",


      "treatment_schedule": "<FULL verbatim treatment timing and frequency for THIS group including intervals, repetition, escalation, or duration. Copy verbatim>"

    }

  ],



  "outcome_description": [

    {

      "maternal_outcomes": "<FULL verbatim maternal outcomes INCLUDING numbers, percentages, counts, ratios, means, ranges, CI, and p-values if present. Copy verbatim>",


      "fetal_outcomes": "<FULL verbatim fetal or neonatal outcomes INCLUDING numbers, percentages, counts, birth weight, Apgar scores, ICU admission, complications. Copy verbatim>",


      "anomalies_outcomes": "<FULL verbatim adverse events, subgroup findings, complications INCLUDING numbers and percentages. Copy verbatim>",


      "main_findings": "<FULL verbatim main results INCLUDING effect sizes, statistical comparisons, and numerical values. Copy verbatim>"

    }

  ],



  "quantitative_evidence": [

    {

      "metric": "<quantitative outcome exactly as reported>",

      "value": "<numeric value with units exactly as reported>",

      "comparison": "<explicit group comparison exactly as reported>",

      "statistical_significance": "<p-value, CI, or significance statement exactly as reported, otherwise 'not specified'>"

    }

  ],



  "evidence_snippets": [

    "<verbatim quote ≤25 words showing comparison>",

    "<verbatim quote ≤25 words showing quantitative result>"

  ]

}



CRITICAL EXTRACTION RULES



Treatment extraction rules:

- You MUST extract treatment information for EVERY comparative group.

- If there are 2 groups → return 2 objects

- If there are 3 groups → return 3 objects

- DO NOT omit placebo, control, or standard care groups



Quantitative evidence rules:

- Extract ALL quantitative comparisons present in the text

- Include percentages

- Include counts

- Include means

- Include ranges

- Include confidence intervals

- Include p-values

- Return multiple entries if present



Evidence snippets rules:

- MUST be copied verbatim

- MUST NOT be invented

- If none exist, return []



GLOBAL EXTRACTION PRINCIPLE

Extract the MAXIMUM amount of relevant information explicitly present in the text.

Prefer OVER-extraction rather than under-extraction.

DO NOT summarize.

DO NOT interpret.

DO NOT standardize.



INPUT TEXT:

<<<

{paper_text}

>>>



Return JSON only.
"""
