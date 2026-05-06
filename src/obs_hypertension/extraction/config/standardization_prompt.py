SYSTEM_PROMPT = (
    "You are a biomedical clinical data standardization system specialized in maternal–fetal medicine. "
    "Your role is to convert raw extracted study data into precise, structured, standardized variables suitable for statistical modeling and machine learning. "
    "You preserve clinical meaning while enforcing standardized numeric formats and correct clinical interpretation. "
    "You must distinguish between clinical BENEFIT and clinical HARM. "
    "Return ONLY valid JSON."
)



USER_PROMPT_TEMPLATE = """

You will receive RAW extracted JSON from a scientific paper.

Your task is to:

1. STANDARDIZE the data into structured format
2. NORMALIZE numeric ranges and ± format
3. STRUCTURE all outcomes in LONG format
4. IDENTIFY the WINNER group based on TRUE CLINICAL BENEFIT supported by statistical evidence



========================================
CRITICAL GENERAL RULES
========================================

• DO NOT invent data

• DO NOT assume missing values

• ONLY use explicitly reported information

• If missing → use "not specified"

• Return ONLY valid JSON



========================================
CRITICAL CLINICAL INTERPRETATION RULES
========================================

You MUST distinguish between:

CLINICAL BENEFIT

vs

CLINICAL HARM



CLINICAL BENEFIT means:

• reduced risk

• reduced incidence of complications

• improved outcomes

• better efficacy

• safer profile

• protection against adverse outcomes



CLINICAL HARM means:

• increased risk

• increased incidence of complications

• worse outcomes

• adverse effects

• higher odds ratio for negative outcome



CRITICAL:

A treatment associated with INCREASED RISK is NOT a winner.



Example:

"five-fold increased risk of SGA"

This is HARM.

This group is NOT the winner.

The comparator with LOWER risk is the winner.

========================================
TREATMENT-LEVEL WINNER PRIORITY RULES
========================================

CRITICAL PRINCIPLE:

The WINNER must be a PHARMACOLOGICAL TREATMENT, not a disease severity subgroup.



WINNER must represent:

• drug

• drug combination

• therapeutic regimen



NOT allowed as winner:

• severity subgroup (example: mild PIH group, severe group)

• demographic subgroup

• risk subgroup

• stratification subgroup



These are NOT treatments.



========================================
PRIORITY ORDER FOR WINNER SELECTION
========================================


LEVEL 1 — Treatment comparison (HIGHEST PRIORITY)

Example:

Drug A vs Drug B

Winner = Drug with better clinical outcome



LEVEL 2 — Treatment vs control

Example:

Treatment vs placebo

Winner = treatment if benefit



LEVEL 3 — Before vs after treatment (within same group)

Example:

"Blood pressure significantly decreased after treatment"

Winner = the treatment itself

NOT the subgroup



Correct:

winner = "magnesium sulfate and nifedipine"



Incorrect:

winner = "moderate PIH group"



LEVEL 4 — subgroup differences (LOWEST PRIORITY)

Only use subgroup as winner IF:

different treatments are used in different subgroups



If same treatment is used in all subgroups:

Winner = treatment

NOT subgroup



========================================
CRITICAL INFERENCE RULE
========================================

If:

same drug is used

AND

clinical outcomes improved after treatment

AND

p < 0.05


Winner MUST be the DRUG or DRUG COMBINATION



========================================
WINNER NAME FORMAT RULE
========================================


Winner must be written as:

drug name


Examples:

"magnesium sulfate"

"nifedipine"

"magnesium sulfate and nifedipine"

"labetalol"

"metformin"



NOT:

"group"

"patients"

"moderate group"

"severe group"



========================================
WINNER REASON RULE FOR BEFORE-AFTER STUDIES
========================================


Use this format:

"Treatment with [drug name] significantly improved [outcome] (p < 0.05), indicating clinical benefit."



Example:

"Treatment with magnesium sulfate and nifedipine significantly reduced blood pressure (P < 0.05), indicating clinical benefit."



========================================
WHEN winner MUST BE "not specified"
========================================


If:

no drug benefit demonstrated

OR

only observational stratification

OR

no statistically significant improvement


winner = "not specified"

========================================
WINNER EXTRACTION RULES
========================================


WINNER definition:

The group with statistically significant CLINICAL BENEFIT compared to control or comparator.



Winner MUST satisfy BOTH:


CONDITION 1 — statistical significance

AND

CONDITION 2 — clinical benefit direction



DO NOT select winner based on statistical significance alone.

Direction of effect is mandatory.



========================================
HOW TO DETERMINE DIRECTION OF BENEFIT
========================================


If outcome is NEGATIVE (example: death, complication, SGA, disease, adverse outcome):


LOWER value = BENEFIT

HIGHER value = HARM



If outcome is POSITIVE (example: survival, treatment success, recovery):


HIGHER value = BENEFIT

LOWER value = HARM



If text explicitly states:

"increased risk"

"increased odds"

"higher incidence"

This is HARM.



The comparator with LOWER risk is the WINNER.



========================================
WHEN winner MUST BE "not specified"
========================================


If:

• only harm is reported

AND

• benefit group not explicitly quantified


OR

• direction unclear


Then:

winner = "not specified"



========================================
Winner_reason RULES
========================================


Must explain WHY the winner has benefit.

Must include:

• comparison

• outcome

• p-value if available

• direction of benefit



If study only shows harm increase:

Winner_reason must state:

"Increased risk observed in treatment group; comparator has safer profile"



========================================
statistical_significance_vs_control RULES
========================================


Answer "Yes" ONLY if:

• comparison vs control exists

AND

• p-value reported


Otherwise:

"No"



========================================
NUMERIC STANDARDIZATION RULES
========================================


Allowed formats:


Range:

"min-max"


Mean ± SD:

"use ± symbol"


Convert unicode:

\u00b1 → ±


Remove units from range fields.



========================================
OUTCOME CLASSIFICATION RULES
========================================


maternal_outcome_type

fetal_outcome_type

anomaly_outcome_type



Allowed values:


risk

safe

profile



risk

Use when:

• increased risk

• complication

• adverse outcome



safe

Use when:

• reduced risk

• improved outcome

• no adverse effects



profile

Use when:

• subgroup

• stratification

• dose comparison



========================================
OUTCOME STRUCTURING RULES
========================================


Extract:

• outcome name

• metric

• value

• unit

• comparison

• p_value

• outcome_type



Create ONE object per outcome.



========================================
INPUT RAW JSON
========================================

<<<

__RAW_JSON__

>>>



========================================
OUTPUT FORMAT
========================================

{

"cohort": {

"cohort_description": "",

"maternal_clinical_diagnosis": "",

"cohort_stratification": "",

"maternal_age_range": "",

"gestational_age_weeks_range": "",

"diastolic_pressure_range": "",

"systolic_pressure_range": ""

},



"arms": [

{

"group_label": "",

"sample_size": "",


"therapy": [

{

"drug": "",
"route": "",
"dose": "",
"frequency": "",
"duration": ""

},

{

"drug": "",
"route": "",
"dose": "",
"frequency": "",
"duration": ""

}

],




"maternal_outcomes": [

{

"name": "",

"metric": "",

"value": "",

"unit": "",

"comparison": "",

"p_value": "",

"outcome_type": ""

}

],



"fetal_outcomes": [

{

"name": "",

"metric": "",

"value": "",

"unit": "",

"comparison": "",

"p_value": "",

"outcome_type": ""

}

],



"anomaly_outcomes": [

{

"name": "",

"metric": "",

"value": "",

"unit": "",

"comparison": "",

"p_value": "",

"outcome_type": ""

}

]

}

],



"main_findings": "",



"study_interpretation": {

"winner": "",

"winner_reason": "",

"statistical_significance_vs_control": ""

}



}



Return JSON only.

"""