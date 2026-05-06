"""
therapy_standardizer.py

Standardizes therapy-related columns:

- drug_*
- route_*
- winner_*

This module DOES NOT read or write files.
It only transforms a DataFrame.

Used by master_standardization_pipeline.py
"""

import pandas as pd
import re
import unicodedata


# ============================================================
# TEXT NORMALIZATION HELPERS
# ============================================================

def _strip_accents(text: str) -> str:

    return "".join(

        c for c in unicodedata.normalize("NFKD", text)

        if not unicodedata.combining(c)

    )


def _norm_text(value) -> str:

    text = str(value).strip()

    text = _strip_accents(text)

    text = text.lower()

    text = text.replace("β", "beta").replace("Î²", "beta")

    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)

    text = re.sub(r"[/,;:]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _pretty(text: str) -> str:

    if not text:

        return text

    return text[0].upper() + text[1:]


# ============================================================
# DRUG STANDARDIZATION
# ============================================================

NULLISH = {

    "not specified",
    "unspecified",
    "na",
    "n/a",
    "none",
    "unknown",
    "",
}


CANONICAL_DRUGS = {

    "labetalol": "Labetalol",
    "nifedipine": "Nifedipine",
    "methyldopa": "Methyldopa",
    "hydralazine": "Hydralazine",
    "magnesium sulfate": "Magnesium sulfate",
    "magnesium sulphate": "Magnesium sulfate", 
    "mgso4": "Magnesium sulfate",
}


BRANDS = {

    "dopegyt": "Methyldopa",

    "normodyne": "Labetalol",

}


REGEX_DRUGS = {

    "Magnesium sulfate": re.compile(r"mgso4|magnesi?um\s+sul[fp]hate", re.I),
    "Labetalol": re.compile(r"labetalol", re.I),
    "Nifedipine": re.compile(r"nifedipine", re.I),
    "Methyldopa": re.compile(r"methyldopa|alphamethydopa|dopegyt", re.I),
    "Hydralazine": re.compile(r"hydralazine", re.I),
    "Nitroglycerin": re.compile(r"nitroglycerine?", re.I), 

}


def standardize_drug(value):

    if pd.isna(value):

        return pd.NA


    raw = str(value).strip()


    norm = _norm_text(raw)


    if norm in NULLISH:

        return "Not specified"


    if norm in BRANDS:

        return BRANDS[norm]
    

    for canon, pattern in REGEX_DRUGS.items():

        if pattern.search(norm):

            return canon


    if norm in CANONICAL_DRUGS:

        return CANONICAL_DRUGS[norm]

    
    return _pretty(raw)


# ============================================================
# ROUTE STANDARDIZATION
# ============================================================

REGEX_ROUTE = {

    "IV": re.compile(

        r"\biv\b|intravenous|intravenously|I/V",

        re.I

    ),

    "IM": re.compile(

        r"\bim\b|intramuscular|I/M",

        re.I

    ),

    "Oral": re.compile(

        r"oral|orally|po",

        re.I

    ),

    "Sublingual": re.compile(

        r"sublingual|tongue",

        re.I

    ),

}


def standardize_route(value):

    if pd.isna(value):

        return pd.NA


    raw = str(value).strip()


    norm = _norm_text(raw)


    if norm in NULLISH:

        return "Not specified"


    routes = set()


    if REGEX_ROUTE["IV"].search(norm):

        routes.add("IV")


    if REGEX_ROUTE["IM"].search(norm):

        routes.add("IM")


    if REGEX_ROUTE["Oral"].search(norm):

        routes.add("Oral")


    if REGEX_ROUTE["Sublingual"].search(norm):

        routes.add("Oral")


    if not routes:

        return _pretty(raw)


    if len(routes) == 1:

        return list(routes)[0]


    return " + ".join(sorted(routes))


# ============================================================
# PUBLIC API
# ============================================================

def standardize_therapy_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    drug_cols = [

        c for c in df.columns

        if c.startswith("drug_")

    ]

    route_cols = [

        c for c in df.columns

        if c.startswith("route_")

    ]

    for col in drug_cols:

        df[col] = df[col].apply(standardize_drug)


    for col in route_cols:

        df[col] = df[col].apply(standardize_route)


    return df