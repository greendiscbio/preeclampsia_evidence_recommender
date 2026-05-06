"""
cohort_standardizer.py

Standardizes cohort-related columns:

- maternal_clinical_diagnosis
- maternal_age_range
- gestational_age_range
- systolic_pressure_range
- diastolic_pressure_range

Adds standardized ranges and one-hot encodings.

This module DOES NOT read or write files.
Used by master_standardization_pipeline.py
"""

import pandas as pd
import re
import unicodedata


# ============================================================
# GENERIC HELPERS
# ============================================================

NULLISH = {

    "not specified",
    "unspecified",
    "n/a",
    "na",
    "unknown",
    "none",
    "",
}


def strip_accents(text: str) -> str:

    return "".join(

        c for c in unicodedata.normalize("NFKD", text)

        if not unicodedata.combining(c)

    )


def fix_mojibake(text: str) -> str:

    """
    Limpia y normaliza texto científico eliminando mojibake, normalizando Unicode 
    y estandarizando símbolos matemáticos/estadísticos.
    """
    if text is None:
        return ""

    # 1. Normalización inicial: Descompone caracteres combinados (como tildes)
    # y elimina variaciones de formato extrañas.
    text = unicodedata.normalize('NFKC', str(text))

    # 2. Diccionario extendido: Mojibake (errores de encoding) + Unicode técnico
    # Se prioriza convertir símbolos complejos a texto estándar (ASCII-like)
    replacements = {
        # --- Errores de encoding comunes (Mojibake UTF-8 -> Latin1) ---
        "Â±": "±",      # Plus-minus
        "â‰¥": ">=",    # Greater-than or equal
        "â‰¤": "<=",    # Less-than or equal
        "â€“": "-",     # En dash
        "â€”": "-",     # Em dash
        "Â°": "°",      # Degree symbol
        "Ã—": "x",      # Multiplication sign
        "Âµ": "u",      # Micro symbol (often used for micromolar/liter)
        "Â½": "1/2",
        "Â¼": "1/4",
        "Â¾": "3/4",

        # --- Símbolos Matemáticos y Estadísticos ---
        "\u2264": "<=", # ≤
        "\u2265": ">=", # ≥
        "\u2260": "!=", # ≠
        "\u00b1": "+/-",# ± (estandarizado para minería de texto)
        "\u2212": "-",  # Minus sign real
        "\u00d7": "x",  # Multiplication
        "\u00f7": "/",  # Division
        "\u2217": "*",  # Asterisk operator
        "\u2248": "~",  # Approximately equal
        "\u221e": "inf",# Infinity
        
        # --- Guiones y Espacios ---
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2212": "-",  # Minus sign
        "\u00a0": " ",  # Non-breaking space
        "\u200b": "",   # Zero-width space (muy común en PDFs)

        # --- Griegos comunes (usados en p-values o dosis) ---
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03bc": "u",     # micro (µ)
        "\u03c0": "pi",
        
        # --- Otros ---
        "\u2122": "(TM)",
        "\u00ae": "(R)",
        "\u00a9": "(C)",
        "\u00b0": " degrees", # o simplemente "°"
    }

    # Aplicar reemplazos del diccionario
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 3. Limpieza de caracteres no imprimibles restantes (control characters)
    # Esto elimina ruidos de saltos de página o caracteres de control de archivos binarios.
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ["\n", "\t"])

    # 4. Colapsar espacios múltiples (típico de minería de PDFs)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# MATERNAL DIAGNOSIS
# ============================================================

def normalize_diagnosis(text):

    text = str(text).strip()

    text = strip_accents(text).lower()

    text = text.replace("pre eclampsia", "preeclampsia")

    text = re.sub(r"\s+", " ", text)

    return text


def standardize_maternal_clinical_diagnosis(val):

    if pd.isna(val):

        return pd.NA


    text = normalize_diagnosis(val)


    if text in NULLISH:

        return "Not specified"


    if "hellp" in text:

        return "HELLP syndrome"


    if "eclampsia" in text and "preeclampsia" in text:

        return "Preeclampsia + Eclampsia"


    if "eclampsia" in text:

        return "Eclampsia"


    if "preeclampsia" in text:

        return "Preeclampsia"


    if "chronic hypertension" in text:

        return "Chronic hypertension"


    if "gestational hypertension" in text:

        return "Gestational hypertension"


    if "hypertension" in text:

        if "severe" in text:

            return "Gestational hypertension (severe)"

        return "Gestational hypertension"


    if "normotensive" in text:

        return "Normotensive"


    return val


# ============================================================
# AGE RANGE
# ============================================================

def parse_range(text, min_allowed, max_allowed):

    text = fix_mojibake(text)

    text = strip_accents(str(text)).lower()


    if text in NULLISH:

        return (None, None)


    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    
    # DETECCIÓN DE MEDIA ± SD
    # Si hay un +/- o ± en el texto y tenemos exactamente 2 números
    if ("+/-" in text or "±" in text) and len(numbers) == 2:
        media = numbers[0]
        sd = numbers[1]
        low = round(media - sd)
        high = round(media + sd)
        
        return (
            max(min_allowed, low),
            min(max_allowed, high)
        )

    # LÓGICA ORIGINAL (Para rangos tipo "10-20" o "10 a 20")
    if len(numbers) >= 2:
        low = round(numbers[0])
        high = round(numbers[1])
        return (
            max(min_allowed, low),
            min(max_allowed, high),
        )

    if len(numbers) == 1:
        n = round(numbers[0])
        return (n, n)

    return (None, None)


def standardize_age_range(val):

    low, high = parse_range(val, 10, 60)


    if low is None:

        return "Not specified"


    return f"{low}-{high}"


# ============================================================
# GESTATIONAL AGE
# ============================================================

def standardize_gestational_age_range(val):

    low, high = parse_range(val, 0, 45)


    if low is None:

        return "Not specified"


    return f"{low}-{high}"


# ============================================================
# SYSTOLIC BP
# ============================================================

def standardize_sbp_range(val):

    low, high = parse_range(val, 50, 300)


    if low is None:

        return "Not specified"


    return f"{low}-{high}"


# ============================================================
# DIASTOLIC BP
# ============================================================

def standardize_dbp_range(val):

    low, high = parse_range(val, 30, 200)


    if low is None:

        return "Not specified"


    return f"{low}-{high}"


# ============================================================
# ONE HOT ENCODING HELPERS
# ============================================================

def one_hot_overlap(df, col, bins):

    parsed = df[col].str.extract(r"(\d+)-(\d+)")

    parsed.columns = ["low", "high"]

    parsed = parsed.astype(float)


    result = {}


    for name, (b0, b1) in bins.items():

        result[name] = (

            (parsed["high"] >= b0) &

            (parsed["low"] <= b1)

        ).astype(int)


    return pd.DataFrame(result)


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def standardize_cohort_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    # Diagnosis

    if "maternal_clinical_diagnosis" in df.columns:

        df["maternal_clinical_diagnosis"] = (

            df["maternal_clinical_diagnosis"]

            .apply(standardize_maternal_clinical_diagnosis)

        )


    # Age

    if "maternal_age_range" in df.columns:

        df["maternal_age_range"] = (

            df["maternal_age_range"]

            .apply(standardize_age_range)

        )


        age_bins = {

            "age_18_20": (18, 20),

            "age_21_23": (21, 23),

            "age_24_26": (24, 26),

            "age_27_29": (27, 29),

            "age_30_32": (30, 32),

            "age_33_35": (33, 35),

            "age_36_38": (36, 38),

            "age_39_41": (39, 41),

            "age_42_44": (42, 44),

        }


        df = pd.concat(

            [df, one_hot_overlap(df, "maternal_age_range", age_bins)],

            axis=1,

        )


    # Gestational age

    if "gestational_age_range" in df.columns:

        df["gestational_age_range"] = (

            df["gestational_age_range"]

            .apply(standardize_gestational_age_range)

        )


        gw_bins = {

            "<20_weeks": (0, 19),

            "20_33_weeks": (20, 33),

            ">33_weeks": (34, 60),

        }


        df = pd.concat(

            [df, one_hot_overlap(df, "gestational_age_range", gw_bins)],

            axis=1,

        )


    # SBP

    # SBP (Dentro del bloque de seguridad)
    if "systolic_pressure_range" in df.columns:
        df["systolic_pressure_range"] = (
            df["systolic_pressure_range"]
            .apply(standardize_sbp_range)
        )

        sbp_bins = {
            "<119_SBP": (0, 119),
            "120_129_SBP": (120, 129),
            "130_139_SBP": (130, 139),
            "140_160_SBP": (140, 160),
            ">161_SBP": (161, 300) 
        } 

        # Solo concatenamos si la columna existe para evitar errores de alineación
        sbp_encoded = one_hot_overlap(df, "systolic_pressure_range", sbp_bins)
        df = pd.concat([df, sbp_encoded], axis=1)


    # DBP

    if "diastolic_pressure_range" in df.columns:

        df["diastolic_pressure_range"] = (

            df["diastolic_pressure_range"]

            .apply(standardize_dbp_range)

        )

        dbp_bins = {

                "<85_DBP": (0, 85),
                "86_90_DBP": (86, 90),
                "91_105_DBP": (91, 105),
                ">106_DBP": (106, 300),
                

            }

        df = pd.concat([df, one_hot_overlap(df, "diastolic_pressure_range", dbp_bins)], axis=1) 


    return df