"""
OUTCOME STANDARDIZER
Standardizes:
- outcome_value columns
- outcome_p_value columns
- winner column -> winner_1, winner_2, is_combination
"""

from __future__ import annotations
import pandas as pd
import re
import unicodedata
from typing import Any, List

# Importamos la función 
from src.obs_hypertension.standardization.standardizers.therapy_standardizer import (
    standardize_drug
)

# ============================================================
# GENERIC TEXT HELPERS
# ============================================================

def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

def _norm_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = _strip_accents(text)
    text = text.lower()
    # Normalizamos separadores comunes antes de procesar
    text = text.replace("&", " and ").replace("+", " and ")
    text = re.sub(r"[(){}\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Regex para dividir fármacos: busca "and", "plus", comas, puntos y coma o barras
_SPLIT_RX = re.compile(r"\band\b|\bplus\b|[,;/+]", re.I)

_NULLISH = {
    "not specified", "unspecified", "na", "n/a", "none", "unknown", "", "nan"
}

# ============================================================
# OUTCOME & P-VALUE LOGIC
# ============================================================

_RX_VS = re.compile(r"\bvs\b", re.I)
_PATTERNS = [
    # Captura: "no significant difference", "no statistical difference", "no statistically significant difference"
    ("No significant difference", 
     re.compile(r"no\s+(statistically\s+)?(significant|statistical)\s+difference", re.I)),
    
    # Captura: "significant difference" o "statistically significant difference"
    ("Significant difference", 
     re.compile(r"(statistically\s+)?significant\s+difference", re.I)),

    # Nueva categoría para seguridad/efectos adversos
    ("No severe adverse effects", 
     re.compile(r"no\s+severe\s+adverse\s+(effects|events)", re.I)),

    ("Higher", re.compile(r"higher", re.I)),
    ("Lower", re.compile(r"lower", re.I)),
]

def standardize_outcome_value(val):
    if pd.isna(val): return pd.NA
    if isinstance(val, (int, float)): return val
    norm = _norm_text(val)
    if norm in _NULLISH: return "Not specified"
    if _RX_VS.search(norm): return "Comparison reported"
    for label, pattern in _PATTERNS:
        if pattern.search(norm): return label
    return str(val).strip()

_RX_P = re.compile(r"[<>=]?\s*([0-9]*\.?[0-9]+)")

def standardize_outcome_p_value(val):
    if pd.isna(val): return pd.NA
    if isinstance(val, (int, float)): return float(val)
    norm = _norm_text(val)
    if norm in _NULLISH: return "Not specified"
    match = _RX_P.search(norm)
    if match:
        try: return float(match.group(1))
        except: return pd.NA
    return val

# ============================================================
# WINNER STANDARDIZATION 
# ============================================================

def standardize_winner_list(value) -> List[str]:
    """
    Divide la cadena del ganador y estandariza cada fármaco individualmente.
    """
    if pd.isna(value):
        return []

    # 1. Limpieza inicial
    raw_text = str(value).strip()
    if _norm_text(raw_text) in _NULLISH:
        return []

    # 2. Dividir por conectores (and, plus, +, comas)
    parts = _SPLIT_RX.split(raw_text)
    
    standardized_parts = []
    for p in parts:
        clean_p = p.strip()
        if clean_p:
            # 3. Aplicamos la lógica de therapy_standardizer a cada pedazo
            std_name = standardize_drug(clean_p)
            if std_name and std_name not in _NULLISH and std_name != "Not specified":
                standardized_parts.append(std_name)

    # 4. Eliminar duplicados manteniendo el orden
    return list(dict.fromkeys(standardized_parts))

def process_winner_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Aplicar la estandarización que devuelve una lista de fármacos
    winners_series = df["winner"].apply(standardize_winner_list)

    # Extraer primer y segundo fármaco
    df["winner_1"] = winners_series.apply(lambda x: x[0] if len(x) > 0 else pd.NA)
    df["winner_2"] = winners_series.apply(lambda x: x[1] if len(x) > 1 else pd.NA)

    # Definir is_combination: 
    # 0 = Un solo fármaco, 1 = Dos o más, 2 = Ninguno/No especificado
    df["is_combination"] = 2  # Default: No especificado
    
    # Caso: Solo uno
    mask_single = df["winner_1"].notna() & df["winner_2"].isna()
    df.loc[mask_single, "is_combination"] = 0
    
    # Caso: Combinación
    mask_combo = df["winner_1"].notna() & df["winner_2"].notna()
    df.loc[mask_combo, "is_combination"] = 1

    return df

# ============================================================
# MASTER FUNCTION
# ============================================================

def standardize_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Columnas de valor
    value_cols = [c for c in df.columns if "outcome_value" in c]
    for c in value_cols:
        df[c] = df[c].apply(standardize_outcome_value)

    # Columnas de p-value
    p_cols = [c for c in df.columns if "outcome_p_value" in c]
    for c in p_cols:
        df[c] = df[c].apply(standardize_outcome_p_value)

    # Tratamiento de Ganadores (Winner)
    if "winner" in df.columns:
        df = process_winner_columns(df)
      
    return df