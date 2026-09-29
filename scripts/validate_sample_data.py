"""Validate the reproducibility sample CSVs against the current code contract."""

from __future__ import annotations

import json
from pathlib import Path

from src.obs_hypertension.recommender.efficiency_scoring.target_builder import (
    define_efficiency_target,
)

import pandas as pd




ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
SCHEMA_PATH = SAMPLE_DIR / "schema.json"
PATIENT_PATH = SAMPLE_DIR / "synthetic_profiles_sample.csv"
EVIDENCE_PATH = SAMPLE_DIR / "evidence_repository_sample.csv"


def _fail(message: str) -> None:
    raise ValueError(message)


def _require_columns(df: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        _fail(f"{dataset}: missing required columns: {missing}")


def _require_unique_nonempty(df: pd.DataFrame, column: str, dataset: str) -> None:
    values = df[column].fillna("").astype(str).str.strip()
    if (values == "").any():
        _fail(f"{dataset}: '{column}' contains empty values")
    if values.duplicated().any():
        _fail(f"{dataset}: '{column}' must be unique")


def _validate_binary_columns(df: pd.DataFrame, columns: list[str], dataset: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            _fail(f"{dataset}: '{column}' must contain only numeric 0/1 values")


def _validate_one_hot_groups(df: pd.DataFrame, groups: dict[str, list[str]], dataset: str) -> None:
    for group_name, columns in groups.items():
        _require_columns(df, columns, dataset)
        _validate_binary_columns(df, columns, dataset)
        row_sums = df[columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        if not (row_sums == 1).all():
            bad_rows = df.index[row_sums != 1].tolist()
            _fail(f"{dataset}: one-hot group '{group_name}' must have exactly one active value; bad rows={bad_rows}")


def _validate_category_ohe(
    df: pd.DataFrame,
    source_column: str,
    prefix: str,
    categories: list[str],
    dataset: str,
) -> None:
    ohe_columns = [f"{prefix}{category}" for category in categories]
    _require_columns(df, ohe_columns, dataset)
    _validate_binary_columns(df, ohe_columns, dataset)
    if not (df[ohe_columns].sum(axis=1) == 1).all():
        _fail(f"{dataset}: OHE family '{prefix}' must have exactly one active value per row")
    for idx, source_value in df[source_column].fillna("").astype(str).items():
        expected = f"{prefix}{source_value}"
        if expected not in ohe_columns or int(df.at[idx, expected]) != 1:
            _fail(f"{dataset}: row {idx} OHE does not match {source_column}={source_value!r}")


def validate_patients(df: pd.DataFrame, schema: dict) -> None:
    dataset = "synthetic_profiles_sample.csv"
    _require_columns(
    df,
    _structured_cols_from_schema(schema),
    dataset,
	)
    _require_columns(df, schema["patient_required_text_columns"], dataset)
    _require_columns(df, schema["patient_required_numeric_columns"], dataset)
    _require_unique_nonempty(df, "patient_id", dataset)
    _validate_one_hot_groups(df, schema["structured_groups"], dataset)

    if not df["Maternal_Diagnosis"].isin(schema["diagnoses"]).all():
        _fail(f"{dataset}: unsupported Maternal_Diagnosis value")
    if not df["Comorbility"].isin(schema["comorbidities"]).all():
        _fail(f"{dataset}: unsupported Comorbility value")
    _validate_binary_columns(df, ["Proteinuria"], dataset)

    if len(df) != schema["sample_expectations"]["patient_rows"]:
        _fail(f"{dataset}: expected {schema['sample_expectations']['patient_rows']} rows, found {len(df)}")


def _validate_outcomes(df: pd.DataFrame, schema: dict) -> None:
    risk_types = set(schema["risk_outcome_types"])
    profile_type = schema["profile_outcome_type"]
    found_risk = False
    found_profile = False

    for family in ("maternal", "fetal", "anomaly"):
        max_slots = schema["similarity_anomaly_slots"] if family == "anomaly" else schema["risk_slots"][family]
        for slot in range(1, max_slots + 1):
            cols = [f"{family}_outcome_{field}_{slot}" for field in ("name", "type", "value", "p_value")]
            _require_columns(df, cols, "evidence_repository_sample.csv")
            for _, row in df[cols].iterrows():
                name, outcome_type, value, p_value = ["" if pd.isna(v) else str(v).strip() for v in row.tolist()]
                if not any((name, outcome_type, value, p_value)):
                    continue
                if not all((name, outcome_type, value, p_value)):
                    _fail(f"evidence_repository_sample.csv: partial outcome payload in {family} slot {slot}")
                normalized = outcome_type.lower()
                if normalized in risk_types and slot <= schema["risk_slots"][family]:
                    found_risk = True
                elif family == "anomaly" and normalized == profile_type:
                    found_profile = True
                else:
                    _fail(f"evidence_repository_sample.csv: unsupported outcome type/slot: {family} slot {slot} type={outcome_type!r}")

    if schema["sample_expectations"]["require_risk_outcome"] and not found_risk:
        _fail("evidence_repository_sample.csv: sample must include at least one risk-scoring outcome")
    if schema["sample_expectations"]["require_profile_anomaly"] and not found_profile:
        _fail("evidence_repository_sample.csv: sample must include at least one profile anomaly")


def validate_evidence(df: pd.DataFrame, schema: dict) -> None:
    dataset = "evidence_repository_sample.csv"
    _require_columns(
    df,
    _structured_cols_from_schema(schema),
    dataset,
)
    _require_columns(df, schema["evidence_required_text_columns"], dataset)
    _require_columns(df, schema["evidence_required_numeric_columns"], dataset)
    _require_columns(
    df,
    _efficiency_feature_cols_combo_from_schema(schema),
    dataset,
)
    _require_unique_nonempty(df, "corpusid", dataset)
    _validate_one_hot_groups(df, schema["structured_groups"], dataset)
    _validate_binary_columns(df, ["is_combination"], dataset)

    if not df["maternal_clinical_diagnosis__stdcat"].isin(schema["evidence_diagnoses"]).all():
        _fail(f"{dataset}: unsupported standardized diagnosis")
    if not df["drug_1__stdcat"].isin(schema["drugs"]).all() or not df["drug_2__stdcat"].isin(schema["drugs"]).all():
        _fail(f"{dataset}: unsupported standardized drug")

    _validate_category_ohe(df, "maternal_clinical_diagnosis__stdcat", "maternal_clinical_diagnosis__stdcat__", schema["evidence_diagnoses"], dataset)
    _validate_category_ohe(df, "drug_1__stdcat", "drug_1__stdcat__", schema["drugs"], dataset)
    _validate_category_ohe(df, "drug_2__stdcat", "drug_2__stdcat__", schema["drugs"], dataset)
    _validate_category_ohe(df, "route_1", "route_1__stdcat__", schema["routes"], dataset)

    route2_for_ohe = df["route_2"].fillna("").astype(str).replace("", "Not specified")
    df_for_route2 = df.copy()
    df_for_route2["route_2_for_validation"] = route2_for_ohe
    _validate_category_ohe(df_for_route2, "route_2_for_validation", "route_2__stdcat__", schema["routes"], dataset)

    if len(df) < schema["sample_expectations"]["minimum_evidence_rows"]:
        _fail(f"{dataset}: expected at least {schema['sample_expectations']['minimum_evidence_rows']} rows")

    targeted, ambiguous_mask = define_efficiency_target(df.copy())
    non_ambiguous = targeted.loc[~ambiguous_mask, "Is_Efficient"].astype(int)
    if len(non_ambiguous) < schema["minimum_evidence_rows_for_efficiency"]:
        _fail(f"{dataset}: fewer than {schema['minimum_evidence_rows_for_efficiency']} non-ambiguous efficiency rows")
    if schema["sample_expectations"]["require_both_efficiency_classes"] and set(non_ambiguous.unique()) != {0, 1}:
        _fail(f"{dataset}: efficiency target must contain both classes 0 and 1")

    _validate_outcomes(df, schema)

def _structured_cols_from_schema(schema: dict) -> list[str]:
    """Reconstruct the structured feature columns from the sample contract."""
    return [
        column
        for columns in schema["structured_groups"].values()
        for column in columns
    ]


def _efficiency_feature_cols_combo_from_schema(schema: dict) -> list[str]:
    """Reconstruct the combination-therapy efficiency feature contract."""
    structured_cols = _structured_cols_from_schema(schema)

    diagnosis_cols = [
        f"maternal_clinical_diagnosis__stdcat__{value}"
        for value in schema["evidence_diagnoses"]
    ]

    drug_1_cols = [
        f"drug_1__stdcat__{value}"
        for value in schema["drugs"]
    ]

    route_1_cols = [
        f"route_1__stdcat__{value}"
        for value in schema["routes"]
    ]

    drug_2_cols = [
        f"drug_2__stdcat__{value}"
        for value in schema["drugs"]
    ]

    route_2_cols = [
        f"route_2__stdcat__{value}"
        for value in schema["routes"]
    ]

    return (
        structured_cols
        + diagnosis_cols
        + drug_1_cols
        + route_1_cols
        + drug_2_cols
        + route_2_cols
    
)


def main() -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    patients = pd.read_csv(PATIENT_PATH, keep_default_na=False)
    evidence = pd.read_csv(EVIDENCE_PATH, keep_default_na=False)
    

    validate_patients(patients, schema)
    validate_evidence(evidence, schema)

    print("Sample data validation passed.")
    print(f"  patients: {len(patients)} rows x {len(patients.columns)} columns")
    print(f"  evidence: {len(evidence)} rows x {len(evidence.columns)} columns")


if __name__ == "__main__":
    main()
