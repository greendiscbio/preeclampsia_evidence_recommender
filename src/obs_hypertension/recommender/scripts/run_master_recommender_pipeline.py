"""
CLI runner for the full master recommender pipeline.

Usage example
-------------
python -m recommender_model.scripts.run_master_recommender_pipeline \
    --database-path ProcessedOutputs/arm_dataset_standardized.csv \
    --synthetic-path synthetic_population_generator/outputs/synthetic_population.csv \
    --output-dir recommender_model/outputs/run_001 \
    --structured-cols age_18_20 age_21_23 age_24_26 age_27_29 age_30_32 age_33_35 age_36_38 age_39_41 age_42_44 \
                     <20_weeks 20_33_weeks >33_weeks \
                     <119_SBP 120_129_SBP 130_139_SBP 140_160_SBP >161_SBP \
                     <85_DBP 86_90_DBP 91_105_DBP >106_DBP \
    --efficiency-feature-cols age_18_20 age_21_23 age_24_26 age_27_29 age_30_32 age_33_35 age_36_38 age_39_41 age_42_44 \
                              <20_weeks 20_33_weeks >33_weeks \
                              <119_SBP 120_129_SBP 130_139_SBP 140_160_SBP >161_SBP \
                              <85_DBP 86_90_DBP 91_105_DBP >106_DBP \
                              is_combination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.obs_hypertension.recommender.efficiency_scoring.config import EfficiencyScorerConfig
from src.obs_hypertension.recommender.final_orchestrator.config import FinalClinicalFilterConfig
from src.obs_hypertension.recommender.clinical_review.config import ClinicalReviewConfig
from src.obs_hypertension.recommender.pipeline.master_recommender_pipeline import (
    MasterRecommenderConfig,
    run_master_recommender_pipeline,
)

from src.obs_hypertension.recommender.config.defaults import (
    DEFAULT_COMORBIDITY_COL,
    DEFAULT_DATABASE_PATH,
    DEFAULT_DEDUPLICATE_BY,
    DEFAULT_DIAGNOSIS_COL,
    DEFAULT_EFFICIENCY_PARAMS,
    DEFAULT_FINAL_SCORE_COL,
    DEFAULT_FINAL_SCORE_WEIGHTS,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENT_ID_COL,
    DEFAULT_PRECLINICAL_TOP_K,
    DEFAULT_PROTOCOL_ID_COL,
    DEFAULT_RISK_MAX_OUTCOMES_PER_TYPE,
    DEFAULT_SCORE_ROUND,
    DEFAULT_SIMILARITY_MAX_ANOMALY_SLOTS,
    DEFAULT_SIMILARITY_WEIGHTS,
    DEFAULT_ST_BATCH_SIZE,
    DEFAULT_ST_MODEL_NAME,
    DEFAULT_SYNTHETIC_PATH,
    DEFAULT_TOP_N,
    EFFICIENCY_FEATURE_COLS,
    STRUCTURED_COLS,
)


# ============================================================
# IO helpers
# ============================================================

def load_csv_dataframe(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Input CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    return pd.read_csv(csv_path)


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save a dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    path : Path
        Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to JSON.

    Parameters
    ----------
    data : Dict[str, Any]
        Serializable data.
    path : Path
        Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# ============================================================
# Serialization helpers
# ============================================================

def build_artifacts_summary(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a lightweight JSON-serializable summary of pipeline outputs.

    Parameters
    ----------
    outputs : Dict[str, Any]
        Output dictionary from run_master_recommender_pipeline.

    Returns
    -------
    Dict[str, Any]
        Serializable summary.
    """
    artifacts = outputs.get("artifacts", {})

    summary = {
        "shapes": {
            "df_db_vec": list(outputs["df_db_vec"].shape),
            "df_syn_vec": list(outputs["df_syn_vec"].shape),
            "protocol_score_df": list(outputs["protocol_score_df"].shape),
            "recs_raw_df": list(outputs["recs_raw_df"].shape),
            "recs_final_df": list(outputs["recs_final_df"].shape),
            "patient_profiles_df": list(outputs["patient_profiles_df"].shape),
            "clinical_recs_df": list(outputs["clinical_recs_df"].shape),
        },
        "master_config": artifacts.get("master_config", {}),
        "final_filter_config": artifacts.get("final_filter_config", {}),
        "review_config": artifacts.get("review_config", {}),
        "efficiency_train_stats": (
            artifacts.get("efficiency_artifacts", {}).get("train_stats", {})
            if artifacts.get("efficiency_artifacts") is not None
            else {}
        ),
    }

    return summary


# ============================================================
# Argument parsing
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Run the full master recommender pipeline and save CSV outputs."
    )

    # Required/optional paths
    parser.add_argument(
        "--database-path",
        type=str,
        default=str(DEFAULT_DATABASE_PATH),
        help="Path to standardized protocol database CSV.",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        default=str(DEFAULT_SYNTHETIC_PATH),
        help="Path to synthetic population CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output CSVs and JSON summary will be saved.",
    )

    # Structured + efficiency features
    parser.add_argument(
        "--structured-cols",
        nargs="+",
        default=None,
        help="Structured columns used for vectorization. If omitted, defaults.py will be used.",
    )
    parser.add_argument(
        "--efficiency-feature-cols",
        nargs="+",
        default=None,
        help="Feature columns used by efficiency scorer. If omitted, defaults.py will be used.",
    )

    # General pipeline config
    parser.add_argument("--patient-id-col", type=str, default=DEFAULT_PATIENT_ID_COL)
    parser.add_argument("--protocol-id-col", type=str, default=DEFAULT_PROTOCOL_ID_COL)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--preclinical-top-k", type=int, default=DEFAULT_PRECLINICAL_TOP_K)

    # Final score weights
    parser.add_argument(
        "--similarity-weight",
        type=float,
        default=DEFAULT_FINAL_SCORE_WEIGHTS["similarity_weight"],
    )
    parser.add_argument(
        "--efficiency-weight",
        type=float,
        default=DEFAULT_FINAL_SCORE_WEIGHTS["efficiency_weight"],
    )
    parser.add_argument(
        "--safety-weight",
        type=float,
        default=DEFAULT_FINAL_SCORE_WEIGHTS["safety_weight"],
    )

    # Vectorizer options
    parser.add_argument(
        "--st-model-name",
        type=str,
        default=DEFAULT_ST_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--st-model-dir",
        type=str,
        default=None,
        help="Optional local directory for a pre-saved SentenceTransformer.",
    )
    parser.add_argument("--st-batch-size", type=int, default=DEFAULT_ST_BATCH_SIZE)
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=DEFAULT_NORMALIZE_EMBEDDINGS,
    )
    parser.add_argument("--save-vectorizer-model", action="store_true", default=False)

    # Similarity options
    parser.add_argument(
        "--similarity-w-struct",
        type=float,
        default=DEFAULT_SIMILARITY_WEIGHTS["w_struct"],
    )
    parser.add_argument(
        "--similarity-w-notes",
        type=float,
        default=DEFAULT_SIMILARITY_WEIGHTS["w_notes"],
    )
    parser.add_argument(
        "--similarity-max-anomaly-slots",
        type=int,
        default=DEFAULT_SIMILARITY_MAX_ANOMALY_SLOTS,
    )

    # Risk options
    parser.add_argument(
        "--risk-max-outcomes-per-type",
        type=int,
        default=DEFAULT_RISK_MAX_OUTCOMES_PER_TYPE,
    )
    parser.add_argument("--risk-st-batch-size", type=int, default=DEFAULT_ST_BATCH_SIZE)

    # Efficiency options
    parser.add_argument(
        "--eff-random-state",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["random_state"],
    )
    parser.add_argument(
        "--eff-min-train-rows",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["min_train_rows"],
    )
    parser.add_argument(
        "--eff-n-estimators",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["n_estimators"],
    )
    parser.add_argument(
        "--eff-max-depth",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["max_depth"],
    )
    parser.add_argument(
        "--eff-min-samples-leaf",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["min_samples_leaf"],
    )
    parser.add_argument(
        "--eff-n-jobs",
        type=int,
        default=DEFAULT_EFFICIENCY_PARAMS["n_jobs"],
    )
    parser.add_argument(
        "--eff-class-weight",
        type=str,
        default=DEFAULT_EFFICIENCY_PARAMS["class_weight"],
    )
    parser.add_argument(
        "--eff-default-score-if-insufficient",
        type=float,
        default=DEFAULT_EFFICIENCY_PARAMS["default_score_if_insufficient"],
    )

    # Final filter options
    parser.add_argument(
        "--deduplicate-by",
        type=str,
        default=DEFAULT_DEDUPLICATE_BY,
        choices=["treatment", "protocol_id", "none"],
        help="Deduplication strategy for final clinical filtering.",
    )
    parser.add_argument("--diagnosis-col", type=str, default=DEFAULT_DIAGNOSIS_COL)
    parser.add_argument("--comorbidity-col", type=str, default=DEFAULT_COMORBIDITY_COL)
    parser.add_argument("--final-score-col", type=str, default=DEFAULT_FINAL_SCORE_COL)

    # Review config
    parser.add_argument("--score-round", type=int, default=DEFAULT_SCORE_ROUND)

    # Misc
    parser.add_argument("--debug", action="store_true", default=False)

    return parser

# ============================================================
# Config builders
# ============================================================

def build_efficiency_config(args: argparse.Namespace) -> EfficiencyScorerConfig:
    """
    Build EfficiencyScorerConfig from CLI args.
    """
    max_depth = None if args.eff_max_depth in (-1, 0) else args.eff_max_depth
    feature_cols = list(args.efficiency_feature_cols) if args.efficiency_feature_cols else EFFICIENCY_FEATURE_COLS

    return EfficiencyScorerConfig(
        feature_cols=feature_cols,
        random_state=args.eff_random_state,
        min_train_rows=args.eff_min_train_rows,
        n_estimators=args.eff_n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.eff_min_samples_leaf,
        n_jobs=args.eff_n_jobs,
        class_weight=args.eff_class_weight,
        default_score_if_insufficient=args.eff_default_score_if_insufficient,
    )

def build_master_config(args: argparse.Namespace) -> MasterRecommenderConfig:
    """
    Build MasterRecommenderConfig from CLI args.
    """
    return MasterRecommenderConfig(
        patient_id_col=args.patient_id_col,
        protocol_id_col=args.protocol_id_col,
        top_n=args.top_n,
        preclinical_top_k=args.preclinical_top_k,
        similarity_weight=args.similarity_weight,
        efficiency_weight=args.efficiency_weight,
        safety_weight=args.safety_weight,
        debug=args.debug,
    )


def build_final_filter_config(args: argparse.Namespace) -> FinalClinicalFilterConfig:
    """
    Build FinalClinicalFilterConfig from CLI args.
    """
    deduplicate_by = None if args.deduplicate_by == "none" else args.deduplicate_by

    return FinalClinicalFilterConfig(
        top_n=args.top_n,
        patient_id_col=args.patient_id_col,
        diagnosis_col=args.diagnosis_col,
        comorbidity_col=args.comorbidity_col,
        final_score_col=args.final_score_col,
        protocol_id_col=args.protocol_id_col,
        deduplicate_by=deduplicate_by,
    )


def build_review_config(args: argparse.Namespace) -> ClinicalReviewConfig:
    """
    Build ClinicalReviewConfig from CLI args.
    """
    return ClinicalReviewConfig(
        patient_id_col=args.patient_id_col,
        score_round=args.score_round,
    )


def build_vectorizer_kwargs(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """
    Build kwargs for vectorization module.
    """
    vectorizer_output_dir = output_dir / "vectorizer_artifacts"

    return {
        "st_model_name": args.st_model_name,
        "st_model_dir": args.st_model_dir,
        "st_batch_size": args.st_batch_size,
        "normalize_embeddings": args.normalize_embeddings,
        "save_model": args.save_vectorizer_model,
        "output_dir": str(vectorizer_output_dir),
        "show_progress": args.debug,
    }


def build_similarity_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build kwargs for similarity scoring module.
    """
    return {
        "w_struct": args.similarity_w_struct,
        "w_notes": args.similarity_w_notes,
        "max_anomaly_slots": args.similarity_max_anomaly_slots,
        "st_batch_size": args.st_batch_size,
        "normalize_embeddings": args.normalize_embeddings,
        "show_progress": args.debug,
    }


def build_risk_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build kwargs for risk scoring module.
    """
    return {
        "max_outcomes_per_type": args.risk_max_outcomes_per_type,
        "st_batch_size": args.risk_st_batch_size,
        "show_progress": args.debug,
    }


def build_efficiency_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build kwargs for efficiency scoring module.
    """
    return {
        "debug": args.debug,
    }


# ============================================================
# Main runner
# ============================================================

def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute full master recommender pipeline from parsed CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    Dict[str, Any]
        Pipeline outputs.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print("🔹 Loading input CSV files...")

    df_database = load_csv_dataframe(args.database_path)
    synthetic_population = load_csv_dataframe(args.synthetic_path)

    efficiency_config = build_efficiency_config(args)
    master_config = build_master_config(args)
    final_filter_config = build_final_filter_config(args)
    review_config = build_review_config(args)

    vectorizer_kwargs = build_vectorizer_kwargs(args, output_dir)
    similarity_kwargs = build_similarity_kwargs(args)
    risk_kwargs = build_risk_kwargs(args)
    efficiency_kwargs = build_efficiency_kwargs(args)

    if args.debug:
        print("🔹 Running full master recommender pipeline...")

    outputs = run_master_recommender_pipeline(
        synthetic_population=synthetic_population,
        df_database=df_database,
        structured_cols=list(args.structured_cols) if args.structured_cols else STRUCTURED_COLS,
        efficiency_config=efficiency_config,
        config=master_config,
        final_filter_config=final_filter_config,
        review_config=review_config,
        vectorizer_kwargs=vectorizer_kwargs,
        similarity_kwargs=similarity_kwargs,
        risk_kwargs=risk_kwargs,
        efficiency_kwargs=efficiency_kwargs,
    )

    if args.debug:
        print("🔹 Saving output CSV files...")

    save_dataframe_csv(outputs["df_db_vec"], output_dir / "df_db_vec.csv")
    save_dataframe_csv(outputs["df_syn_vec"], output_dir / "df_syn_vec.csv")
    save_dataframe_csv(outputs["protocol_score_df"], output_dir / "protocol_score_df.csv")
    save_dataframe_csv(outputs["recs_raw_df"], output_dir / "recs_raw_df.csv")
    save_dataframe_csv(outputs["recs_final_df"], output_dir / "recs_final_df.csv")
    save_dataframe_csv(outputs["patient_profiles_df"], output_dir / "patient_profiles_df.csv")
    save_dataframe_csv(outputs["clinical_recs_df"], output_dir / "clinical_recs_df.csv")

    artifacts_summary = build_artifacts_summary(outputs)
    save_json(artifacts_summary, output_dir / "artifacts_summary.json")

    if args.debug:
        print("Master recommender pipeline finished successfully.")
        print(f"Outputs saved in: {output_dir}")

    return outputs


def main() -> None:
    """
    CLI entrypoint.
    """
    parser = build_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()