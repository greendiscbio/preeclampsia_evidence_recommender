"""
MASTER RECOMMENDER PIPELINE

End-to-end workflow
-------------------
1. Vectorize database + synthetic population
2. Compute similarity scores
3. Compute protocol-level risk scores
4. Compute protocol-level efficiency scores
5. Combine scores into final_score
6. Build ranked recommendation table (recs_raw_df)
7. Apply final clinical filters (recs_final_df)
8. Build clinical review tables:
   - patient_profiles_df
   - clinical_recs_df

Main outputs
------------
- similarity_score
- risk_score
- efficiency_score
- final_score
- rank
- recs_raw_df
- recs_final_df
- patient_profiles_df
- clinical_recs_df
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.obs_hypertension.recommender.vectorization.vectorizer import vectorize_recommender_inputs
from src.obs_hypertension.recommender.similarity_scoring.similarity_builder import (
    compute_similarity_matrices_for_population,
)
from src.obs_hypertension.recommender.risk_scoring.risk_builder import compute_protocol_risk_scores
from src.obs_hypertension.recommender.efficiency_scoring.config import EfficiencyScorerConfig
from src.obs_hypertension.recommender.efficiency_scoring.efficiency_builder import (
    compute_protocol_efficiency_scores,
)
from src.obs_hypertension.recommender.final_orchestrator.config import FinalClinicalFilterConfig
from src.obs_hypertension.recommender.final_orchestrator.orchestrator import apply_final_clinical_filters
from src.obs_hypertension.recommender.clinical_review.config import ClinicalReviewConfig
from src.obs_hypertension.recommender.clinical_review.review_builders import (
    build_clinical_recommendations_df,
    build_patient_review_df,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_PROTOCOL_METADATA_COLS: Tuple[str, ...] = (
    "corpusid",
    "drug_1",
    "route_1",
    "dose_1",
    "drug_2",
    "route_2",
    "dose_2",
    "winner_1__stdcat",
    "winner_2__stdcat",
    "statistical_significance_vs_control",
    "is_combination",
)


@dataclass
class MasterRecommenderConfig:
    """
    Configuration for the full recommender pipeline.

    Parameters
    ----------
    patient_id_col : str
        Patient identifier column.
    protocol_id_col : str
        Protocol identifier column.
    top_n : int
        Final number of recommendations per patient after clinical filtering.
    preclinical_top_k : int
        Number of ranked candidates kept per patient before clinical filtering.
    similarity_weight : float
        Weight for similarity_score in final score.
    efficiency_weight : float
        Weight for efficiency_score in final score.
    safety_weight : float
        Weight for safety_score in final score, where safety_score = 1 - normalized_risk.
    protocol_metadata_cols : Tuple[str, ...]
        Database columns copied into recommendation outputs.
    copy_inputs : bool
        Whether to work on dataframe copies.
    debug : bool
        Whether to print progress messages.
    """
    patient_id_col: str = "patient_id"
    protocol_id_col: str = "corpusid"

    top_n: int = 3
    preclinical_top_k: int = 20

    similarity_weight: float = 0.50
    efficiency_weight: float = 0.30
    safety_weight: float = 0.20

    protocol_metadata_cols: Tuple[str, ...] = DEFAULT_PROTOCOL_METADATA_COLS
    copy_inputs: bool = True
    debug: bool = True


# ============================================================
# Validation helpers
# ============================================================

def _validate_master_pipeline_inputs(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    structured_cols: Sequence[str],
    efficiency_config: EfficiencyScorerConfig,
    config: MasterRecommenderConfig,
) -> None:
    """
    Validate master pipeline inputs.
    """
    if synthetic_population is None or synthetic_population.empty:
        raise ValueError("synthetic_population is empty.")

    if df_database is None or df_database.empty:
        raise ValueError("df_database is empty.")

    if not structured_cols:
        raise ValueError("structured_cols cannot be empty.")

    if not isinstance(efficiency_config, EfficiencyScorerConfig):
        raise TypeError("efficiency_config must be an instance of EfficiencyScorerConfig.")

    if config.top_n <= 0:
        raise ValueError("config.top_n must be greater than 0.")

    if config.preclinical_top_k <= 0:
        raise ValueError("config.preclinical_top_k must be greater than 0.")

    weights = [
        config.similarity_weight,
        config.efficiency_weight,
        config.safety_weight,
    ]
    if any(w < 0 for w in weights):
        raise ValueError("All final score weights must be non-negative.")

    if sum(weights) <= 0:
        raise ValueError("At least one final score weight must be positive.")


# ============================================================
# Numeric helpers
# ============================================================

def _minmax_scale_1d(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max scale a 1D numpy array to [0, 1].
    Constant arrays become zeros.
    """
    values = np.asarray(values, dtype="float32")
    if values.size == 0:
        return values

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if abs(max_val - min_val) < eps:
        return np.zeros_like(values, dtype="float32")

    return ((values - min_val) / (max_val - min_val + eps)).astype("float32")


def _recompute_rank(
    df: pd.DataFrame,
    patient_id_col: str,
    score_col: str = "final_score",
) -> pd.DataFrame:
    """
    Recompute dense rank per patient based on descending score.
    """
    if df.empty:
        return df.copy()

    df_out = df.copy()
    df_out = df_out.sort_values([patient_id_col, score_col], ascending=[True, False]).copy()
    df_out["rank"] = (
        df_out.groupby(patient_id_col)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df_out


# ============================================================
# Protocol-level score builders
# ============================================================

def _build_protocol_score_frame(
    df_database: pd.DataFrame,
    similarity_payload: Dict[str, Any],
    risk_scores: np.ndarray,
    efficiency_scores: np.ndarray,
    config: MasterRecommenderConfig,
) -> pd.DataFrame:
    """
    Build protocol-level dataframe with metadata and protocol-level scores.
    """
    n_protocols = len(df_database)

    if len(risk_scores) != n_protocols:
        raise ValueError("risk_scores length does not match df_database length.")

    if len(efficiency_scores) != n_protocols:
        raise ValueError("efficiency_scores length does not match df_database length.")

    protocol_df = pd.DataFrame({
        "protocol_row_idx": np.arange(n_protocols, dtype=int),
        "risk_score": np.asarray(risk_scores, dtype="float32"),
        "efficiency_score": np.asarray(efficiency_scores, dtype="float32"),
    })

    if similarity_payload.get("protocol_ids") is not None:
        protocol_df["protocol_id"] = similarity_payload["protocol_ids"]
    elif config.protocol_id_col in df_database.columns:
        protocol_df["protocol_id"] = df_database[config.protocol_id_col].tolist()
    else:
        protocol_df["protocol_id"] = protocol_df["protocol_row_idx"]

    meta_cols = [col for col in config.protocol_metadata_cols if col in df_database.columns]
    if meta_cols:
        protocol_df = pd.concat(
            [protocol_df, df_database[meta_cols].reset_index(drop=True)],
            axis=1,
        )

    protocol_df["risk_score_normalized"] = _minmax_scale_1d(protocol_df["risk_score"].to_numpy())
    protocol_df["safety_score"] = (1.0 - protocol_df["risk_score_normalized"]).astype("float32")

    return protocol_df


def _compute_final_score_matrix(
    similarity_matrix: np.ndarray,
    efficiency_scores: np.ndarray,
    risk_scores_normalized: np.ndarray,
    config: MasterRecommenderConfig,
) -> np.ndarray:
    """
    Compute final score matrix.

    final_score =
        similarity_weight * similarity_score
        + efficiency_weight * efficiency_score
        + safety_weight * (1 - normalized_risk)
    """
    efficiency_matrix = np.asarray(efficiency_scores, dtype="float32")[None, :]
    risk_norm_matrix = np.asarray(risk_scores_normalized, dtype="float32")[None, :]
    safety_matrix = 1.0 - risk_norm_matrix

    final_matrix = (
        config.similarity_weight * similarity_matrix
        + config.efficiency_weight * efficiency_matrix
        + config.safety_weight * safety_matrix
    )

    return final_matrix.astype("float32")


# ============================================================
# Recommendation table builders
# ============================================================

def _build_ranked_recommendations_df(
    similarity_payload: Dict[str, Any],
    protocol_score_df: pd.DataFrame,
    final_score_matrix: np.ndarray,
    config: MasterRecommenderConfig,
) -> pd.DataFrame:
    """
    Build patient-protocol ranked recommendation dataframe before clinical filtering.
    Keeps top preclinical_top_k candidates per patient.
    """
    similarity_matrix = np.asarray(similarity_payload["similarity"], dtype="float32")
    sim_struct_matrix = np.asarray(similarity_payload["sim_struct"], dtype="float32")
    sim_notes_matrix = np.asarray(similarity_payload["sim_notes"], dtype="float32")
    patient_ids = similarity_payload["patient_ids"]

    n_patients, n_protocols = similarity_matrix.shape
    if final_score_matrix.shape != (n_patients, n_protocols):
        raise ValueError("final_score_matrix shape is incompatible with similarity matrices.")

    top_k = min(config.preclinical_top_k, n_protocols)
    output_chunks: List[pd.DataFrame] = []

    protocol_base = protocol_score_df.copy()

    for patient_idx, patient_id in enumerate(patient_ids):
        patient_final_scores = final_score_matrix[patient_idx]
        top_protocol_indices = np.argsort(-patient_final_scores)[:top_k]

        patient_chunk = protocol_base.iloc[top_protocol_indices].copy().reset_index(drop=True)
        patient_chunk[config.patient_id_col] = patient_id
        patient_chunk["similarity_score"] = similarity_matrix[patient_idx, top_protocol_indices]
        patient_chunk["similarity_structured_score"] = sim_struct_matrix[patient_idx, top_protocol_indices]
        patient_chunk["similarity_notes_score"] = sim_notes_matrix[patient_idx, top_protocol_indices]
        patient_chunk["final_score"] = final_score_matrix[patient_idx, top_protocol_indices]
        patient_chunk["rank"] = np.arange(1, len(patient_chunk) + 1, dtype=int)

        output_chunks.append(patient_chunk)

    if not output_chunks:
        return pd.DataFrame()

    recs_raw_df = pd.concat(output_chunks, axis=0, ignore_index=True)

    numeric_cols = [
        "similarity_score",
        "similarity_structured_score",
        "similarity_notes_score",
        "risk_score",
        "risk_score_normalized",
        "safety_score",
        "efficiency_score",
        "final_score",
    ]
    for col in numeric_cols:
        if col in recs_raw_df.columns:
            recs_raw_df[col] = pd.to_numeric(recs_raw_df[col], errors="coerce")

    recs_raw_df = recs_raw_df.sort_values(
        [config.patient_id_col, "rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return recs_raw_df


# ============================================================
# Main pipeline
# ============================================================

def run_master_recommender_pipeline(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    structured_cols: List[str],
    efficiency_config: EfficiencyScorerConfig,
    config: MasterRecommenderConfig = MasterRecommenderConfig(),
    final_filter_config: Optional[FinalClinicalFilterConfig] = None,
    review_config: ClinicalReviewConfig = ClinicalReviewConfig(),
    vectorizer_kwargs: Optional[Dict[str, Any]] = None,
    similarity_kwargs: Optional[Dict[str, Any]] = None,
    risk_kwargs: Optional[Dict[str, Any]] = None,
    efficiency_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full recommender pipeline end-to-end.

    Parameters
    ----------
    synthetic_population : pd.DataFrame
        Synthetic patient population.
    df_database : pd.DataFrame
        Protocol database.
    structured_cols : List[str]
        Structured feature columns used by vectorization.
    efficiency_config : EfficiencyScorerConfig
        Efficiency model configuration.
    config : MasterRecommenderConfig
        Main pipeline configuration.
    final_filter_config : Optional[FinalClinicalFilterConfig]
        Optional configuration for final clinical filtering.
    review_config : ClinicalReviewConfig
        Configuration for review tables.
    vectorizer_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to vectorizer.
    similarity_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to similarity scoring.
    risk_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to risk scoring.
    efficiency_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to efficiency scoring.

    Returns
    -------
    Dict[str, Any]
        Dictionary with all major outputs and artifacts:
        - df_db_vec
        - df_syn_vec
        - st_model
        - similarity_payload
        - risk_scores
        - efficiency_scores
        - protocol_score_df
        - recs_raw_df
        - recs_final_df
        - patient_profiles_df
        - clinical_recs_df
        - artifacts
    """
    _validate_master_pipeline_inputs(
        synthetic_population=synthetic_population,
        df_database=df_database,
        structured_cols=structured_cols,
        efficiency_config=efficiency_config,
        config=config,
    )

    syn = synthetic_population.copy() if config.copy_inputs else synthetic_population
    db = df_database.copy() if config.copy_inputs else df_database

    vectorizer_kwargs = vectorizer_kwargs or {}
    similarity_kwargs = similarity_kwargs or {}
    risk_kwargs = risk_kwargs or {}
    efficiency_kwargs = efficiency_kwargs or {}

    if config.debug:
        print("🔹 [MasterPipeline] Step 1/8 - Vectorizing inputs...")

    df_db_vec, df_syn_vec, st_model = vectorize_recommender_inputs(
        df_database=db,
        synthetic_population=syn,
        structured_cols=structured_cols,
        **vectorizer_kwargs,
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 2/8 - Computing similarity matrices...")

    similarity_payload = compute_similarity_matrices_for_population(
        synthetic_population=df_syn_vec,
        df_database=df_db_vec,
        st_model=st_model,
        patient_id_col=config.patient_id_col,
        db_id_col=config.protocol_id_col,
        **similarity_kwargs,
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 3/8 - Computing protocol risk scores...")

    risk_scores = compute_protocol_risk_scores(
        df_database=db,
        df_bbdd_vec=df_db_vec,
        st_model=st_model,
        **risk_kwargs,
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 4/8 - Computing protocol efficiency scores...")

    efficiency_scores, df_efficiency, efficiency_artifacts = compute_protocol_efficiency_scores(
        df_database=db,
        config=efficiency_config,
        return_dataframe=True,
        debug=config.debug,
        **efficiency_kwargs,
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 5/8 - Building protocol score table...")

    protocol_score_df = _build_protocol_score_frame(
        df_database=db,
        similarity_payload=similarity_payload,
        risk_scores=risk_scores,
        efficiency_scores=efficiency_scores,
        config=config,
    )

    final_score_matrix = _compute_final_score_matrix(
        similarity_matrix=np.asarray(similarity_payload["similarity"], dtype="float32"),
        efficiency_scores=protocol_score_df["efficiency_score"].to_numpy(dtype="float32"),
        risk_scores_normalized=protocol_score_df["risk_score_normalized"].to_numpy(dtype="float32"),
        config=config,
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 6/8 - Building pre-clinical ranking table...")

    recs_raw_df = _build_ranked_recommendations_df(
        similarity_payload=similarity_payload,
        protocol_score_df=protocol_score_df,
        final_score_matrix=final_score_matrix,
        config=config,
    )

    if final_filter_config is None:
        final_filter_config = FinalClinicalFilterConfig(
            top_n=config.top_n,
            patient_id_col=config.patient_id_col,
            protocol_id_col=config.protocol_id_col,
        )
    else:
        final_filter_config = replace(
            final_filter_config,
            top_n=config.top_n,
            patient_id_col=config.patient_id_col,
            protocol_id_col=config.protocol_id_col,
        )

    if config.debug:
        print("[MasterPipeline] Step 7/8 - Applying final clinical filters...")

    recs_final_df = apply_final_clinical_filters(
        recs_raw_df=recs_raw_df,
        synthetic_population=syn,
        config=final_filter_config,
    )

    recs_final_df = _recompute_rank(
        df=recs_final_df,
        patient_id_col=config.patient_id_col,
        score_col="final_score",
    )

    if config.debug:
        print("🔹 [MasterPipeline] Step 8/8 - Building clinical review tables...")

    patient_profiles_df = build_patient_review_df(
        synthetic_population=syn,
        config=review_config,
    )

    clinical_recs_df = build_clinical_recommendations_df(
        recs_df=recs_final_df,
        synthetic_population=syn,
        config=review_config,
    )

    if config.debug:
        print("[MasterPipeline] Pipeline complete.")
        print(f"   - df_db_vec           : {df_db_vec.shape}")
        print(f"   - df_syn_vec          : {df_syn_vec.shape}")
        print(f"   - recs_raw_df         : {recs_raw_df.shape}")
        print(f"   - recs_final_df       : {recs_final_df.shape}")
        print(f"   - patient_profiles_df : {patient_profiles_df.shape}")
        print(f"   - clinical_recs_df    : {clinical_recs_df.shape}")

    artifacts = {
        "st_model": st_model,
        "similarity_payload": similarity_payload,
        "risk_scores": risk_scores,
        "efficiency_scores": efficiency_scores,
        "efficiency_dataframe": df_efficiency,
        "efficiency_artifacts": efficiency_artifacts,
        "protocol_score_df": protocol_score_df,
        "final_score_matrix": final_score_matrix,
        "master_config": config.__dict__,
        "final_filter_config": final_filter_config.__dict__,
        "review_config": review_config.__dict__,
    }

    return {
        "df_db_vec": df_db_vec,
        "df_syn_vec": df_syn_vec,
        "st_model": st_model,
        "similarity_payload": similarity_payload,
        "risk_scores": risk_scores,
        "efficiency_scores": efficiency_scores,
        "protocol_score_df": protocol_score_df,
        "recs_raw_df": recs_raw_df,
        "recs_final_df": recs_final_df,
        "patient_profiles_df": patient_profiles_df,
        "clinical_recs_df": clinical_recs_df,
        "artifacts": artifacts,
    }


# ============================================================
# Optional helper for simple unpacking
# ============================================================

def run_master_recommender_pipeline_compact(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    structured_cols: List[str],
    efficiency_config: EfficiencyScorerConfig,
    config: MasterRecommenderConfig = MasterRecommenderConfig(),
    final_filter_config: Optional[FinalClinicalFilterConfig] = None,
    review_config: ClinicalReviewConfig = ClinicalReviewConfig(),
    vectorizer_kwargs: Optional[Dict[str, Any]] = None,
    similarity_kwargs: Optional[Dict[str, Any]] = None,
    risk_kwargs: Optional[Dict[str, Any]] = None,
    efficiency_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compact wrapper returning only:
    - patient_profiles_df
    - clinical_recs_df
    - recs_final_df
    """
    outputs = run_master_recommender_pipeline(
        synthetic_population=synthetic_population,
        df_database=df_database,
        structured_cols=structured_cols,
        efficiency_config=efficiency_config,
        config=config,
        final_filter_config=final_filter_config,
        review_config=review_config,
        vectorizer_kwargs=vectorizer_kwargs,
        similarity_kwargs=similarity_kwargs,
        risk_kwargs=risk_kwargs,
        efficiency_kwargs=efficiency_kwargs,
    )

    return (
        outputs["patient_profiles_df"],
        outputs["clinical_recs_df"],
        outputs["recs_final_df"],
    )