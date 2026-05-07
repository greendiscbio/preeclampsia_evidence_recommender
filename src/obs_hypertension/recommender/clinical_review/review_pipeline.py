"""
Meta-orchestrator for clinical recommendation review outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.obs_hypertension.recommender.clinical_review.config import ClinicalReviewConfig
from src.obs_hypertension.recommender.clinical_review.review_builders import (
    build_clinical_recommendations_df,
    build_patient_review_df,
)
from src.obs_hypertension.recommender.clinical_review.utils import (
    optionally_sample_dataframe,
)


def _validate_review_pipeline_inputs(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    fn_vectorize,
    fn_orchestrator,
    fn_similarity,
    fn_risk,
    fn_efficiency,
) -> None:
    """
    Validate main inputs for review pipeline.

    Parameters
    ----------
    synthetic_population : pd.DataFrame
        Synthetic patient dataframe.
    df_database : pd.DataFrame
        Protocol database dataframe.
    fn_vectorize : callable
        Vectorization function.
    fn_orchestrator : callable
        Master recommender/orchestrator function.
    fn_similarity : callable
        Similarity function.
    fn_risk : callable
        Risk function.
    fn_efficiency : callable
        Efficiency function.
    """
    if synthetic_population is None or synthetic_population.empty:
        raise ValueError("synthetic_population is empty.")

    if df_database is None or df_database.empty:
        raise ValueError("df_database is empty.")

    if fn_vectorize is None:
        raise ValueError("fn_vectorize cannot be None.")

    if fn_orchestrator is None:
        raise ValueError("fn_orchestrator cannot be None.")

    if fn_similarity is None:
        raise ValueError("fn_similarity cannot be None.")

    if fn_risk is None:
        raise ValueError("fn_risk cannot be None.")

    if fn_efficiency is None:
        raise ValueError("fn_efficiency cannot be None.")


def run_clinical_recommendation_review(
    synthetic_population: pd.DataFrame,
    df_database: pd.DataFrame,
    fn_vectorize,
    fn_orchestrator,
    efficiency_config,
    fn_similarity,
    fn_risk,
    fn_efficiency,
    structured_cols: Optional[List[str]] = None,
    vectorizer_kwargs: Optional[Dict[str, Any]] = None,
    orchestrator_kwargs: Optional[Dict[str, Any]] = None,
    review_config: ClinicalReviewConfig = ClinicalReviewConfig(),
    top_n: int = 3,
    sample_size: Optional[int] = None,
    random_state: int = 42,
    debug: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run full review-oriented recommendation workflow.

    Workflow
    --------
    1. Optionally sample synthetic population
    2. Vectorize database and sampled synthetic patients
    3. Run master recommender orchestrator
    4. Build clinical validation tables

    Parameters
    ----------
    synthetic_population : pd.DataFrame
        Synthetic patient population.
    df_database : pd.DataFrame
        Protocol database.
    fn_vectorize : callable
        Function that vectorizes database and synthetic population.
    fn_orchestrator : callable
        Main recommendation orchestrator function.
    efficiency_config : Any
        Efficiency scoring configuration object.
    fn_similarity : callable
        Similarity scoring function.
    fn_risk : callable
        Risk scoring function.
    fn_efficiency : callable
        Efficiency scoring function.
    structured_cols : Optional[List[str]]
        Structured columns used by vectorizer.
    vectorizer_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to vectorizer.
    orchestrator_kwargs : Optional[Dict[str, Any]]
        Extra kwargs passed to main orchestrator.
    review_config : ClinicalReviewConfig
        Configuration for final clinical review tables.
    top_n : int
        Maximum number of recommendations per patient.
    sample_size : Optional[int]
        Optional number of synthetic patients to sample.
    random_state : int
        Sampling seed.
    debug : bool
        Whether to print debug messages.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        patient_profiles_df,
        clinical_recs_df,
        recs_raw_df
    """
    _validate_review_pipeline_inputs(
        synthetic_population=synthetic_population,
        df_database=df_database,
        fn_vectorize=fn_vectorize,
        fn_orchestrator=fn_orchestrator,
        fn_similarity=fn_similarity,
        fn_risk=fn_risk,
        fn_efficiency=fn_efficiency,
    )

    syn = optionally_sample_dataframe(
        df=synthetic_population,
        sample_size=sample_size,
        random_state=random_state,
    )

    if debug and sample_size is not None and len(syn) < len(synthetic_population):
        print(f"🔹 [ClinicalReview] Sampled synthetic patients: {len(syn)}")

    if debug:
        print("🔹 [ClinicalReview] Vectorizing inputs once...")

    vectorizer_kwargs = vectorizer_kwargs or {}
    if structured_cols is not None:
        vectorizer_kwargs = {
            **vectorizer_kwargs,
            "structured_cols": structured_cols,
        }

    df_db_vec, df_syn_vec, st_model = fn_vectorize(
        df_database=df_database,
        synthetic_population=syn,
        **vectorizer_kwargs,
    )

    if debug:
        print("🔹 [ClinicalReview] Running master recommender orchestrator...")

    orchestrator_kwargs = orchestrator_kwargs or {}

    recs_long_df, recs_raw_df = fn_orchestrator(
        df_database=df_database,
        synthetic_population=syn,
        st_model=st_model,
        efficiency_config=efficiency_config,
        df_db_vec=df_db_vec,
        df_syn_vec=df_syn_vec,
        fn_similarity=fn_similarity,
        fn_risk=fn_risk,
        fn_efficiency=fn_efficiency,
        top_n=top_n,
        debug=debug,
        **orchestrator_kwargs,
    )

    patient_profiles_df = build_patient_review_df(
        synthetic_population=syn,
        config=review_config,
    )

    clinical_recs_df = build_clinical_recommendations_df(
        recs_df=recs_long_df,
        synthetic_population=syn,
        config=review_config,
    )

    if debug:
        print("✅ [ClinicalReview] Clinical review tables built.")
        print(f"   - patient_profiles_df: {patient_profiles_df.shape}")
        print(f"   - clinical_recs_df   : {clinical_recs_df.shape}")
        print(f"   - recs_raw_df        : {recs_raw_df.shape}")

    return patient_profiles_df, clinical_recs_df, recs_raw_df