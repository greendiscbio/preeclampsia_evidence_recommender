"""
Main builder for protocol-level efficiency scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.obs_hypertension.recommender.efficiency_scoring.config import EfficiencyScorerConfig
from src.obs_hypertension.recommender.efficiency_scoring.feature_builder import ensure_numeric_matrix
from src.obs_hypertension.recommender.efficiency_scoring.target_builder import define_efficiency_target


def _validate_efficiency_inputs(
    df_database: pd.DataFrame,
    config: EfficiencyScorerConfig,
) -> None:
    """
    Validate efficiency-scoring inputs.

    Parameters
    ----------
    df_database : pd.DataFrame
        Input protocol dataframe.
    config : EfficiencyScorerConfig
        Efficiency model configuration.
    """
    if df_database.empty:
        raise ValueError("df_database is empty.")

    if not config.feature_cols:
        raise ValueError("EfficiencyScorerConfig.feature_cols cannot be empty.")

    if config.min_train_rows <= 0:
        raise ValueError("min_train_rows must be greater than 0.")

    if not (0.0 <= config.default_score_if_insufficient <= 1.0):
        raise ValueError("default_score_if_insufficient must be between 0 and 1.")

    if config.n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0.")

    if config.min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be greater than 0.")


def _build_efficiency_pipeline(config: EfficiencyScorerConfig) -> Pipeline:
    """
    Build leakage-safe preprocessing + RandomForest pipeline.

    Parameters
    ----------
    config : EfficiencyScorerConfig
        Efficiency model configuration.

    Returns
    -------
    Pipeline
        Sklearn pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                    class_weight=config.class_weight,
                    n_jobs=config.n_jobs,
                ),
            ),
        ]
    )


def _build_fallback_response(
    df_targeted: pd.DataFrame,
    config: EfficiencyScorerConfig,
    n_train: int,
    pos: int,
    neg: int,
    return_dataframe: bool,
    debug: bool,
) -> Tuple[np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Build fallback output when training signal is insufficient.
    """
    scores = np.full(
        len(df_targeted),
        config.default_score_if_insufficient,
        dtype="float32",
    )

    df_out = df_targeted.copy()
    df_out["efficiency_score"] = scores

    if debug:
        print("⚠️ Insufficient training signal for efficiency model.")
        print(
            f"   Returning constant efficiency_score={config.default_score_if_insufficient:.4f}"
        )

    artifacts = {
        "model": None,
        "pipeline": None,
        "feature_cols": config.feature_cols,
        "config": config.__dict__,
        "train_stats": {
            "n_train": n_train,
            "pos": pos,
            "neg": neg,
        },
    }

    return scores, (df_out if return_dataframe else None), artifacts


def compute_protocol_efficiency_scores(
    df_database: pd.DataFrame,
    config: EfficiencyScorerConfig,
    sig_col: str = "statistical_significance_vs_control",
    is_comb_col: str = "is_combination",
    drug1_col: str = "drug_1__stdcat",
    drug2_col: str = "drug_2__stdcat",
    winner1_col: str = "winner_1__stdcat",
    winner2_col: str = "winner_2__stdcat",
    return_dataframe: bool = True,
    debug: bool = True,
) -> Tuple[np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Compute protocol-level efficiency scores using RandomForestClassifier.

    Workflow
    --------
    1. Build Is_Efficient target
    2. Exclude ambiguous rows from training
    3. Train leakage-safe pipeline on selected numeric/OHE features
    4. Predict efficiency probability for all protocols
    5. Impute ambiguous rows with mean probability from non-ambiguous predictions

    Parameters
    ----------
    df_database : pd.DataFrame
        Input protocol dataframe.
    config : EfficiencyScorerConfig
        Model configuration with feature columns and RF hyperparameters.
    sig_col : str
        Statistical significance column.
    is_comb_col : str
        Combination therapy flag column.
    drug1_col : str
        First protocol drug column.
    drug2_col : str
        Second protocol drug column.
    winner1_col : str
        First winning treatment column.
    winner2_col : str
        Second winning treatment column.
    return_dataframe : bool
        Whether to return dataframe copy with Is_Efficient and efficiency_score.
    debug : bool
        Whether to print debug summaries.

    Returns
    -------
    Tuple[np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]
        - efficiency_scores: np.ndarray (K,)
        - df_out: dataframe copy with new columns, optional
        - artifacts: dict containing model, pipeline, config, train stats
    """
    _validate_efficiency_inputs(
        df_database=df_database,
        config=config,
    )

    df = df_database.copy()

    df_targeted, ambiguous_mask = define_efficiency_target(
        df=df,
        sig_col=sig_col,
        is_comb_col=is_comb_col,
        drug1_col=drug1_col,
        drug2_col=drug2_col,
        winner1_col=winner1_col,
        winner2_col=winner2_col,
    )

    df_train = df_targeted.loc[~ambiguous_mask].copy()

    y_train = df_train["Is_Efficient"].astype(int)
    n_train = len(df_train)
    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())

    if debug:
        print(f"🧪 Efficiency training rows (non-ambiguous): {n_train}")
        print(f"✅ Positives (Is_Efficient=1): {pos} | Negatives: {neg}")

    if n_train < config.min_train_rows or pos == 0 or neg == 0:
        return _build_fallback_response(
            df_targeted=df_targeted,
            config=config,
            n_train=n_train,
            pos=pos,
            neg=neg,
            return_dataframe=return_dataframe,
            debug=debug,
        )

    X_train_df = ensure_numeric_matrix(
        df=df_train,
        feature_cols=config.feature_cols,
        fill_missing_cols_with=0.0,
    )

    X_full_df = ensure_numeric_matrix(
        df=df_targeted,
        feature_cols=config.feature_cols,
        fill_missing_cols_with=0.0,
    )

    pipeline = _build_efficiency_pipeline(config)
    pipeline.fit(X_train_df, y_train)

    proba = pipeline.predict_proba(X_full_df)[:, 1].astype("float32")

    non_ambiguous_mask_array = ~ambiguous_mask.values
    mean_non_amb = float(proba[non_ambiguous_mask_array].mean())
    proba[ambiguous_mask.values] = mean_non_amb

    df_targeted["efficiency_score"] = proba

    if debug:
        print("📊 Efficiency score summary:")
        print(pd.Series(proba).describe())

    artifacts = {
        "model": pipeline.named_steps["rf"],
        "pipeline": pipeline,
        "feature_cols": config.feature_cols,
        "config": config.__dict__,
        "train_stats": {
            "n_train": n_train,
            "pos": pos,
            "neg": neg,
            "mean_non_amb": mean_non_amb,
        },
        "target_columns_used": {
            "sig_col": sig_col,
            "is_comb_col": is_comb_col,
            "drug1_col": drug1_col,
            "drug2_col": drug2_col,
            "winner1_col": winner1_col,
            "winner2_col": winner2_col,
        },
    }

    return proba, (df_targeted if return_dataframe else None), artifacts