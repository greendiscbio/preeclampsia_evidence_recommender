"""
Main orchestration pipeline for synthetic population generation.
"""

from __future__ import annotations

import pandas as pd

from src.obs_hypertension.synthetic_population.generators import generate_population_base
from src.obs_hypertension.synthetic_population.postprocessing import postprocess_population
from src.obs_hypertension.synthetic_population.schemas import GeneratorConfig


def generate_synthetic_population_pipeline(config: GeneratorConfig) -> pd.DataFrame:
    """
    Generate and postprocess a synthetic population according to configuration.
    """
    df = generate_population_base(
        n=config.n_patients,
        seed=config.seed,
        patient_prefix=config.patient_prefix,
    )

    df = postprocess_population(
        df=df,
        include_numeric_proxies=config.include_numeric_proxies,
        include_one_hot_columns=config.include_one_hot_columns,
    )

    return df