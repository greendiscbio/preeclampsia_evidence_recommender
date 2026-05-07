from pathlib import Path

from src.obs_hypertension.filtering.spark.stage1_regex import stage1_high_recall
from src.obs_hypertension.filtering.spark.stage2_keyword_density import (
    register_stage2_keyword_udfs, stage2_keyword_density
)
from src.obs_hypertension.filtering.semantic.stage3_semantic import (
    stage3_semantic_distributed
)


def save_stage(df, base_path, stage, batch_id):
    """
    Guarda un dataframe intermedio para análisis posterior.
    """
    path = Path(base_path) / stage / f"batch_{batch_id:04d}"
    path.mkdir(parents=True, exist_ok=True)

    (
        df
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(str(path))
    )

    print(f"[INFO] Saved {stage} → {path}")

def save_counts(base_path, batch_id, counts: dict):
    """
    Guarda los conteos por etapa (sin guardar los dataframes).
    """
    stats_path = Path(base_path) / "stats"
    stats_path.mkdir(parents=True, exist_ok=True)

    out_file = stats_path / f"batch_{batch_id:04d}_counts.csv"

    with open(out_file, "w") as f:
        f.write("stage,count\n")
        for k, v in counts.items():
            f.write(f"{k},{v}\n")

    print(f"[INFO] Saved counts → {out_file}")


def run_pipeline(
    df,
    semantic_query: str,
    model_name: str,
    density_threshold: float,
    sim_threshold: float,
    base_output_path,
    batch_id: int,
    save_intermediate: bool = True
):
    """
    Full filtering pipeline:
    Stage 1: Regex high recall
    Stage 2: Keyword Density
    Stage 3: Semantic verification (distributed)

    Guarda los resultados intermedios de cada etapa.
    """
    counts = {}

    # ---------- INPUT ----------
    counts["input"] = df.count()
    # ---------- Stage 1 ----------
    df1 = stage1_high_recall(df)
    counts["stage1_regex"] = df1.count()

    if save_intermediate:
        save_stage(df1, base_output_path, "stage1_regex", batch_id)

    # ---------- Stage 2 ----------
    count_udf = register_stage2_keyword_udfs(df.sparkSession)

    df2 = stage2_keyword_density(
        df1,
        count_udf,
        thr_normalized_t=density_threshold
    )

    counts["stage2_keyword_density"] = df2.count()

    if save_intermediate:
        save_stage(df2, base_output_path, "stage2_keyword_density", batch_id)

    # ---------- Stage 3 ----------
    df3 = stage3_semantic_distributed(
         df2,
         semantic_query=semantic_query,
         model_name=model_name,
         sim_threshold=sim_threshold
     )

    counts["stage3_semantic"] = df3.count()

    if save_intermediate:
         save_stage(df3, base_output_path, "stage3_semantic", batch_id)

    save_counts(base_output_path, batch_id, counts)

    return df3
