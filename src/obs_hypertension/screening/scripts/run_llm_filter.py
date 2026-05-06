import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession

from src.obs_hypertension.screening.config.paths import OUTPUT_ROOT
from src.obs_hypertension.screening.llm.inclusion_filter import apply_llm_filter_2


# =========================
# INPUT PATHS
# =========================

INPUT_BEFORE80 = (
    "/home/juandiegoarevalo/hypertension_project/"
    "filtering_high_recall_semantic/outputs/before80/"
    "from0_to80_regex_density_biobert_thr_P95_v2/stage3_semantic"
)

INPUT_AFTER80 = (
    "/home/juandiegoarevalo/hypertension_project/"
    "filtering_high_recall_semantic/outputs/after80/"
    "from81_to269_regex_density_biobert_thr_P95_v2/stage3_semantic"
)


# =========================
# LOAD FUNCTION
# =========================

def load_parquet_folder(spark: SparkSession, path: str) -> pd.DataFrame:

    base = Path(path)

    batch_dirs = sorted(base.glob("batch_*"))

    if not batch_dirs:

        raise RuntimeError(f"No batch folders found in {path}")

    dfs = []

    for b in batch_dirs:

        try:

            print(f"[INFO] Loading {b}")

            df = spark.read.parquet(str(b))

            dfs.append(df)

        except Exception as e:

            print(f"[WARN] Skipping {b} → {e}")

    if not dfs:

        raise RuntimeError(f"No valid parquet files found in {path}")

    df_all = dfs[0]

    for d in dfs[1:]:

        df_all = df_all.unionByName(
            d,
            allowMissingColumns=True
        )

    return df_all.toPandas()


# =========================
# MAIN
# =========================

def main():

    print("[INFO] Starting Spark session")

    spark = (
        SparkSession.builder
        .appName("LLMFilterLoader")
        .getOrCreate()
    )

    print("[INFO] Loading BEFORE80 dataset")

    df_before80 = load_parquet_folder(
        spark,
        INPUT_BEFORE80
    )

    print("[INFO] Loading AFTER80 dataset")

    df_after80 = load_parquet_folder(
        spark,
        INPUT_AFTER80
    )

    spark.stop()

    print("[INFO] Merging datasets")

    df_all = pd.concat(
        [df_before80, df_after80],
        ignore_index=True
    )

    print(f"[INFO] Total docs loaded: {len(df_all)}")


    # =========================
    # RUN LLM FILTER
    # =========================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    kept_df, rejected_df, errors_df = apply_llm_filter_2(

        df_all,

        max_docs=None ,   # cambiar a None cuando esté listo

        save_raw_jsonl=True,

        raw_jsonl_path=str(
            OUTPUT_ROOT / "llm_screening_raw.jsonl"
        ),

        text_col="text",

        id_col="corpusid",
    )


    # =========================
    # SAVE RESULTS
    # =========================

    kept_path = OUTPUT_ROOT / "llm_kept.csv"

    rej_path  = OUTPUT_ROOT / "llm_rejected.csv"

    err_path  = OUTPUT_ROOT / "llm_errors.csv"


    kept_df.to_csv(
        kept_path,
        index=False
    )

    rejected_df.to_csv(
        rej_path,
        index=False
    )

    errors_df.to_csv(
        err_path,
        index=False
    )


    print(f"[DONE] Kept: {len(kept_df)} → {kept_path}")

    print(f"[DONE] Rejected: {len(rejected_df)} → {rej_path}")

    print(f"[DONE] Errors: {len(errors_df)} → {err_path}")


# =========================

if __name__ == "__main__":

    main()
