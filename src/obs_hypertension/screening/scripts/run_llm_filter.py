import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession

from src.obs_hypertension.screening.config.paths import (
    INPUT_BEFORE80,
    INPUT_AFTER80,
    OUTPUT_ROOT,
)
from src.obs_hypertension.screening.llm.inclusion_filter import apply_llm_filter_2


# =========================
# LOAD FUNCTION
# =========================

def load_parquet_folder(
    spark: SparkSession,
    path,
) -> pd.DataFrame:

    base = Path(path)

    batch_dirs = sorted(base.glob("batch_*"))

    if not batch_dirs:
        raise RuntimeError(
            f"No batch folders found in {path}"
        )

    dfs = []

    for batch_dir in batch_dirs:

        try:

            print(f"[INFO] Loading {batch_dir}")

            df = spark.read.parquet(
                str(batch_dir)
            )

            dfs.append(df)

        except Exception as exc:

            print(
                f"[WARN] Skipping {batch_dir} → {exc}"
            )

    if not dfs:
        raise RuntimeError(
            f"No valid parquet files found in {path}"
        )

    df_all = dfs[0]

    for df in dfs[1:]:

        df_all = df_all.unionByName(
            df,
            allowMissingColumns=True,
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

    try:

        print("[INFO] Loading BEFORE80 dataset")

        df_before80 = load_parquet_folder(
            spark,
            INPUT_BEFORE80,
        )

        print("[INFO] Loading AFTER80 dataset")

        df_after80 = load_parquet_folder(
            spark,
            INPUT_AFTER80,
        )

    finally:

        spark.stop()

    print("[INFO] Merging datasets")

    df_all = pd.concat(
        [
            df_before80,
            df_after80,
        ],
        ignore_index=True,
    )

    print(
        f"[INFO] Total docs loaded: {len(df_all)}"
    )

    # =========================
    # RUN LLM FILTER
    # =========================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    kept_df, rejected_df, errors_df = (
        apply_llm_filter_2(
            df_all,
            max_docs=None,
            save_raw_jsonl=True,
            raw_jsonl_path=str(
                OUTPUT_ROOT
                / "llm_screening_raw.jsonl"
            ),
            text_col="text",
            id_col="corpusid",
        )
    )

    # =========================
    # SAVE RESULTS
    # =========================

    kept_path = OUTPUT_ROOT / "llm_kept.csv"
    rejected_path = OUTPUT_ROOT / "llm_rejected.csv"
    errors_path = OUTPUT_ROOT / "llm_errors.csv"

    kept_df.to_csv(
        kept_path,
        index=False,
    )

    rejected_df.to_csv(
        rejected_path,
        index=False,
    )

    errors_df.to_csv(
        errors_path,
        index=False,
    )

    print(
        f"[DONE] Kept: {len(kept_df)} → {kept_path}"
    )

    print(
        f"[DONE] Rejected: {len(rejected_df)} "
        f"→ {rejected_path}"
    )

    print(
        f"[DONE] Errors: {len(errors_df)} "
        f"→ {errors_path}"
    )


# =========================

if __name__ == "__main__":
    main()
