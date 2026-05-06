from src.obs_hypertension.filtering.config.paths import (
    DATA_0_80,
    OUTPUT_BEFORE80,
)
from src.obs_hypertension.filtering.config.semantic import (
    SEMANTIC_QUERY,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_SIM_THRESHOLD,
    DENSITY_KEYWORD_THRESHOLD

)
from src.obs_hypertension.filtering.config.spark import init_spark
from src.obs_hypertension.filtering.scripts.run_batch import run_batch

# =========================
# Configuración
# =========================
BATCH_SIZE = 10


def main():
    # 1️⃣ Inicializar Spark
    spark = init_spark(app_name="full_pipeline_0_80")

    # 🔑 Checkpoint para jobs largos
    spark.sparkContext.setCheckpointDir(
        "/home/juandiegoarevalo/spark_checkpoints"
    )

    # 2️⃣ Listar archivos
    files = sorted(DATA_0_80.glob("*.json.gz"))
    #files = sorted(DATA_0_80.glob("*.json.gz"))
    print(f"[INFO] Total files found in DATA_0_80: {len(files)}")

    if not files:
        print("[ERROR] No files found in DATA_0_80")
        return

    base_output_path = (
        OUTPUT_BEFORE80 / "from0_to80_regex_density_biobert_thr_P95_v2"
    )
    base_output_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE

        final_path = base_output_path / "final" / f"batch_{batch_id:04d}"

        if final_path.exists():
            print(f"[SKIP] Batch {batch_id} already processed")
            continue

        print(
            f"[INFO] Running batch {batch_id} "
            f"({len(batch_files)} files)"
        )

        run_batch(
            spark=spark,
            input_files=batch_files,
            base_output_path=base_output_path,
            batch_id=batch_id,
            semantic_query=SEMANTIC_QUERY,
            model_name=SEMANTIC_MODEL_NAME,
            density_threshold=DENSITY_KEYWORD_THRESHOLD,
            sim_threshold=SEMANTIC_SIM_THRESHOLD,
        )

    print("[DONE] Full 0–80 corpus processed successfully")


if __name__ == "__main__":
    main()
