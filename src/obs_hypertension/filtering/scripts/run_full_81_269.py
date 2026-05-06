from src.obs_hypertension.filtering.config.paths import (DATA_81_269, 
                                                         OUTPUT_AFTER80,
                                                         )
from src.obs_hypertension.filtering.config.semantic import (
    SEMANTIC_QUERY,
    SEMANTIC_MODEL_NAME,
    SEMANTIC_SIM_THRESHOLD,
    DENSITY_KEYWORD_THRESHOLD,
)
from src.obs_hypertension.filtering.config.spark import init_spark
from src.obs_hypertension.filtering.scripts.run_batch import run_batch

BATCH_SIZE = 10  # ajustable según memoria/GPU

def main():
    spark = init_spark(app_name="full_pipeline_81_269")

    spark.sparkContext.setCheckpointDir(
        "/home/juandiegoarevalo/spark_checkpoints"
    )

    files = sorted(DATA_81_269.glob("*.json.gz"))
    print(f"[INFO] Total files found: {len(files)}")

    if not files:
        print("[ERROR] No files found in DATA_81_269")
        return
    
    base_output_path = (
        OUTPUT_AFTER80 / "from81_to269_regex_density_biobert_thr_P95_v2"
    )
    base_output_path.mkdir(parents=True, exist_ok=True)

    # 3️⃣ Procesamiento por batches
    for i in range(0, len(files), BATCH_SIZE):
        batch_files = files[i:i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE

        final_path = base_output_path / "final" / f"batch_{batch_id:04d}"

        # Reanudabilidad
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

    print("[DONE] Full 81–269 corpus processed successfully")


if __name__ == "__main__":
    main()

