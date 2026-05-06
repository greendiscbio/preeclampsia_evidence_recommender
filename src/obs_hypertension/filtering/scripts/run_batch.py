
from src.obs_hypertension.filtering.pipeline.run_pipeline import run_pipeline

def run_batch(
    spark,
    input_files,
    base_output_path,
    batch_id,
    semantic_query,
    model_name,
    density_threshold,
    sim_threshold,
):
    print(f"[INFO] Processing {len(input_files)} files")

    # ---------- Leer batch ----------
    df = (
        spark.read
        .json([str(p) for p in input_files])
        .select("corpusid", "content.text")
        .withColumnRenamed("text", "text")
        .filter("text IS NOT NULL")
    )

    # ---------- Input count ----------
    input_count = df.count()

    stats_path = base_output_path / "stats"
    stats_path.mkdir(parents=True, exist_ok=True)

    with open(stats_path / f"batch_{batch_id:04d}_input_count.txt", "w") as f:
        f.write(str(input_count))

    print(f"[INFO] Input raw count (batch {batch_id}) = {input_count}")

    if df.rdd.isEmpty():
        print("[WARN] Batch vacío tras lectura, se omite")
        return

    # ---------- Pipeline ----------
    df_filtered = run_pipeline(
        df,
        semantic_query=semantic_query,
        model_name=model_name,
        density_threshold=density_threshold,
        sim_threshold=sim_threshold,
        base_output_path=base_output_path,
        batch_id=batch_id,
        save_intermediate=True
    )

    if df_filtered.rdd.isEmpty():
        print("[INFO] Batch sin resultados tras filtrado semántico")
        return

    # ---------- Checkpoint ----------
    df_filtered = df_filtered.checkpoint(eager=True)

    # ---------- Escritura final ----------
    #final_path = base_output_path / "final" / f"batch_{batch_id:04d}"
    #final_path.mkdir(parents=True, exist_ok=True)

    # (
    #     df_filtered
    #     .repartition(200)
    #     .write
    #     .mode("overwrite")
    #     .parquet(str(final_path))
    # )

    # print(f"[SUCCESS] Final batch saved → {final_path}")