def stage3_semantic_distribution(
    df,
    semantic_query,
    model_name,
    sample_fraction=0.2
):
    """
    Devuelve TODOS los semantic_score (sin threshold)
    sobre una muestra del dataframe
    """
    from src.obs_hypertension.filtering.semantic.biobert_scorer import BioBERTScorer

    df_sample = df.sample(fraction=sample_fraction, seed=42)

    scorer = BioBERTScorer(
        model_name=model_name,
        semantic_query=semantic_query,
        sim_threshold=0.0,          # irrelevante aquí
        return_all_scores=False      # 👈 clave
    )

    rdd = (
        df_sample
        .select("corpusid", "text")
        .rdd
        .mapPartitions(scorer.score_partition)
    )

    return rdd.toDF(["corpusid","semantic_score","text"])
