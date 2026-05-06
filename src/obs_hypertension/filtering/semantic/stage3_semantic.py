def stage3_semantic_distributed(
    df,
    semantic_query,
    model_name,
    sim_threshold
):
    from src.obs_hypertension.filtering.semantic.biobert_scorer import BioBERTScorer
    from pyspark.sql import functions as F
    scorer = BioBERTScorer(
        model_name=model_name,
        semantic_query=semantic_query,
        sim_threshold=sim_threshold,
        return_all_scores= False
    )

    rdd = (
            df.select("corpusid", "text")
            .rdd
            .mapPartitions(scorer.score_partition)
        )
    
    if rdd.isEmpty():
        return df.sparkSession.createDataFrame(
            [], schema="corpusid string, semantic_score double"
        )
    
    df_scores = rdd.toDF(["corpusid", "semantic_score"])

    # Join scores back with original df to recover text column
    df_out = (
        df
        .join(df_scores, on="corpusid", how="inner")
        .select("corpusid","semantic_score","text")
    )

    return  df_out
