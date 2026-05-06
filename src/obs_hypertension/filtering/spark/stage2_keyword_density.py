from typing import List
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

from src.obs_hypertension.filtering.regex.regex_defs import PRIMARY_DRUGS


def count_kwds(text: str, keyword_list: List[str]) -> int:
    if text is None:
        return 0
    text_lower = text.lower()
    return sum(text_lower.count(k.lower()) for k in keyword_list)


def register_stage2_keyword_udfs(spark):
    return F.udf(lambda x: count_kwds(x, PRIMARY_DRUGS), IntegerType())


def stage2_keyword_density(
    df,
    count_udf,
    text_column="text",
    length_column="Doc_Length",
    thr_normalized_t=0.000045,
):
    """
    Stage 2 basado en densidad normalizada de keywords clínicas.
    Score = (# keywords treatment) / longitud del documento
    """

    df = df.withColumn(
        length_column,
        F.length(F.col(text_column))
    )

    df = df.withColumn(
        "Kwd_Frequency_T",
        count_udf(F.col(text_column))
    )

    df = df.withColumn(
        "Kwd_Normalized_Score_T",
        F.col("Kwd_Frequency_T").cast(DoubleType())
        / F.col(length_column)
    )

    df_filtered = df.filter(
        F.col("Kwd_Normalized_Score_T") >= thr_normalized_t
    )

    return df_filtered


# def stage2_keyword_density(
#     df,
#     count_udf,
#     text_column="text",
#     length_column="Doc_Length",
#     thr_normalized_t= 0.000045,
# ):
#     """
#     Stage 2 basado en densidad normalizada de keywords clínicas.
#     Score = (# keywords treatment) / longitud del documento
#     """

#     df = df.withColumn(
#         "Kwd_Frequency_T",
#         count_udf(F.col(text_column))
#     )

#     df = df.withColumn(
#         "Kwd_Normalized_Score_T",
#         F.col("Kwd_Frequency_T").cast(DoubleType())
#         / F.col(length_column)
#     )

#     df_filtered = df.filter(
#         F.col("Kwd_Normalized_Score_T") >= thr_normalized_t
#     )

#     return df_filtered

