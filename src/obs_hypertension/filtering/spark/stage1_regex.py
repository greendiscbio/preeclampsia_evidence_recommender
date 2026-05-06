from pyspark.sql import functions as F
from src.obs_hypertension.filtering.regex.regex_defs import (
    PREGNANCY_REGEX, HYPERTENSION_REGEX,  
)

def stage1_high_recall(df, text_col="text"):
    df = df.withColumn(
        text_col,
        F.regexp_replace(F.col(text_col).cast("string"), r"[\n\r]+", " ")
    )
    return (
        df.filter(F.col(text_col).rlike(PREGNANCY_REGEX))
          .filter(F.col(text_col).rlike(HYPERTENSION_REGEX))
    )


# def stage1_high_recall(df, text_col="text"):
#     df = df.withColumn(
#         text_col,
#         F.regexp_replace(F.col(text_col).cast("string"), r"[\n\r]+", " ")
#     )
#     return (
#         df.filter(F.col(text_col).rlike(PREGNANCY_REGEX))
#           .filter(F.col(text_col).rlike(HYPERTENSION_REGEX))
#           #.filter(F.col(text_col).rlike(DRUG_REGEX))
#           #.filter(F.col(text_col).rlike(TREATMENT_REGEX))
#     )