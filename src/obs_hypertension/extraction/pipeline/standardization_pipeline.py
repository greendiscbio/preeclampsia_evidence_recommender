import json
import pandas as pd
from tqdm import tqdm
import json

from src.obs_hypertension.extraction.llm.gpt_standardizer import query_standardizer


def run_standardization(input_csv, output_csv):

    df = pd.read_csv(input_csv)
    #df = df.head(10).copy()

    standardized_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        raw_json = {

            "corpusid" : row["corpusid"],
            "title": row["title"],
            "year": row["year"],
            "doi": row["doi"],
            "study_design" : row["study_design"],

            "cohort_summary": json.loads(row["cohort_summary"]),

            "treatment_description": json.loads(row["treatment_description"]),

            "outcome_description": json.loads(row["outcome_description"]),

            "quantitative_evidence": json.loads(row["quantitative_evidence"])

        }

        try:

            standardized = query_standardizer(raw_json)  # salida es un json estandarizado por articulo

            standardized_rows.append({

                                        "corpusid": raw_json["corpusid"],
                                        "title": row["title"],
                                        "year": row["year"],
                                        "doi": row["doi"],
                                        "study_design" : row["study_design"],
                                        "standardized": json.dumps(standardized)

                                    })

        except Exception as e:

            print("Error:", e)

    pd.DataFrame(standardized_rows).to_csv(

        output_csv,

        index=False

    ) # Hacemos un dataframe de la lista de Jsons

    # Dataframe estandarizado
