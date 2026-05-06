import pandas as pd

from tqdm import tqdm

from src.obs_hypertension.standardization.config.utils import safe_json_loads, safe_get

from src.obs_hypertension.standardization.pipeline.flatteners import (

    flatten_all_therapies,
    flatten_outcomes

)

from src.obs_hypertension.standardization.config.config import COLUMN_ORDER



def build_arm_dataset_df(input_file: str) -> pd.DataFrame:


    df = pd.read_csv(input_file)

    rows = []


    for _, row in tqdm(df.iterrows(), total=len(df)):


        standardized = safe_json_loads(row["standardized"])


        base = {

            "corpusid": safe_get(row.get("corpusid")),

            "title": safe_get(row.get("title")),

            "year": safe_get(row.get("year")),

            "doi": safe_get(row.get("doi")),

            "study_design": safe_get(row.get("study_design")),

            "cohort_description":

                safe_get(standardized.get("cohort", {}).get("cohort_description")),

            "maternal_clinical_diagnosis":

                safe_get(standardized.get("cohort", {}).get("maternal_clinical_diagnosis")),

            "cohort_stratification":

                safe_get(standardized.get("cohort", {}).get("cohort_stratification")),

            "maternal_age_range":

                safe_get(standardized.get("cohort", {}).get("maternal_age_range")),

            "gestational_age_range":

                safe_get(standardized.get("cohort", {}).get("gestational_age_weeks_range")),

            "systolic_pressure_range":

                safe_get(standardized.get("cohort", {}).get("systolic_pressure_range")),

            "diastolic_pressure_range":

                safe_get(standardized.get("cohort", {}).get("diastolic_pressure_range")),

            "main_findings":

                safe_get(standardized.get("main_findings")),

            "winner":

                safe_get(standardized.get("study_interpretation", {}).get("winner")),

            "winner_reason":

                safe_get(standardized.get("study_interpretation", {}).get("winner_reason")),

            "statistical_significance_vs_control":

                safe_get(

                    standardized.get(

                        "study_interpretation",

                        {}

                    ).get(

                        "statistical_significance_vs_control"

                    )

                )

        }


        arms = standardized.get("arms", [])


        for arm in arms:


            arm_row = base.copy()


            arm_row.update({

                "group_label": safe_get(arm.get("group_label")),

                "sample_size": safe_get(arm.get("sample_size"))

            })


            arm_row.update(

                flatten_all_therapies(

                    arm.get("therapy", [])

                )

            )


            arm_row.update(

                flatten_outcomes(

                    arm.get("maternal_outcomes", []),

                    "maternal"

                )

            )


            arm_row.update(

                flatten_outcomes(

                    arm.get("fetal_outcomes", []),

                    "fetal"

                )

            )


            arm_row.update(

                flatten_outcomes(

                    arm.get("anomaly_outcomes", []),

                    "anomaly"

                )

            )


            rows.append(arm_row)



    final_df = pd.DataFrame(rows)


    for col in COLUMN_ORDER:

        if col not in final_df.columns:

            final_df[col] = "not specified"


    final_df = final_df[COLUMN_ORDER]


    return final_df



def save_arm_dataset(df: pd.DataFrame, output_file: str):

    df.to_csv(output_file, index=False)

    print()

    print("Arm dataset saved:")

    print(output_file)

    print("Rows:", len(df))