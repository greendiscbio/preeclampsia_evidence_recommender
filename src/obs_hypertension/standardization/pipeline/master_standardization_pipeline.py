"""
MASTER STANDARDIZATION PIPELINE

Build arm dataset
Apply therapy standardization
Apply cohort standardization

FINAL OUTPUT:
arm_dataset_standardized.csv
"""

import pandas as pd


from src.obs_hypertension.standardization.pipeline.arm_level_organizer import (

    build_arm_dataset_df,
    save_arm_dataset

)


from src.obs_hypertension.standardization.standardizers.therapy_standardizer import (

    standardize_therapy_columns

)


from src.obs_hypertension.standardization.standardizers.cohort_standardizer import (

    standardize_cohort_columns

)

from src.obs_hypertension.standardization.standardizers.outcome_standardizer import (
    standardize_outcome_columns
)

from src.obs_hypertension.standardization.standardizers.one_hot_encoder import (
    one_hot_encode_standardized_columns,
)


def run_master_pipeline(
    input_file: str,
    arm_output: str,
    final_output: str,
) -> pd.DataFrame:
    """
    Run the full master standardization pipeline.

    Parameters
    ----------
    input_file : str
        Path to raw input file.
    arm_output : str
        Path to save intermediate arm-level dataset.
    final_output : str
        Path to save final standardized dataset.

    Returns
    -------
    pd.DataFrame
        Final processed dataframe.
    """

    print()
    print("STEP 1: Building arm dataset")
    print()

    arm_df = build_arm_dataset_df(input_file)

    save_arm_dataset(
        arm_df,
        arm_output,
    )

    print()
    print("STEP 2: Standardizing therapy columns")
    print()

    arm_df = standardize_therapy_columns(arm_df)

    print()
    print("STEP 3: Standardizing cohort columns")
    print()

    arm_df = standardize_cohort_columns(arm_df)

    print()
    print("STEP 4: Standardizing outcome columns")
    print()

    arm_df = standardize_outcome_columns(arm_df)

    print()
    print("STEP 5: Applying one-hot encoding")
    print()

    arm_df = one_hot_encode_standardized_columns(arm_df)

    print()
    print("STEP 6: Saving final dataset")
    print()

    arm_df.to_csv(
        final_output,
        index=False,
    )

    print()
    print("MASTER PIPELINE COMPLETE")
    print()
    print("Final file:")
    print(final_output)

    return arm_df


if __name__ == "__main__":
    run_master_pipeline()