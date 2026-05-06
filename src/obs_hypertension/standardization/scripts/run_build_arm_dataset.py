from src.obs_hypertension.standardization.pipeline.master_standardization_pipeline import run_master_pipeline

# =====================================================

INPUT_FILE = "/home/juandiegoarevalo/hypertension_project/llm_standardized_parameters/outputs/standardized_database_json_missing.csv"

ARM_OUTPUT = "/home/juandiegoarevalo/hypertension_project/standardization_code_process/outputs/arm_level_dataset_json_missing.csv"

FINAL_OUTPUT = "/home/juandiegoarevalo/hypertension_project/standardization_code_process/outputs/arm_dataset_standardized_json_missing.csv"

# =====================================================



def main():

    run_master_pipeline(

        INPUT_FILE,

        ARM_OUTPUT,

        FINAL_OUTPUT

    )



if __name__ == "__main__":

    main()