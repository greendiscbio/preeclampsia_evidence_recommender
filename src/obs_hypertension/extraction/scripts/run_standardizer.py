from src.obs_hypertension.extraction.pipeline.standardization_pipeline import run_standardization

INPUT = "/home/juandiegoarevalo/hypertension_project/llm_inclusion_exclusion/outputs/llm_kept_missing.csv"

OUTPUT = "/home/juandiegoarevalo/hypertension_project/llm_standardized_parameters/outputs/standardized_database_json_missing.csv"


def main():

    run_standardization(

        INPUT,

        OUTPUT

    )


if __name__ == "__main__":

    main()
