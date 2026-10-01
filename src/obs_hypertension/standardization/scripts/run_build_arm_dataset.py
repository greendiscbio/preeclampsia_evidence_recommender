from src.obs_hypertension.standardization.config.paths import (
    INPUT_FILE,
    ARM_OUTPUT,
    FINAL_OUTPUT,
)
from src.obs_hypertension.standardization.pipeline.master_standardization_pipeline import (
    run_master_pipeline,
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Extraction output not found: {INPUT_FILE}\n"
            "Run the extraction pipeline before standardization."
        )

    ARM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    run_master_pipeline(
        str(INPUT_FILE),
        str(ARM_OUTPUT),
        str(FINAL_OUTPUT),
    )


if __name__ == "__main__":
    main()
