from src.obs_hypertension.extraction.config.paths import (
    INPUT_CSV,
    OUTPUT_CSV,
)
from src.obs_hypertension.extraction.pipeline.standardization_pipeline import (
    run_standardization,
)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Screening output not found: {INPUT_CSV}\n"
            "Run the screening pipeline before extraction."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    run_standardization(
        str(INPUT_CSV),
        str(OUTPUT_CSV),
    )


if __name__ == "__main__":
    main()
