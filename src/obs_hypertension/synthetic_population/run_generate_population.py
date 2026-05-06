"""
Executable script to generate synthetic population dataset.
"""

from __future__ import annotations

from pathlib import Path

from src.obs_hypertension.synthetic_population.pipeline import generate_synthetic_population_pipeline
from src.obs_hypertension.synthetic_population.schemas import GeneratorConfig


def main() -> None:
    config = GeneratorConfig(
        n_patients=50,
        seed=42,
        patient_prefix="PT",
        include_numeric_proxies=True,
        include_one_hot_columns=True,
    )

    output_path = Path("synthetic_population_generator") / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_population_pipeline(config)

    file_path = output_path / "synthetic_population.csv"
    df.to_csv(file_path, index=False)

    print()
    print("SYNTHETIC POPULATION GENERATION COMPLETE")
    print()
    print(f"Rows generated: {len(df)}")
    print(f"Output file: {file_path}")
    print()


if __name__ == "__main__":
    main()