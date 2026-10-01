# Reproducibility Guide

This document describes how to reproduce the bundled end-to-end sample run of the antihypertensive treatment recommender.

The sample workflow verifies that the recommender can be executed from standardized synthetic inputs through the complete recommendation pipeline and produce the expected output artifacts.

The bundled sample data are intended for software reproducibility, interface validation, and smoke testing. They are not clinical data and are not intended to reproduce the statistical properties or results of the complete study corpus.

## 1. Prerequisites

Run all commands from the repository root.

The repository targets:

- Python 3.8.10
- PySpark 3.5.1
- a Java runtime compatible with the bundled PySpark version

The development environment used for the reproducibility check included OpenJDK 11.

The expected Python version is also recorded in `.python-version`.

Check the environment with:

```bash
python --version
python -m pip --version
java -version
```

## 2. Create the Python environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the runtime and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Check dependency consistency:

```bash
python -m pip check
```

A consistent environment should report:

```text
No broken requirements found.
```

## 3. Bundled sample data

The reproducibility workflow uses two version-controlled synthetic input files:

```text
data/sample/synthetic_profiles_sample.csv
data/sample/evidence_repository_sample.csv
```

Their machine-readable data contract is stored in:

```text
data/sample/schema.json
```

Additional information about the fixtures is provided in:

```text
data/sample/README.md
```

These files are synthetic reproducibility fixtures. They do not contain patient data and must not be interpreted as clinical evidence.

## 4. Validate the sample inputs

Before running the recommender, validate the bundled sample data:

```bash
PYTHONPATH=. python scripts/validate_sample_data.py
```

For the current bundled fixtures, successful validation reports:

```text
Sample data validation passed.
  patients: 6 rows x 31 columns
  evidence: 30 rows x 107 columns
```

This validation checks that the sample inputs satisfy the data contract required by the current recommender implementation.

## 5. Optional PySpark smoke test

PySpark is included among the repository dependencies and requires a working Java runtime.

A minimal local smoke test can be performed with:

```bash
PYTHONPATH=. python - <<'PY'
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("reproducibility-smoke-test")
    .getOrCreate()
)

print("Spark version:", spark.version)

df = spark.createDataFrame(
    [(1, "test"), (2, "reproducibility")],
    ["id", "text"],
)

rows = df.count()
print("Spark test rows:", rows)

assert rows == 2

spark.stop()

print("SPARK SMOKE TEST PASSED")
PY
```

The validated environment reports PySpark 3.5.1 and successfully executes the local test.

Platform-specific Spark warnings may appear even when the smoke test completes successfully.

## 6. Run the end-to-end recommender sample

Use a temporary output directory so that generated artifacts do not modify the repository working tree:

```bash
SAMPLE_OUTPUT="/tmp/preeclampsia_sample_run"

rm -rf "$SAMPLE_OUTPUT"

PYTHONPATH=. python \
  src/obs_hypertension/recommender/scripts/run_master_recommender_pipeline.py \
  --database-path data/sample/evidence_repository_sample.csv \
  --synthetic-path data/sample/synthetic_profiles_sample.csv \
  --output-dir "$SAMPLE_OUTPUT"
```

The master pipeline executes the recommender using the repository defaults unless command-line overrides are explicitly supplied.

The manuscript-aligned default final-score weights are:

```text
similarity_weight = 0.30
efficiency_weight = 0.50
safety_weight = 0.20
```

## 7. Generated artifacts

A successful sample run currently produces:

```text
artifacts_summary.json
clinical_recs_df.csv
df_db_vec.csv
df_syn_vec.csv
patient_profiles_df.csv
protocol_score_df.csv
recs_final_df.csv
recs_raw_df.csv
```

Inspect the generated files with:

```bash
find "$SAMPLE_OUTPUT" -maxdepth 2 -type f -printf '%P\n' | sort
```

The current bundled sample produces the following artifact shapes:

| Artifact | Rows | Columns |
|---|---:|---:|
| `df_db_vec.csv` | 30 | 112 |
| `df_syn_vec.csv` | 6 | 35 |
| `protocol_score_df.csv` | 30 | 17 |
| `recs_raw_df.csv` | 120 | 23 |
| `recs_final_df.csv` | 16 | 24 |
| `patient_profiles_df.csv` | 6 | 8 |
| `clinical_recs_df.csv` | 16 | 17 |

The expected configuration and observed artifact dimensions are also recorded in `artifacts_summary.json`.

## 8. Validate the generated sample run

Validate the generated artifacts with:

```bash
PYTHONPATH=. python scripts/validate_sample_run.py \
  --output-dir "$SAMPLE_OUTPUT"
```

A successful validation reports:

```text
Sample recommender run validation passed.
```

This validation is intended to detect violations of the expected output contract and reproducibility configuration.

## 9. Run the automated test suite

Run the repository tests with:

```bash
PYTHONPATH=. pytest -q
```

The current repository test suite should complete without failures.

## 10. Final environment checks

Check Python dependency consistency again:

```bash
python -m pip check
```

Optionally verify that the reproducibility run did not create tracked repository outputs:

```bash
git status --short
```

When the workflow is executed with the temporary output directory shown above, the generated sample artifacts remain outside the repository.

## 11. Reproducibility boundaries

The bundled workflow demonstrates that the current recommender implementation can execute end to end from version-controlled synthetic inputs and satisfy its expected software and data contracts.

It does not reproduce the complete experimental study from raw literature acquisition because large source datasets, complete processed evidence repositories, external API interactions, and some research artifacts are not distributed with the repository.

Accordingly, the bundled sample should be interpreted as an executable end-to-end reproducibility and smoke-testing workflow for the recommender implementation, rather than as a replacement for the complete study dataset.

For additional implementation parameters and source-of-truth configuration files, see `docs/model_metadata.md`.
