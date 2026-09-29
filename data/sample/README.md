# Reproducibility sample data

This directory contains a small, non-clinical dataset for exercising the repository's existing recommender pipeline. It is intentionally designed from the current code contracts rather than from a generic replacement schema.

## Files

- `synthetic_profiles_sample.csv`: 6 synthetic patient profiles using the same 21 structured one-hot features and patient fields produced by the repository's synthetic-population generator. The rows cover normotensive, non-severe, severe, and contraindication-relevant examples.
- `evidence_repository_sample.csv`: 30 synthetic evidence units using the standardized treatment, efficiency-feature, outcome, and anomaly fields consumed by the recommender. It contains both efficiency target classes, risk-scoring outcomes, and `profile` anomalies.
- `schema.json`: machine-readable contract used by the validator.
- `../../scripts/validate_sample_data.py`: executable validation of required columns, binary/one-hot structure, categories, efficiency training signal, and outcome payloads.

## Validation

From the repository root, run:

```bash
python scripts/validate_sample_data.py
```

A valid sample prints `Sample data validation passed.` together with the observed row/column counts.

## Scope

These files are synthetic fixtures for reproducibility and smoke testing. They are not patient data, do not reproduce the prevalence or statistical distribution of the full study corpus, and must not be interpreted as clinical evidence. The 30-row evidence sample is deliberately large enough to exceed the current efficiency model's `min_train_rows=25` threshold while containing both target classes.
