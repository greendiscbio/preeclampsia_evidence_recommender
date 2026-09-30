# Model and experiment metadata

This document records the model identifiers, API settings, processing parameters, and reproducibility-relevant defaults that are explicitly encoded in the repository. It is intended to describe the implementation represented by this repository rather than to introduce new experimental settings.

## 1. Software environment

The repository targets Python 3.8 and pins the principal runtime dependencies in `requirements.txt`:

| Package | Version |
|---|---:|
| numpy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| sentence-transformers | 3.2.1 |
| torch | 2.4.1 |
| pyspark | 3.5.1 |
| openai | 1.82.1 |
| python-dotenv | 1.0.1 |
| tqdm | 4.67.1 |

Development/test dependencies are recorded separately in `requirements-dev.txt`.

## 2. Semantic literature filtering

The semantic filtering stage uses the SentenceTransformer checkpoint:

`pritamdeka/S-BioBert-snli-multinli-stsb`

The semantic query encoded in the repository is:

> clinical use of antihypertensive drugs (labetalol, methyldopa, nifedipine, hydralazine, magnesium) to control blood pressure in pregnant women

The configured semantic similarity threshold is `0.4290`. The keyword-density threshold stored alongside this configuration is `0.000045`.

`BioBERTScorer` selects `cuda:0` when CUDA is available and otherwise uses CPU. Both the query and document embeddings are generated with `normalize_embeddings=True`; similarity is then computed as the matrix product between the normalized document embeddings and the normalized query embedding, equivalent to cosine similarity under this normalization. The scorer's default batch size is `16`.

For semantic-score distribution analysis, the repository samples `20%` of the input (`sample_fraction=0.2`) with seed `42`.

## 3. LLM eligibility screening

The eligibility-screening stage uses the OpenAI Chat Completions API with the model identifier:

`gpt-4o`

The repository fixes the following settings:

| Parameter | Value |
|---|---:|
| temperature | 0 |
| response format | JSON object |
| max output tokens | 4000 |
| request timeout | 120 s |
| maximum retries | 3 |
| initial retry sleep | 2.0 s |

Retries use an increasing sleep interval (`retry_sleep * (attempt + 1)`). The API key is read from the `OPENAI_API_KEY` environment variable through `python-dotenv` and is not stored in the repository.

The exact system and user prompt texts are externalized under `prompts/`. Their character-level equivalence to the legacy in-code prompts is checked by the repository test suite.

### API-version note

The code records the model alias `gpt-4o`, not a dated OpenAI model snapshot/checkpoint identifier. Therefore, a more specific model snapshot cannot be claimed from the repository alone. This limitation should be considered when interpreting exact long-term reproducibility of API-generated outputs.

## 4. LLM evidence standardization

The evidence-standardization stage also uses the OpenAI Chat Completions API with:

| Parameter | Value |
|---|---:|
| model | `gpt-4o` |
| temperature | 0 |
| response format | JSON object |
| maximum retries | 3 |
| retry sleep after failure | 1.5 s |

The standardization client does **not** explicitly set `max_tokens` or a request timeout in the current implementation. These values must therefore not be reported as fixed experimental parameters for this stage.

As with eligibility screening, the exact standardization system and user prompts are stored under `prompts/`, and the API credential is supplied through the `OPENAI_API_KEY` environment variable.

## 5. Synthetic population generation

The executable synthetic-population generator is configured with:

| Parameter | Value |
|---|---:|
| number of patients | 50 |
| random seed | 42 |
| patient ID prefix | `PT` |
| numeric proxy variables | enabled |
| one-hot variables | enabled |

The generator uses predefined age and gestational-age probability distributions in `src/obs_hypertension/synthetic_population/config.py`. The small files under `data/sample/` are separate reproducibility fixtures and should not be interpreted as the full synthetic cohort used by the generator.

## 6. Recommender vectorization and similarity

The recommender uses the same SentenceTransformer checkpoint as the semantic filtering stage:

`pritamdeka/S-BioBert-snli-multinli-stsb`

Its default embedding configuration is:

| Parameter | Value |
|---|---:|
| batch size | 64 |
| normalize embeddings | `True` |
| structured-profile weight | 0.60 |
| general-notes weight | 0.40 |

The recommender batch size (`64`) is distinct from the semantic literature-filter scorer default (`16`).

## 7. Protocol efficiency model

Protocol efficiency is modeled with `sklearn.ensemble.RandomForestClassifier`. The repository defaults are:

| Parameter | Value |
|---|---:|
| random state | 42 |
| minimum training rows | 25 |
| number of trees (`n_estimators`) | 400 |
| maximum tree depth | 12 |
| minimum samples per leaf | 4 |
| parallel jobs (`n_jobs`) | -1 |
| class weight | `balanced` |
| fallback score if training signal is insufficient | 0.50 |

The efficiency target is constructed from statistical significance and agreement between the reported winning treatment and the treatment represented by the protocol. Monotherapy and combination-therapy rows are handled separately by the target-building logic.

## 8. Final recommendation scoring

The composite recommendation score uses the definitive weighting configuration reported in the manuscript and encoded as the repository defaults:

| Component | Weight |
|---|---:|
| patient/evidence similarity | 0.30 |
| expected efficiency | 0.50 |
| safety | 0.20 |

Thus, the final score is computed using a relative contribution of 30% similarity, 50% expected efficiency, and 20% safety. These weights are the configuration intended for reproduction of the manuscript experiment.

Additional defaults relevant to ranking and risk/similarity processing are:

| Parameter | Value |
|---|---:|
| final recommendations (`top_n`) | 3 |
| pre-clinical-filter candidate pool (`top_k`) | 20 |
| maximum risk outcomes per type | 2 |
| maximum anomaly slots used for similarity | 5 |
| displayed score rounding | 4 decimals |

Command-line arguments in the master recommender pipeline may override configurable defaults when explicitly supplied. For reproduction of the manuscript configuration, the weights above should be retained unless an alternative configuration is being evaluated explicitly.

## 9. Reproducibility boundaries

This repository contains three complementary reproducibility layers:

1. **Prompt reproducibility:** exact prompt templates are stored under `prompts/` and protected by equivalence tests.
2. **Environment reproducibility:** Python and principal package versions are pinned through `.python-version`, `requirements.txt`, and `requirements-dev.txt`.
3. **Data-contract reproducibility:** `data/sample/` provides synthetic/fictitious inputs that conform to the current code contract and can be checked with `scripts/validate_sample_data.py` without requiring confidential clinical data.

The sample data are intended to validate interfaces and execution contracts. They are not a substitute for the complete study corpus or the full experimental datasets used to obtain the results reported in the manuscript.

## 10. Source-of-truth files

The principal implementation files underlying this metadata are:

- `src/obs_hypertension/filtering/config/semantic.py`
- `src/obs_hypertension/filtering/semantic/biobert_scorer.py`
- `src/obs_hypertension/filtering/semantic/semantic_distribution.py`
- `src/obs_hypertension/screening/config/llm.py`
- `src/obs_hypertension/screening/llm/gpt_client.py`
- `src/obs_hypertension/extraction/config/openai.py`
- `src/obs_hypertension/extraction/llm/gpt_standardizer.py`
- `src/obs_hypertension/synthetic_population/config.py`
- `src/obs_hypertension/synthetic_population/run_generate_population.py`
- `src/obs_hypertension/recommender/config/defaults.py`
- `src/obs_hypertension/recommender/efficiency_scoring/config.py`
- `src/obs_hypertension/recommender/efficiency_scoring/target_builder.py`
- `requirements.txt`

When this documentation and the implementation disagree, the versioned source code associated with the archived release should be treated as the executable source of truth.