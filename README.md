# An Explainable Artificial Intelligence Framework for Evidence-Guided Clinical Decision Support: A Proof-of-Concept Study in Hypertensive Disorders of Pregnancy
Official code repository accompanying the scientific manuscript:

> **An Explainable Artificial Intelligence Framework for Evidence-Guided Clinical Decision Support: A Proof-of-Concept Study in Hypertensive Disorders of Pregnancy**

---

## Overview

The exponential growth of scientific literature has created major challenges for transforming heterogeneous evidence into actionable and context-aware decision-making systems. Traditional evidence synthesis methodologies such as systematic reviews provide rigorous aggregation of knowledge but remain resource-intensive, difficult to scale, and poorly adapted for personalized recommendation scenarios.

This repository presents a unified artificial intelligence framework that integrates:

* Automated large-scale scientific evidence retrieval
* Multi-stage filtering and semantic screening
* LLM-based inclusion and exclusion assessment
* Structured evidence extraction and standardization
* Synthetic population generation
* Recommendation-oriented decision modeling
* Risk-aware and efficiency-aware ranking
* Interpretable recommendation scoring

Although the framework is domain-agnostic by design, the current implementation is evaluated in a clinical proof-of-concept focused on antihypertensive therapies during pregnancy.

---

# Framework Architecture

The repository is organized into modular components that reproduce the complete evidence-to-recommendation pipeline described in the manuscript.

```text
Scientific Literature
        ↓
High-Recall Retrieval Pipeline
        ↓
LLM-Based Screening
        ↓
Structured Evidence Extraction
        ↓
Clinical Standardization Pipeline
        ↓
Structured Evidence Repository
        ↓
Synthetic Population Generation
        ↓
Recommendation Engine
        ↓
Risk & Efficiency-Aware Ranking
        ↓
Interpretable Decision Support
```

---

# Repository Structure

```text
obs_hypertension/
│
├── src/
│   └── obs_hypertension/
│       ├── extraction/
│       ├── filtering/
│       ├── recommender/
│       ├── screening/
│       ├── standardization/
│       └── synthetic_population/
│
├── logs/
├── archive/
├── setup.py
└── README.md
```

---

# Core Modules

## 1. Filtering Module

Location:

```text
src/obs_hypertension/filtering/
```

Implements the large-scale evidence retrieval and filtering pipeline.

### Main functionalities

* Rule-based keyword retrieval
* High-recall regex filtering
* Keyword-density filtering
* Semantic similarity filtering using BioBERT embeddings
* Spark-based distributed processing
* PRISMA-oriented evidence selection pipeline

### Submodules

| Submodule   | Description                            |
| ----------- | -------------------------------------- |
| `config/`   | Pipeline configuration files           |
| `pipeline/` | End-to-end filtering orchestration     |
| `regex/`    | High-recall regex retrieval rules      |
| `semantic/` | Semantic similarity scoring components |
| `spark/`    | Distributed processing utilities       |
| `scripts/`  | Executable filtering pipelines         |

---

## 2. Screening Module

Location:

```text
src/obs_hypertension/screening/
```

Implements high-precision LLM-based screening for study inclusion and exclusion.

### Main functionalities

* GPT-based evidence eligibility screening
* Inclusion/exclusion decision modeling
* Structured JSON-based evidence validation
* Automated screening workflows

### Included components

| Component             | Description                       |
| --------------------- | --------------------------------- |
| `gpt_client.py`       | OpenAI/GPT communication layer    |
| `inclusion_filter.py` | LLM-based study eligibility logic |
| `run_llm_filter.py`   | Main execution script             |

---

## 3. Extraction Module

Location:

```text
src/obs_hypertension/extraction/
```

Transforms screened scientific studies into structured evidence representations.

### Main functionalities

* LLM-assisted parameter extraction
* Comparative arm extraction
* Structured cohort representation
* Pipeline orchestration for evidence parsing

### Submodules

| Submodule   | Description                   |
| ----------- | ----------------------------- |
| `config/`   | Extraction configuration      |
| `llm/`      | LLM extraction logic          |
| `pipeline/` | Extraction workflows          |
| `scripts/`  | Executable extraction scripts |

---

## 4. Standardization Module

Location:

```text
src/obs_hypertension/standardization/
```

Performs harmonization and normalization of extracted evidence.

### Main functionalities

* Drug name standardization
* Dosage harmonization
* Clinical terminology normalization
* Structured cohort alignment
* Comparative treatment-arm dataset construction

### Submodules

| Submodule        | Description                   |
| ---------------- | ----------------------------- |
| `config/`        | Standardization configuration |
| `pipeline/`      | Standardization workflows     |
| `standardizers/` | Core harmonization functions  |
| `scripts/`       | Dataset construction scripts  |

---

## 5. Synthetic Population Module

Location:

```text
src/obs_hypertension/synthetic_population/
```

Generates synthetic target populations for recommendation evaluation.

### Main functionalities

* Clinical profile generation
* Conditional population simulation
* Structured patient schema generation
* Postprocessing and cohort balancing

### Main files

| File                | Description                         |
| ------------------- | ----------------------------------- |
| `generators.py`     | Synthetic patient generation logic  |
| `pipeline.py`       | Population generation orchestration |
| `schemas.py`        | Structured patient schemas          |
| `postprocessing.py` | Cohort balancing and cleanup        |

---

## 6. Recommender Module

Location:

```text
src/obs_hypertension/recommender/
```

Implements the recommendation-oriented decision support framework.

### Main functionalities

* Similarity-based retrieval
* Evidence-derived efficiency scoring
* Risk-aware penalization
* Multi-objective ranking
* Clinical-rule postprocessing
* Interpretable recommendation generation

### Submodules

| Submodule             | Description                        |
| --------------------- | ---------------------------------- |
| `similarity_scoring/` | Similarity retrieval engine        |
| `efficiency_scoring/` | Efficiency estimation models       |
| `risk_scoring/`       | Risk-aware scoring framework       |
| `vectorization/`      | Embedding and vector pipelines     |
| `clinical_review/`    | Clinical validation components     |
| `final_orchestrator/` | Final recommendation orchestration |
| `pipeline/`           | End-to-end recommender execution   |
| `scripts/`            | Executable recommendation scripts  |

---

# Methodological Pipeline

The framework follows a multi-stage evidence processing pipeline:

1. Large-scale scientific document retrieval
2. Rule-based high-recall filtering
3. Semantic similarity refinement
4. LLM-based inclusion/exclusion screening
5. Structured evidence extraction
6. Standardization and harmonization
7. Synthetic target population generation
8. Similarity-based retrieval
9. Efficiency estimation
10. Risk-aware penalization
11. Multi-objective ranking
12. Interpretable recommendation generation

---

# Recommendation Framework

The recommendation engine combines three major dimensions:

## Similarity-Based Retrieval

Retrieves evidence units most compatible with the target profile.

## Efficiency Estimation

Estimates evidence-derived therapeutic effectiveness using structured clinical variables.

## Risk-Aware Penalization

Incorporates adverse outcomes and anomaly severity into the ranking process.

The final recommendation score is formulated as:

```text
Final Score = α(Similarity) + β(Efficiency) − γ(Risk)
```

where:

* Similarity represents cohort compatibility
* Efficiency estimates expected treatment effectiveness
* Risk penalizes adverse or unsafe profiles

---

# Installation

## Clone repository

```bash
git clone git@github.com:greendiscbio/obs_hypertension.git
cd obs_hypertension
```

## Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Example Usage

## Run filtering pipeline

```bash
python src/obs_hypertension/filtering/scripts/run_full_81_269.py
```

## Run LLM screening

```bash
python src/obs_hypertension/screening/scripts/run_llm_filter.py
```

## Build standardized dataset

```bash
python src/obs_hypertension/standardization/scripts/run_build_arm_dataset.py
```

## Generate synthetic population

```bash
python src/obs_hypertension/synthetic_population/run_generate_population.py
```

---

# Research Context

This repository accompanies ongoing research on:

* Automated evidence synthesis
* Retrieval-augmented recommendation systems
* Interpretable AI for decision support
* Clinical AI pipelines
* Evidence-driven recommendation frameworks
* Risk-aware ranking systems

The current proof-of-concept implementation is evaluated within a maternal health scenario focused on antihypertensive therapy during pregnancy.

---

# Reproducibility Notes

Some large datasets, intermediate outputs, model artifacts, and processed evidence files are intentionally excluded from the repository due to:

* Storage limitations
* Data governance considerations
* Research reproducibility management
* Computational constraints

The repository focuses on providing:

* Core framework architecture
* Reproducible pipeline logic
* Modular implementation structure
* Recommendation methodology
* Experimental orchestration code

---

# Citation

If you use this repository or methodology in academic work, please cite:

```text
A Unified AI Framework for Automated Evidence Retrieval and Recommendation-Oriented Decision Support
```

Citation details will be updated after publication.

---

# Disclaimer

This repository is intended for research and educational purposes only.

The current implementation is a proof-of-concept research framework and is not intended for direct clinical deployment or autonomous medical decision-making.

All recommendations generated by the system require expert human interpretation and validation.

---

# Authors

Developed as part of ongoing research on AI-driven evidence synthesis and recommendation-oriented clinical decision support.

---

# License

License information will be added upon publication.
