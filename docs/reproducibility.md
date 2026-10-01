# Reproducibility Guide

This document describes how to reproduce the bundled end-to-end sample run of the antihypertensive treatment recommender.

The sample workflow is intended to verify that the recommender can be executed from standardized sample inputs through the complete recommendation pipeline and produce the expected output artifacts.

## 1. Prerequisites

Run all commands from the repository root.

A Python environment with the project dependencies must be available. Install the required dependencies using:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
