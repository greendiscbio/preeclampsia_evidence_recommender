import importlib


RUNTIME_DEPENDENCIES = [
    "numpy",
    "pandas",
    "sklearn",
    "sentence_transformers",
    "torch",
    "pyspark",
    "openai",
    "dotenv",
    "tqdm",
]


def test_runtime_dependencies_are_importable():
    """Verify that all declared direct runtime dependencies can be imported."""
    for module_name in RUNTIME_DEPENDENCIES:
        module = importlib.import_module(module_name)
        assert module is not None

