"""
Schemas and dataclasses for synthetic population generator.
"""

from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    n_patients: int = 50
    seed: int = 42
    patient_prefix: str = "PT"
    include_numeric_proxies: bool = True
    include_one_hot_columns: bool = True