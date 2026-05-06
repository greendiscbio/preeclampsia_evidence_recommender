# config/semantic.py

SEMANTIC_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

SEMANTIC_QUERY = (
    "clinical use of antihypertensive drugs "
    "(labetalol, methyldopa, nifedipine, hydralazine, magnesium) "
    "to control blood pressure in pregnant women"
)

# Default threshold (will be tuned later)
DENSITY_KEYWORD_THRESHOLD = 0.000045
SEMANTIC_SIM_THRESHOLD = 0.4290
