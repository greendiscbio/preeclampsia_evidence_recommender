from pathlib import Path

# ===========================
# Raíz lógica del pipeline
# ===========================
# Esta carpeta contendrá todo lo necesario para este pipeline.
# Es independiente del resto del proyecto.
PIPELINE_ROOT = Path(__file__).resolve().parents[1] # sube 1 nivel desde config/

# Ruta al dataset 0-80
DATA_0_80 = Path("/home/juandiegoarevalo/TDCS/DATA80/20240326/rawdata/s2orc")
# Ruta al dataset 81-269
DATA_81_269 = Path("/home/juandiegoarevalo/TDCS/DataLake/20240326/rawdata/s2orc")

# ===========================
# Directorios de salida
# ===========================
OUTPUT_ROOT = PIPELINE_ROOT / "outputs"  # Carpeta raíz de resultados
OUTPUT_BEFORE80 = OUTPUT_ROOT / "before80"
OUTPUT_AFTER80 = OUTPUT_ROOT / "after80"

# Crear carpetas de salida automáticamente si no existen
for path in [OUTPUT_ROOT, OUTPUT_BEFORE80, OUTPUT_AFTER80]:
    path.mkdir(parents=True, exist_ok=True)

# ===========================
# Otros paths opcionales
# ===========================
# Modelo de semantic search o embeddings
#SEMANTIC_MODEL_PATH = PIPELINE_ROOT / "models" / "semantic_model"

# Carpeta para logs
#LOGS_DIR = PIPELINE_ROOT / "logs"
#LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ===========================
# Summary (opcional, para debug)
# ===========================
def print_paths_summary():
    print("Pipeline root:", PIPELINE_ROOT)
    print("Data 0-80:", DATA_0_80)
    print("Data 81-269:", DATA_81_269)
    print("Output before80:", OUTPUT_BEFORE80)
    print("Output after80:", OUTPUT_AFTER80)
    #print("Semantic model path:", SEMANTIC_MODEL_PATH)
    #print("Logs dir:", LOGS_DIR)

# Ejemplo de uso:
# if __name__ == "__main__":
#     print_paths_summary()
