from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
REQUEST_SLEEP = 0.5  # para rate limiting


''' 
¿Por qué gpt-4o?
Pros
- Mejor coste/beneficio actual
- Muy buen desempeño en clasificación
- Mucho más rápido
- Excelente en extracción estructurada
- Soporta contexto largo

Contras
-Ligeramente menos “profundo” que gpt-5
'''