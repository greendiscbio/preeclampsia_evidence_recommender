from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
REQUEST_SLEEP = 1.0  # para rate limiting
