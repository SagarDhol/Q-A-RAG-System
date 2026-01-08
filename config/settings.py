import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model settings
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL = "llama3"  # Local model name for Ollama

# Vector store settings
VECTOR_STORE_PATH = str(BASE_DIR / "data/vector_store.faiss")

# Document settings
DOCUMENTS_DIR = BASE_DIR / "data/documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 3

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
