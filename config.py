import os
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# اختيارية
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- INDEX CONFIG ---
INDEX_NAME = "medical-rag"
EMBEDDING_DIM = 384  # مناسب لموديل all-MiniLM-L6-v2

# --- MODEL CONFIG ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenRouter model string
LLM_MODEL = "openai/gpt-4o-mini"

# إعدادات الـ Generation
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_output_tokens": 1024,
}