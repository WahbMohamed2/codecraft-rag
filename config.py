import os
from dotenv import load_dotenv

load_dotenv()

# ── API ───────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = (
    "deepseek/deepseek-r1-0528-qwen3-8b:free"  # Supports system prompts reliably
)

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_humaneval"
CHROMA_COLLECTION = "humaneval_prompts"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 3

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME = "openai/openai_humaneval"
DATASET_SPLIT = "test"
