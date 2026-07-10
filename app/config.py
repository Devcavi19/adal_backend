"""Env-driven settings, shared by the ingest CLI and the API.

Values come from the environment (loaded from a local ``.env`` if present).
Only ``PINECONE_API_KEY`` is required; everything else has a working default so
a fresh checkout runs without a fully populated ``.env``.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent; real env vars still win


def _require(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# --- Pinecone ---------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "adal-theses").strip()
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine").strip()

# One namespace per corpus (theses now; handbooks/faqs later).
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "theses").strip()

# --- Embeddings -------------------------------------------------------------
# Local Ollama model, called through Ollama's native /api/embed endpoint. The
# "ollama/<name>" prefix is optional and stripped before the call (kept so the
# id reads the same as the LiteLLM chat models).
# The index dimension is LOCKED to this model's output width -- switching models
# later means recreating the index and re-embedding everything.
# bge-m3: 1024-dim, 8192-token context. Chosen to match the existing 1024-dim
# Pinecone index. (nomic-embed-text is faster at 768-dim but would need the
# index recreated; mxbai-embed-large is 1024 but its 512-token context is too
# short for the ~800-token chunks.)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ollama/bge-m3").strip()
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

# Batch sizes for embedding and upsert calls.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "100"))


def require_pinecone_key() -> str:
    """Return the Pinecone API key or raise a clear error if it is unset."""
    return _require("PINECONE_API_KEY")
