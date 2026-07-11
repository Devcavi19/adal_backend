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

# --- Retrieval --------------------------------------------------------------
# Chunks fetched per query and the cosine-score floor below which matches are
# dropped. Fewer, better chunks = faster + cheaper LLM calls and fewer
# hallucinated citations. bge-m3 cosine scores are compressed: unrelated text
# (greetings, off-topic) lands at ~0.44-0.49 and real matches at ~0.55-0.75,
# so 0.50 keeps relevant chunks while sending no context for smalltalk or
# general librarian questions. (The plan's 0.35 baseline filtered nothing.)
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.50"))
# Floor used when a metadata filter (program/year) scopes the query: the
# filter already guarantees domain relevance, and browse-style questions
# ("what BSENTREP studies exist?") match content chunks at only ~0.44-0.49.
RETRIEVAL_MIN_SCORE_FILTERED = float(os.getenv("RETRIEVAL_MIN_SCORE_FILTERED", "0.40"))
# Cap chunks per thesis so one paper can't fill every slot -- "what theses
# cover X?" needs breadth. Query over-fetches, then trims to RETRIEVAL_TOP_K.
RETRIEVAL_MAX_PER_THESIS = int(os.getenv("RETRIEVAL_MAX_PER_THESIS", "2"))

# --- Chat LLM ---------------------------------------------------------------
# The chat models a user may pick, in menu order (first = default). Mix local
# (``ollama/<name>``) and cloud (``mistral/...``, ``gemini/...``) freely: local
# Ollama models are free + private; cloud models cost per token but need no GPU.
# The API exposes this list to the frontend and validates every requested model
# against it -- the client must never be able to run an unlisted (costly) model.
LLM_MODELS = [m.strip() for m in os.getenv(
    "LLM_MODELS", "ollama/qwen3.5:latest,mistral/mistral-small-latest").split(",") if m.strip()]

# Human labels for the frontend dropdown; unknown ids fall back to the id itself.
MODEL_LABELS = {
    "ollama/qwen3.5:latest": "Qwen 3.5 (local · fast · free)",
    "ollama/llama3.1:8b": "Llama 3.1 8B (local · free)",
    "mistral/mistral-small-latest": "Mistral Small (cloud)",
    "mistral/mistral-medium-latest": "Mistral Medium (cloud)",
    "mistral/mistral-large-latest": "Mistral Large (cloud)",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash (cloud)",
}

# Default model when a request names none, and a last-resort fallback tried if the
# chosen model errors (a local model is ideal here -- free and always available).
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", LLM_MODELS[0] if LLM_MODELS else "").strip()
CHAT_FALLBACK_MODEL = os.getenv("CHAT_FALLBACK_MODEL", "ollama/qwen3.5:latest").strip()

# "Thinking" Ollama models (qwen3.x, ...) emit a long reasoning trace before the
# answer -- slow, and often empty within a token budget. Off by default for RAG.
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").strip().lower() in ("1", "true", "yes")

# Max answer length (tokens) for a chat generation. Long answers (markdown
# lists of theses with citations) overflow 512 and get cut off mid-sentence,
# so leave headroom; OLLAMA_NUM_CTX comfortably covers prompt + answer.
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1536"))

# Context window Ollama allocates for the chat model. Without this Ollama uses
# the model's maximum (262K for qwen3.5 = ~14 GB of KV cache on top of the
# weights). 16K comfortably fits the ~6.5K-token RAG prompt plus history.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

# How long Ollama keeps models in memory after a request. Default -1 = forever,
# so no user pays the multi-second model reload after an idle period. Accepts
# seconds (int) or an Ollama duration string like "2h".
_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "-1").strip()
try:
    OLLAMA_KEEP_ALIVE: int | str = int(_keep_alive)
except ValueError:
    OLLAMA_KEEP_ALIVE = _keep_alive

# --- API ----------------------------------------------------------------
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173").strip()


def require_pinecone_key() -> str:
    """Return the Pinecone API key or raise a clear error if it is unset."""
    return _require("PINECONE_API_KEY")
