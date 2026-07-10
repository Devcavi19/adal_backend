"""LiteLLM wrappers.

For Task 3 this exposes embeddings only. Embeddings call Ollama's native
``/api/embed`` endpoint directly (via ``requests``) rather than going through
LiteLLM: LiteLLM's sync ``embedding()`` reuses a module-level async client whose
event loop gets closed between calls, so a multi-batch upsert fails intermittently
with "Event loop is closed". The native endpoint batches and is more reliable.
The chat completion wrapper + provider fallback (LiteLLM) lands in a later task.
"""

import requests

from . import config


def _ollama_url() -> str:
    """Full /api/embed URL, tolerating a base with or without an http scheme."""
    base = config.OLLAMA_BASE_URL
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base.rstrip("/") + "/api/embed"


def _model_name() -> str:
    """Bare Ollama model name (strip the optional ``ollama/`` prefix)."""
    name = config.EMBEDDING_MODEL
    return name[len("ollama/"):] if name.startswith("ollama/") else name


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of texts with the configured model, preserving order.

    Calls the local Ollama model in batches. Returns one vector per input text,
    each of length ``config.EMBED_DIM``.
    """
    batch_size = batch_size or config.EMBED_BATCH
    url = _ollama_url()
    model = _model_name()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = requests.post(url, json={"model": model, "input": batch}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None or len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama embed returned {len(embeddings or [])} vectors for "
                f"{len(batch)} inputs (model {model}). Response: {str(data)[:200]}"
            )
        vectors.extend(embeddings)

    if vectors and len(vectors[0]) != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding width {len(vectors[0])} != configured EMBED_DIM "
            f"{config.EMBED_DIM} for model {config.EMBEDDING_MODEL}. "
            f"Fix EMBED_DIM (and the Pinecone index dimension) to match."
        )
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
