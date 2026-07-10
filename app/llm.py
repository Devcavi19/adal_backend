"""LiteLLM wrappers.

For Task 3 this exposes embeddings only (local Ollama model via LiteLLM). The
chat completion wrapper + provider fallback lands in a later task.
"""

import litellm

from . import config


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of texts with the configured model, preserving order.

    Calls the local Ollama model through LiteLLM in batches. Returns one vector
    per input text, each of length ``config.EMBED_DIM``.
    """
    batch_size = batch_size or config.EMBED_BATCH
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = litellm.embedding(
            model=config.EMBEDDING_MODEL,
            input=batch,
            api_base=config.OLLAMA_BASE_URL,
        )
        # litellm returns data ordered by input index; sort defensively.
        rows = sorted(resp.data, key=lambda r: r["index"])
        vectors.extend(row["embedding"] for row in rows)

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
