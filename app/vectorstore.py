"""Pinecone client helpers.

A standard (self-managed) serverless index: we supply the vectors ourselves
(embedded locally via Ollama), so the index dimension is locked to the
embedding model's width -- see ``config.EMBED_DIM``.
"""

import time

from pinecone import Pinecone, ServerlessSpec

from . import config

# Metadata fields carried alongside each vector. ``text`` is included so a query
# returns the chunk content directly (needed by the RAG answer step).
_META_FIELDS = ("text", "title", "authors", "year", "program", "section", "source_file")

_pc: Pinecone | None = None


def client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=config.require_pinecone_key())
    return _pc


def ensure_index(*, wait: bool = True) -> None:
    """Create the configured index if it does not exist yet (idempotent).

    If the index already exists but its dimension differs from the configured
    ``EMBED_DIM`` (e.g. the embedding model changed), raise instead of silently
    upserting mismatched vectors -- fixing it means recreating the index.
    """
    pc = client()
    existing = {ix["name"] for ix in pc.list_indexes()}
    if config.PINECONE_INDEX in existing:
        dim = pc.describe_index(config.PINECONE_INDEX).dimension
        if dim != config.EMBED_DIM:
            raise RuntimeError(
                f"Index '{config.PINECONE_INDEX}' has dimension {dim} but "
                f"EMBED_DIM is {config.EMBED_DIM} (model {config.EMBEDDING_MODEL}). "
                f"Delete and recreate the index, or align EMBED_DIM to the model."
            )
        return
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=config.EMBED_DIM,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    if wait:
        while not pc.describe_index(config.PINECONE_INDEX).status.get("ready", False):
            time.sleep(1)


def get_index():
    return client().Index(config.PINECONE_INDEX)


def _metadata(chunk: dict) -> dict:
    """Pinecone metadata: keep known fields, drop nulls (Pinecone rejects None)."""
    return {k: chunk[k] for k in _META_FIELDS if chunk.get(k) not in (None, "")}


def upsert_chunks(chunks: list[dict], *, namespace: str | None = None,
                  batch_size: int | None = None, embedder=None) -> int:
    """Embed and upsert chunk dicts. Deterministic ``id`` makes this idempotent
    -- re-running an updated thesis overwrites its old vectors. Returns count."""
    from .llm import embed_texts  # local import: avoids importing litellm for query-only use

    namespace = namespace or config.DEFAULT_NAMESPACE
    batch_size = batch_size or config.UPSERT_BATCH
    embedder = embedder or embed_texts
    index = get_index()

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder([c["text"] for c in batch])
        index.upsert(
            vectors=[
                {"id": c["id"], "values": v, "metadata": _metadata(c)}
                for c, v in zip(batch, vectors)
            ],
            namespace=namespace,
        )
        total += len(batch)
    return total


def query(text: str, *, top_k: int = 5, namespace: str | None = None,
          min_score: float = 0.35, metadata_filter: dict | None = None) -> list[dict]:
    """Embed ``text`` and return the top matches above ``min_score``."""
    from .llm import embed_query

    namespace = namespace or config.DEFAULT_NAMESPACE
    res = get_index().query(
        vector=embed_query(text),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=metadata_filter,
    )
    return [
        {"id": m["id"], "score": m["score"], **(m.get("metadata") or {})}
        for m in res.get("matches", [])
        if m["score"] >= min_score
    ]
