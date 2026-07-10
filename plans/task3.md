# Task 3 — Pinecone vector database

Source: `docs/backend-setup-plan.md` — Phase 2, Build order item 3.

## Goal
Create the Pinecone index, upsert the sample thesis, and sanity-check retrieval.

## 3.1 Create the index
Serverless, region `us-east-1` (free starter tier):
```python
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name="adal-theses",
    dimension=1536,                # must match the embedding model
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
)
```

Decisions:
- **One index, one namespace per corpus** (`theses` now; `handbooks`, `faqs` later). Namespaces are free and scope queries.
- **Dimension is locked to the embedding model** — `text-embedding-3-small` = 1536. Switching embedding models later requires re-embedding everything, so pick once.
- Alternative that removes one API dependency: create the index **with integrated embeddings** (`pc.create_index_for_model(..., embed={"model": "llama-text-embed-v2"})`) and upsert raw text — Pinecone embeds server-side, and queries embed automatically. Recommended for the minimum moving parts.

## 3.2 Upsert (`ingest/run_ingest.py`)
```python
# pseudocode of the CLI flow
for jsonl_file in data/processed/*.jsonl:
    chunks = load(jsonl_file)
    vectors = embed_batch([c["text"] for c in chunks])   # litellm.embedding(), batches of 100
    index.upsert(
        vectors=[(c["id"], v, metadata(c)) for c, v in zip(chunks, vectors)],
        namespace="theses",
    )
```
- Batch upserts (100 per call).
- Make the script **idempotent**: deterministic IDs (`slug__section__n`) mean re-running an updated thesis overwrites its old chunks.
- Verify with `index.describe_index_stats()` — vector count should match chunk count.

## 3.3 Query pattern (used later by the API)
```python
results = index.query(
    vector=embed(query), top_k=5, namespace="theses",
    include_metadata=True,
    # optional: filter={"program": "BSIT"} or {"year": {"$gte": 2022}}
)
```
Keep `top_k=5` and drop matches below ~0.35 cosine score — fewer, better chunks = faster + cheaper LLM calls and fewer hallucinated citations.

## Steps
- [ ] Create the `adal-theses` Pinecone index (serverless, `us-east-1`, dimension matching the embedding model).
- [ ] Decide: self-managed embeddings (LiteLLM `text-embedding-3-small`) vs. Pinecone integrated embeddings.
- [ ] Implement `ingest/run_ingest.py` to upsert the sample thesis's chunks in batches of 100.
- [ ] Run the ingest CLI against the sample thesis's processed `.jsonl`.
- [ ] Verify vector count via `index.describe_index_stats()` matches the chunk count.
- [ ] Sanity-query it, e.g. `"what is the methodology of <thesis>?"`, and confirm it returns Chapter 3 chunks.

## Done when
The sample thesis's chunks are searchable in Pinecone and a methodology-style query returns the correct chapter.
