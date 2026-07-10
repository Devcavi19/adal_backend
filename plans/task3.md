# Task 3 — Pinecone vector database

Source: `docs/backend-setup-plan.md` — Phase 2, Build order item 3.

## Goal
Create the Pinecone index, upsert the sample thesis, and sanity-check retrieval.

**Embedding decision (locked):** self-managed embeddings via a **local Ollama
model** — `bge-m3` (**1024 dims, 8192-token context**), called through LiteLLM
(`ollama/bge-m3`). No per-token cost, no external embedding key.
> Note: `mxbai-embed-large` (also 1024-dim) was the first pick but its 512-token
> context can't fit the ~800-token thesis chunks — every chunk overflowed. `bge-m3`
> keeps the 1024 dimension (index unchanged) with a context long enough for our chunks.

Trade-off: Ollama must be running wherever embeddings happen — at ingest **and**
at query time (the live API), since the query must be embedded by the same model.

## 3.1 Create the index
Standard serverless index, region `us-east-1` (free starter tier). Handled by
`app/vectorstore.ensure_index()`:
```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name="adal-theses",
    dimension=1024,                # locked to mxbai-embed-large
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

Decisions:
- **One index, one namespace per corpus** (`theses` now; `handbooks`, `faqs` later). Namespaces are free and scope queries.
- **Dimension is locked to the embedding model** — `mxbai-embed-large` = 1024. Switching embedding models later requires recreating the index and re-embedding everything, so pick once.
- **Standard (self-managed) index**, not integrated — we supply vectors embedded locally. (Pinecone's integrated `create_index_for_model` was considered but rejected in favour of a fully local, no-cost embedding path.)

## 3.2 Upsert (`ingest/run_ingest.py --upsert`)
```
python -m ingest.run_ingest --upsert            # data/processed/*.jsonl -> Pinecone
```
Flow (`upsert_processed` -> `app.vectorstore.upsert_chunks`):
```python
for jsonl_file in data/processed/*.jsonl:
    chunks  = load(jsonl_file)
    vectors = embed_texts([c["text"] for c in chunks])   # litellm ollama, batched
    index.upsert(
        vectors=[{"id": c["id"], "values": v, "metadata": meta(c)} for c, v in ...],
        namespace="theses",
    )
```
- Batch upserts (100 per call); chunk text is stored in metadata so a match returns the passage directly.
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
- [x] Decide embeddings: local Ollama `bge-m3` (1024) via native `/api/embed`, self-managed vectors. (Switched from LiteLLM: its sync `embedding()` reuses a module-level async client whose event loop closes between calls, so multi-batch upserts failed intermittently with "Event loop is closed".)
- [x] Create the `adal-theses` Pinecone index (serverless `us-east-1`, dimension 1024) — `ensure_index()` (now guards against a dimension/model mismatch).
- [x] Implement `ingest/run_ingest.py --upsert` to upsert the sample thesis's chunks in batches of 100.
- [x] Run the ingest CLI against the sample thesis's processed `.jsonl` — `AB ELS 4A_Rhetorical Analysis…` → 36 chunks.
- [x] Verify vector count via `index.describe_index_stats()` matches the chunk count — 36 chunks = 36 vectors (ns=`theses`).
- [x] Sanity-query it — `"What is the research methodology and design of the study?"` → top hit (0.557) is the "Chapter 2: Research Methods" chapter (this thesis puts methodology in Ch. 2).

## Done when
The sample thesis's chunks are searchable in Pinecone and a methodology-style query returns the correct chapter.
