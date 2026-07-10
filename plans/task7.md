# Task 7 — Tune retrieval and generation quality

Source: `docs/backend-setup-plan.md` — §2.3, §3.2, §3.3, Build order item 7.

## Goal
With the full corpus live (Task 6) and the API working (Task 4), tune the system prompt, retrieval parameters, and provider fallback order for quality and cost.

## Areas to tune

### System prompt (`app/rag.py`)
Essentials to iterate on:
- Adal's persona (CSPC academic librarian).
- Answer only from the provided thesis context when the question is about local research; explicitly say so when the context doesn't cover it.
- Cite thesis titles inline.
- Allow general librarian skills (citation formatting, search strategy) to use model knowledge beyond the retrieved context.

### `top_k` / score threshold (`app/vectorstore.py`, query pattern)
```python
results = index.query(
    vector=embed(query), top_k=5, namespace="theses",
    include_metadata=True,
    # optional: filter={"program": "BSIT"} or {"year": {"$gte": 2022}}
)
```
- Baseline: `top_k=5`, drop matches below ~0.35 cosine score.
- Fewer, better chunks = faster + cheaper LLM calls and fewer hallucinated citations.
- Consider adding metadata filters (`program`, `year`) when a query implies scope.

### Provider fallback order (`app/llm.py`)
```python
MODEL_CHAIN = os.getenv("LLM_MODELS", "gemini/gemini-2.5-flash,gpt-5-mini,claude-haiku-4-5").split(",")
```
- The fallback chain is a cost lever: put the cheapest capable model first.
- Adjust via `.env`'s `LLM_MODELS` — no code change needed.

## Steps
- [ ] Run a set of representative test questions (including out-of-scope ones) against `/api/chat`.
- [ ] Iterate on the system prompt until grounding behavior and citation style are correct.
- [ ] Experiment with `top_k` and the cosine score cutoff; check retrieval precision vs. recall.
- [ ] Try metadata filters (`program`, `year`) for scoped queries.
- [ ] Reorder `LLM_MODELS` in `.env` to balance cost vs. quality; confirm fallback triggers correctly when the primary model errors.

## Done when
Answers are consistently grounded and correctly cited, retrieval returns relevant chunks with low noise, and the fallback chain has a sensible cost-ordered default.
