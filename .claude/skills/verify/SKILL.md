---
name: verify
description: Build/launch/drive recipe for verifying changes to the Adal RAG backend (FastAPI + Ollama + Pinecone).
---

# Verifying the Adal backend

Prereqs (all usually already running): Ollama at `localhost:11434` with
`qwen3.5:latest` + `bge-m3` pulled, a populated Pinecone index (`.env` has the
key), Python deps from `requirements.txt` importable.

## Launch

```bash
python -m uvicorn app.main:app --port 8321   # any free port; ~5s to ready
curl -s http://localhost:8321/api/health      # {"status":"ok",...}
```

Startup fires a background warmup thread (Pinecone TLS + Ollama model loads);
give it ~10s before timing anything.

## Drive

```bash
# Non-stream chat (JSON: {text, sources})
curl -s -X POST localhost:8321/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"What BSIT theses cover machine learning?","history":[]}'

# Streaming chat (SSE: `event: sources`, then `data:` token pieces, `event: done`;
# `event: error` if all models fail)
curl -sN -X POST localhost:8321/api/chat/stream -H 'Content-Type: application/json' \
  -d '{"message":"...","history":[]}'
```

`history` is `[{role, content}, ...]` of prior plain-text turns. Model pick via
`"model": "..."` must be in `LLM_MODELS` (else 400). Avoid driving the
`mistral/*` models — they spend cloud API credits; local qwen3.5 is the default.

## Gotchas

- First LLM answer after an Ollama model reload takes ~10s extra; check
  `curl -s localhost:11434/api/ps` to see what's loaded (and with which
  `context_length`/VRAM — should be 16384 / ~9 GB for qwen3.5, per
  `OLLAMA_NUM_CTX`).
- A "good" full answer takes 5-10s at ~70 tok/s; SSE first token ~4-5s.
- Retrieval floor quirk: even smalltalk ("hello!") can return one thesis in
  `sources` if some chunk scores ≥ 0.50; the LLM still replies appropriately.
