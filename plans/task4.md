# Task 4 — Backend API (FastAPI)

Source: `docs/backend-setup-plan.md` — Phase 3, Build order item 4.

## Goal
Build `/api/chat` with retrieval + local/cloud LLM completion, and verify it via `/docs`.

## 4.1 Endpoints
| Method | Path          | Purpose                                                |
| ------ | ------------- | ------------------------------------------------------ |
| `POST` | `/api/chat`   | Main RAG chat endpoint (matches the frontend contract) |
| `GET`  | `/api/health` | Liveness + which chat model is active                  |
| `GET`  | `/api/models` | Selectable chat models for a frontend dropdown          |

Request/response shape (matches `chatService.ts` with minimal change):
```jsonc
// POST /api/chat
{ "conversationId": "abc", "message": "How do I cite in IEEE?",
  "history": [{ "role": "user", "content": "…" }],   // client sends last ~10 turns
  "model": "ollama/qwen3.5:latest" }                 // optional; must be in LLM_MODELS, defaults to DEFAULT_CHAT_MODEL

// 200
{ "text": "markdown answer…",
  "sources": [{ "title": "…", "year": 2023, "section": "…" }] }
```
The frontend renders markdown (`react-markdown`), so return markdown text; `sources` lets the UI later show citation chips without a contract change. `model` is additive — the frontend can ignore it and always get `DEFAULT_CHAT_MODEL`.

`GET /api/models` returns `app.llm.available_models()` — `[{"id": "...", "label": "..."}]` — so the frontend can render a dropdown without hardcoding model ids.

## 4.2 Core flow (`app/rag.py`) — not yet implemented (stub only)
```python
def answer(message: str, history: list[dict], *, model: str | None = None) -> ChatResponse:
    chunks = vectorstore.query(message)             # Pinecone top-k, drops sub-0.35-score matches
    system = SYSTEM_PROMPT + format_context(chunks)
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": message}]
    reply = llm.chat(messages, model=model)          # raises ChatError if every attempt fails
    return ChatResponse(text=reply, sources=dedupe_sources(chunks))
```
System prompt essentials: Adal's persona (CSPC academic librarian), answer only from the provided thesis context when the question is about local research, say so when the context doesn't cover it, cite thesis titles inline, and general librarian skills (citation formatting, search strategy) may use model knowledge.

Note: `llm.chat` is currently sync (`requests`/LiteLLM sync calls), so `answer()` and the `/api/chat` route can stay sync too — no need to force `async def` just for FastAPI's sake.

## 4.3 Chat LLM layer (`app/llm.py`) — implemented, diverged from the original plan
Reality ended up different from the original "one `MODEL_CHAIN`, try every provider in order" design:

- **`LLM_MODELS` is a user-selectable menu, not a blind fallback chain.** The frontend calls `GET /api/models` (backed by `available_models()`) and lets the user pick one; `/api/chat` validates the pick against `config.LLM_MODELS` and rejects anything else, so the client can never trigger an unlisted (possibly costly) model.
- **Local models bypass LiteLLM entirely.** `ollama/*` ids go straight to Ollama's native `/api/chat` via `requests` (thinking disabled — `OLLAMA_THINK=false` — so a "thinking" model like `qwen3.5` returns the answer instead of burning the token budget on a reasoning trace). Only cloud ids (`mistral/...`, `gemini/...`, etc.) go through LiteLLM's sync `completion()`.
- **Fallback is single-hop, not chain-of-everything.** If the chosen model errors, `chat()` retries once against `CHAT_FALLBACK_MODEL` (a local Ollama model — free and always available) and raises `ChatError` only if both attempts fail. It does not walk the whole `LLM_MODELS` list.
- **Embeddings are a separate, non-LiteLLM path.** `embed_texts`/`embed_query` call Ollama's native `/api/embed` directly with `requests`, because LiteLLM's sync `embedding()` reuses a module-level async client whose event loop closes between calls, causing intermittent "Event loop is closed" errors on multi-batch upserts. See [[task3]] for the full embedding-model decision.

```python
# app/llm.py (actual shape)
def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]: ...
def embed_query(text: str) -> list[float]: ...
def available_models() -> list[dict]: ...   # [{"id", "label"}] for the frontend
def chat(messages: list[dict], *, model: str | None = None) -> str: ...  # validates + single fallback
```

- Set only the API keys you have (`MISTRAL_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); LiteLLM picks them up by name for the cloud entries in `LLM_MODELS`.
- Switching the menu = editing `LLM_MODELS` (and optionally `MODEL_LABELS`, `DEFAULT_CHAT_MODEL`, `CHAT_FALLBACK_MODEL`) in `.env`/`config.py`. No code change needed for a same-shape id.
- Local dev needs zero cloud keys: `ollama/qwen3.5:latest` (or any pulled Ollama model) works standalone as long as Ollama is running.

## 4.4 App skeleton (`app/main.py`) — not yet implemented (stub only)
```python
app = FastAPI(title="Adal API")
app.add_middleware(CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_methods=["POST", "GET"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "default_model": config.DEFAULT_CHAT_MODEL}

@app.get("/api/models")
def models():
    return llm.available_models()

@app.post("/api/chat")
def chat(req: ChatRequest):
    return rag.answer(req.message, req.history, model=req.model)
```
Run: `uvicorn app.main:app --reload --port 8000`. Interactive docs at `/docs` for free.

Keep-it-lightweight rules: no ORM/DB, no auth beyond a simple `X-API-Key` check (add when deploying publicly), no background workers — ingestion is an offline CLI, the API only reads.

## 4.5 Environment (`.env.example`) — implemented, superset of the original plan
```bash
PINECONE_API_KEY=
PINECONE_INDEX=adal-theses
# Local Ollama embedding model (native /api/embed). Index dimension locks to it.
EMBEDDING_MODEL=ollama/bge-m3
EMBED_DIM=1024
OLLAMA_BASE_URL=localhost:11434
# Chat models the frontend may offer, in menu order (first = default). The API
# validates the user's pick against this list. Mix local (ollama/<name>) + cloud.
LLM_MODELS=ollama/qwen3.5:latest,mistral/mistral-small-latest
CHAT_FALLBACK_MODEL=ollama/qwen3.5:latest   # tried if the chosen model errors (local = free/always up)
OLLAMA_THINK=false        # keep off: thinking models waste the token budget for RAG
LLM_NUM_PREDICT=512       # max answer length (tokens)
MISTRAL_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
FRONTEND_ORIGIN=http://localhost:5173
```
`config.py` also derives `DEFAULT_CHAT_MODEL` (first entry of `LLM_MODELS` unless overridden) and holds `MODEL_LABELS`, the human-readable names for `available_models()`.

## Steps
- [x] Implement `app/config.py` (env-driven settings, load `.env`) — done; also covers embedding + chat-model settings needed by Task 3 and this task.
- [x] Implement `app/vectorstore.py` (Pinecone client + query helper) — done in [[task3]], reused as-is here (`vectorstore.query`).
- [x] Implement `app/llm.py` — done, but reshaped per §4.3: native-Ollama chat/embeddings + LiteLLM only for cloud models, model-menu validation, single-hop fallback instead of a full chain.
- [x] Implement `app/rag.py` (`answer()` combining retrieval + system prompt + `llm.chat()`) — done.
- [x] Implement `app/main.py`: FastAPI app, CORS middleware, `POST /api/chat`, `GET /api/health`, `GET /api/models` — done. Also adds `config.FRONTEND_ORIGIN` (was referenced by the plan's `.env.example` but missing from `config.py`) and a 400 for a `model` not in `LLM_MODELS`.
- [x] Write the system prompt (Adal persona, grounding rules, citation instructions) — in `rag.SYSTEM_PROMPT`.
- [x] Run `uvicorn app.main:app --reload --port 8000` and exercise `/api/chat` via `/docs` — verified via curl: `/api/health`, `/api/models`, a grounded+cited thesis question, a general-knowledge citation-formatting question, and rejection of an unlisted `model`.

## Done when
`/api/chat` returns a grounded, cited answer for a question about the sample thesis, `/api/health` reports the active model, and `/api/models` lists the selectable chat models.
