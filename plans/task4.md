# Task 4 — Backend API (FastAPI)

Source: `docs/backend-setup-plan.md` — Phase 3, Build order item 4.

## Goal
Build `/api/chat` with retrieval + multi-provider LLM completion, and verify it via `/docs`.

## 4.1 Endpoints
| Method | Path          | Purpose                                                |
| ------ | ------------- | ------------------------------------------------------ |
| `POST` | `/api/chat`   | Main RAG chat endpoint (matches the frontend contract) |
| `GET`  | `/api/health` | Liveness + which LLM provider/model is active          |

Request/response shape (matches `chatService.ts` with minimal change):
```jsonc
// POST /api/chat
{ "conversationId": "abc", "message": "How do I cite in IEEE?",
  "history": [{ "role": "user", "content": "…" }] }   // client sends last ~10 turns

// 200
{ "text": "markdown answer…",
  "sources": [{ "title": "…", "year": 2023, "section": "…" }] }
```
The frontend renders markdown (`react-markdown`), so return markdown text; `sources` lets the UI later show citation chips without a contract change.

## 4.2 Core flow (`app/rag.py`)
```python
async def answer(message: str, history: list[Msg]) -> ChatResponse:
    chunks = retrieve(message)                     # Pinecone top-k (skip if greeting/smalltalk)
    system = SYSTEM_PROMPT + format_context(chunks)
    reply = await llm_complete(system, history, message)
    return ChatResponse(text=reply, sources=dedupe_sources(chunks))
```
System prompt essentials: Adal's persona (CSPC academic librarian), answer only from the provided thesis context when the question is about local research, say so when the context doesn't cover it, cite thesis titles inline, and general librarian skills (citation formatting, search strategy) may use model knowledge.

## 4.3 Multi-provider LLM layer (`app/llm.py`)
```python
import litellm

MODEL_CHAIN = os.getenv("LLM_MODELS", "gemini/gemini-2.5-flash,gpt-5-mini,claude-haiku-4-5").split(",")

async def llm_complete(system: str, history, user: str) -> str:
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": user}]
    for model in MODEL_CHAIN:                      # fallback chain
        try:
            resp = await litellm.acompletion(model=model, messages=messages,
                                             temperature=0.3, max_tokens=1024)
            return resp.choices[0].message.content
        except Exception:
            continue
    raise HTTPException(503, "All LLM providers unavailable")
```
- Set only the API keys you have (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`); LiteLLM picks them up by name.
- Switching providers/models = editing `LLM_MODELS` in `.env`. No code change. Local dev without any key: `ollama/llama3.1`.
- The fallback chain doubles as a cost lever: put the cheapest capable model first.

## 4.4 App skeleton (`app/main.py`)
```python
app = FastAPI(title="Adal API")
app.add_middleware(CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["POST", "GET"], allow_headers=["*"])
```
Run: `uvicorn app.main:app --reload --port 8000`. Interactive docs at `/docs` for free.

Keep-it-lightweight rules: no ORM/DB, no auth beyond a simple `X-API-Key` check (add when deploying publicly), no background workers — ingestion is an offline CLI, the API only reads.

## 4.5 Environment (`.env.example`)
```bash
PINECONE_API_KEY=
PINECONE_INDEX=adal-theses
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODELS=gemini/gemini-2.5-flash,gpt-5-mini   # first = primary, rest = fallbacks
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
FRONTEND_ORIGIN=http://localhost:5173
```

## Steps
- [ ] Implement `app/config.py` (env-driven settings, load `.env`).
- [ ] Implement `app/vectorstore.py` (Pinecone client + query helper, reuses Task 3's query pattern).
- [ ] Implement `app/llm.py` with the `MODEL_CHAIN` fallback logic above.
- [ ] Implement `app/rag.py` (`answer()` combining retrieval + system prompt + LLM call).
- [ ] Implement `app/main.py`: FastAPI app, CORS middleware, `POST /api/chat`, `GET /api/health`.
- [ ] Write the system prompt (Adal persona, grounding rules, citation instructions).
- [ ] Run `uvicorn app.main:app --reload --port 8000` and exercise `/api/chat` via `/docs`.

## Done when
`/api/chat` returns a grounded, cited answer for a question about the sample thesis, and `/api/health` reports the active model.
