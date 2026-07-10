# Adal Backend — Setup Plan & Guide

A lightweight RAG (Retrieval-Augmented Generation) backend for **Adal**, CSPC's AI academic librarian. It turns undergraduate thesis PDFs into a searchable knowledge base (Pinecone) and exposes a small HTTP API that the existing React frontend consumes through its one swap point: `src/services/chatService.ts`.

```
┌─────────────────────┐        ┌──────────────────────────────────────────┐
│  React frontend      │        │  Backend (FastAPI)                       │
│  chatService.ts      │ HTTP   │                                          │
│  sendMessage(id,text)├───────▶│  POST /api/chat                          │
│                      │        │   1. embed query                         │
└─────────────────────┘        │   2. Pinecone top-k search               │
                               │   3. build prompt w/ retrieved chunks    │
        offline, one-time      │   4. LLM answer (any provider)           │
┌─────────────────────┐        └───────────────┬──────────────────────────┘
│  Thesis PDFs         │  ingest pipeline       │
│  (raw files)         ├───────────────────────▶│  Pinecone (vector DB)
└─────────────────────┘  extract→clean→chunk    │  namespace: "theses"
                          →embed→upsert         └──────────────────────────
```

**Stack (chosen for "lightweight"):**

| Concern              | Choice                                                              | Why                                                                                                    |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Language / framework | Python 3.12 + **FastAPI** + uvicorn                                 | Tiny, async, auto OpenAPI docs; Python owns the PDF-processing ecosystem                               |
| PDF extraction       | **PyMuPDF** (`pymupdf`)                                             | Fastest pure extraction, no Java/system deps                                                           |
| Vector DB            | **Pinecone Serverless**                                             | Free starter tier, zero ops, integrated embeddings available                                           |
| LLM access           | **LiteLLM** (library, not proxy)                                    | One `completion()` call for OpenAI / Anthropic / Gemini / Groq / Ollama — swap models via env var only |
| Embeddings           | `text-embedding-3-small` (or Pinecone-hosted `llama-text-embed-v2`) | Cheap, 1536-dim, good retrieval quality                                                                |

Deliberately **not** used: LangChain/LlamaIndex (heavy, unnecessary for one pipeline), a database (conversations stay client-side for now), Docker (optional later). Total runtime deps: ~6 packages.

---

## Repository layout

Create a sibling repo (e.g. `adal_backend/`) — keep the frontend repo pure UI:

```
adal_backend/
├── app/
│   ├── main.py            # FastAPI app, CORS, routes
│   ├── config.py          # env-driven settings
│   ├── rag.py             # retrieve → prompt → LLM answer
│   ├── llm.py             # LiteLLM wrapper + provider fallback
│   └── vectorstore.py     # Pinecone client helpers
├── ingest/
│   ├── extract.py         # PDF → raw text (PyMuPDF)
│   ├── clean.py           # cleaning rules (below)
│   ├── chunk.py           # section-aware chunking
│   └── run_ingest.py      # CLI: python -m ingest.run_ingest ./pdfs
├── data/
│   ├── raw/               # original PDFs (gitignored)
│   └── processed/         # cleaned .jsonl per thesis (inspectable!)
├── .env.example
├── requirements.txt
└── README.md
```

`requirements.txt`:

```txt
fastapi
uvicorn[standard]
pinecone
litellm
pymupdf
python-dotenv
```

---

## Phase 1 — Thesis PDF → clean knowledge-base data

Goal: convert messy scanned/exported thesis PDFs into small, self-contained, metadata-rich text chunks so retrieval is **fast** (fewer, denser vectors) and answers are **grounded** (each chunk cites its thesis).

### 1.1 Extract (`ingest/extract.py`)

```python
import pymupdf  # PyMuPDF

def extract_pages(pdf_path: str) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text", sort=True) for page in doc]
```

- `sort=True` fixes reading order on two-column layouts.
- If a thesis is a **scanned image PDF** (extraction returns near-empty text), OCR it once with `ocrmypdf input.pdf output.pdf` and re-extract. Don't build OCR into the pipeline; treat it as a pre-step.

### 1.2 Clean (`ingest/clean.py`)

Apply in order — each rule targets a real thesis-PDF artifact:

1. **Strip repeating headers/footers** — lines that appear on >50 % of pages (running titles, "CAMARINES SUR POLYTECHNIC COLLEGES", page numbers). Detect by frequency, not position.
2. **Fix hyphenation** — join `word-\nbreaks` → `wordbreaks`; then collapse single newlines inside paragraphs, keep blank lines as paragraph breaks.
3. **Drop boilerplate sections** — title page, approval sheet, certification, acknowledgment, table of contents, list of figures/tables, appendices of raw questionnaires. Keep: abstract, chapters 1–5, conclusions, recommendations. Match on standard thesis headings (`ABSTRACT`, `CHAPTER I`, `APPROVAL SHEET`, …).
4. **Normalize whitespace & unicode** — NFKC normalize, collapse runs of spaces, remove form feeds and stray control chars.
5. **Handle the references list** — don't chunk it as prose; either drop it or store it as one `references` chunk per thesis (useful when students ask "what sources did X use?").
6. **Filter junk chunks** — after chunking, drop chunks that are <100 chars, mostly digits/symbols (tables that extracted badly), or duplicated across the same thesis.

Write the result to `data/processed/<thesis-slug>.jsonl` — one JSON object per chunk — so you can **eyeball the cleaning quality before paying for embeddings**.

### 1.3 Chunk (`ingest/chunk.py`)

- **Split on section headings first** (chapters, numbered headings), then split long sections into ~**800-token windows with 100-token overlap**. Section-aware beats fixed-size: a chunk never straddles "Methodology" and "Results".
- Prefix each chunk with a one-line context header so it makes sense in isolation:

  ```
  [Thesis: "IoT-Based Fish Feeder", BSIT 2023 — Chapter 3: Methodology]
  <chunk text…>
  ```

- Chunk metadata (stored in Pinecone alongside the vector):

```json
{
  "id": "iot-fish-feeder-2023__ch3__004",
  "text": "…",
  "title": "IoT-Based Fish Feeder",
  "authors": "Dela Cruz, J.; Santos, M.",
  "year": 2023,
  "program": "BSIT",
  "section": "Chapter 3: Methodology",
  "source_file": "iot-fish-feeder-2023.pdf"
}
```

Capture title/authors/year/program once per PDF in a small `manifest.csv` (filename → metadata) rather than trying to parse title pages automatically — 10 minutes of manual work per batch, far more reliable. Format below.

### 1.4 Metadata manifest (`manifest.csv`)

One row per PDF, keyed by filename. Lives at `data/manifest.csv`, committed to git (it's the human-curated part of the pipeline).

```csv
source_file,title,authors,year,program,adviser,keywords
iot-fish-feeder-2023.pdf,"IoT-Based Fish Feeder with Water Quality Monitoring","Dela Cruz, Juan; Santos, Maria",2023,BSIT,"Engr. R. Reyes","iot; aquaculture; sensors"
enrollment-system-2022.pdf,"Web-Based Enrollment System for CSPC","Aquino, Pedro",2022,BSIS,"Prof. L. Cruz","enrollment; web app"
grade-predictor-2024.pdf,"Student Grade Prediction Using Machine Learning","Ramos, Ana; Villanueva, Ben; Ocampo, Cy",2024,BSCS,,"machine learning; education"
```

Column rules:

| Column | Required | Notes |
|---|---|---|
| `source_file` | yes | Exact filename in `data/raw/` — this is the join key |
| `title` | yes | Official thesis title; quote it (titles contain commas) |
| `authors` | yes | **Semicolon-separated** (`Last, First; Last, First`) so commas inside names don't break the CSV |
| `year` | yes | Defense/publication year, plain integer |
| `program` | yes | Short code (`BSIT`, `BSCS`, `BSIS`, …) — used for Pinecone metadata filters |
| `adviser` | no | Leave empty if unknown (note the empty field in row 3) |
| `keywords` | no | Semicolon-separated; useful extra retrieval signal in the chunk header |

Conventions:

- The chunk-ID **slug is derived from `source_file`** (filename minus `.pdf`), so name PDFs meaningfully up front: `<short-title>-<year>.pdf`, lowercase, hyphens, no spaces. Renaming a PDF later changes its IDs and orphans old vectors.
- Easiest way to author it: Google Sheets / Excel with these headers, then **File → Download → CSV (UTF-8)**. The spreadsheet handles the quoting for you.
- Any field containing a comma must be wrapped in double quotes — automatic if you export from a spreadsheet; only a concern if editing by hand.

`run_ingest.py` loads it with the stdlib and **fails fast** on gaps, so a typo never becomes an unlabeled vector in Pinecone:

```python
import csv, pathlib, sys

def load_manifest(path="data/manifest.csv") -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:   # -sig eats Excel's BOM
        rows = {r["source_file"]: r for r in csv.DictReader(f)}

    pdfs = {p.name for p in pathlib.Path("data/raw").glob("*.pdf")}
    missing = pdfs - rows.keys()          # PDFs with no metadata row
    stale   = rows.keys() - pdfs          # rows pointing at deleted/renamed PDFs
    if missing or stale:
        sys.exit(f"manifest.csv mismatch — missing: {missing or '∅'}, stale: {stale or '∅'}")

    for name, r in rows.items():
        if not (r["title"] and r["authors"] and r["year"].isdigit() and r["program"]):
            sys.exit(f"manifest.csv: bad row for {name}")
    return rows
```

During ingestion each chunk merges its manifest row into the Pinecone metadata (§1.3) and the context header, e.g. `[Thesis: "IoT-Based Fish Feeder…", BSIT 2023 — Chapter 3: Methodology]`.

---

## Phase 2 — Pinecone vector database

### 2.1 Create the index

Console (or script) — **Serverless**, region `us-east-1` (free starter tier):

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

- **One index, one namespace per corpus** (`theses` now; `handbooks`, `faqs` later). Namespaces are free and let you scope queries.
- **Dimension is locked to the embedding model** — `text-embedding-3-small` = 1536. If you switch embedding models later you must re-embed everything, so pick once.
- Alternative that removes one API dependency: create the index **with integrated embeddings** (`pc.create_index_for_model(..., embed={"model": "llama-text-embed-v2"})`) and upsert raw text — Pinecone embeds server-side, and queries embed automatically too. Recommended if you want the absolute minimum moving parts.

### 2.2 Upsert (`ingest/run_ingest.py`)

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

- Batch upserts (100 per call), and make the script **idempotent**: deterministic IDs (`slug__section__n`) mean re-running an updated thesis overwrites its old chunks.
- Verify with `index.describe_index_stats()` — vector count should match chunk count.

### 2.3 Query pattern (used by the API)

```python
results = index.query(
    vector=embed(query), top_k=5, namespace="theses",
    include_metadata=True,
    # optional: filter={"program": "BSIT"} or {"year": {"$gte": 2022}}
)
```

Keep `top_k=5` and drop matches below ~0.35 cosine score — fewer, better chunks = faster + cheaper LLM calls and fewer hallucinated citations.

---

## Phase 3 — Backend API (FastAPI)

### 3.1 Endpoints

| Method | Path          | Purpose                                                |
| ------ | ------------- | ------------------------------------------------------ |
| `POST` | `/api/chat`   | Main RAG chat endpoint (matches the frontend contract) |
| `GET`  | `/api/health` | Liveness + which LLM provider/model is active          |

Request/response shaped to drop into `chatService.ts` with minimal change:

```jsonc
// POST /api/chat
{ "conversationId": "abc", "message": "How do I cite in IEEE?",
  "history": [{ "role": "user", "content": "…" }] }   // client sends last ~10 turns

// 200
{ "text": "markdown answer…",
  "sources": [{ "title": "…", "year": 2023, "section": "…" }] }
```

The frontend already renders markdown (`react-markdown`), so the backend returns markdown text; `sources` lets the UI later show citation chips without a contract change.

### 3.2 Core flow (`app/rag.py`)

```python
async def answer(message: str, history: list[Msg]) -> ChatResponse:
    chunks = retrieve(message)                     # Pinecone top-k (skip if greeting/smalltalk)
    system = SYSTEM_PROMPT + format_context(chunks)
    reply = await llm_complete(system, history, message)
    return ChatResponse(text=reply, sources=dedupe_sources(chunks))
```

System prompt essentials: Adal's persona (CSPC academic librarian), _answer only from the provided thesis context when the question is about local research; say so when the context doesn't cover it_, cite thesis titles inline, and general librarian skills (citation formatting, search strategy) may use model knowledge.

### 3.3 Multi-provider LLM layer (`app/llm.py`)

LiteLLM makes every provider look like one API — the model is **just a string from env**:

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

### 3.4 App skeleton (`app/main.py`)

```python
app = FastAPI(title="Adal API")
app.add_middleware(CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["POST", "GET"], allow_headers=["*"])
```

Run: `uvicorn app.main:app --reload --port 8000`. Interactive docs at `/docs` for free.

Keep-it-lightweight rules: no ORM/DB, no auth beyond a simple `X-API-Key` check (add when deploying publicly), no background workers — ingestion is an offline CLI, the API only reads.

### 3.5 Environment (`.env.example`)

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

---

## Phase 4 — Frontend integration

Only `src/services/chatService.ts` changes (its signature is the contract the UI depends on):

```ts
export interface ChatReply {
  text: string;
  sources?: { title: string; year: number; section: string }[];
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function sendMessage(
  conversationId: string,
  text: string,
): Promise<ChatReply> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversationId, message: text }),
  });
  if (!res.ok) throw new Error(`Adal API error: ${res.status}`);
  return res.json();
}
```

Add `VITE_API_URL` to the frontend `.env`. Later upgrades that don't break this contract: SSE streaming (`text/event-stream`) for token-by-token replies, and rendering `sources` as citation chips under assistant messages.

---

## Phase 5 — Deployment (later, still lightweight)

- **Render / Railway / Fly.io** free-hobby tier runs `uvicorn` directly from the repo — no Docker needed (both auto-detect `requirements.txt`).
- Pinecone serverless + a cheap model (Gemini Flash / GPT-5 mini) keeps a thesis-defense-scale demo effectively free.
- Set `FRONTEND_ORIGIN` to the deployed frontend URL; add the `X-API-Key` header check.

---

## Build order checklist

- [ ] **1. Scaffold** `adal_backend/` repo, venv, `requirements.txt`, `.env`
- [ ] **2. Ingest pipeline**: extract → clean → chunk one sample thesis; inspect `data/processed/*.jsonl` by hand until chunks read cleanly
- [ ] **3. Pinecone**: create index, upsert the sample thesis, sanity-query it (`"what is the methodology of <thesis>?"` should return Chapter 3 chunks)
- [ ] **4. API**: `/api/chat` with retrieval + LiteLLM; test via `/docs`
- [ ] **5. Wire frontend**: swap `chatService.ts`, run both dev servers, chat end-to-end
- [ ] **6. Full corpus**: fill `manifest.csv`, ingest all thesis PDFs
- [ ] **7. Tune**: system prompt, `top_k`/score threshold, provider fallback order
- [ ] **8. Deploy** backend + point `VITE_API_URL` at it
