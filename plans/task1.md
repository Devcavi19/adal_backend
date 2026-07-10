# Task 1 — Scaffold the backend repo

Source: `docs/backend-setup-plan.md` — Repository layout, Build order item 1.

## Goal
Create the `adal_backend/` repo skeleton with its Python environment and config files.

## Steps
- [x] Create the repo layout:
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
  │   ├── clean.py           # cleaning rules
  │   ├── chunk.py           # section-aware chunking
  │   └── run_ingest.py      # CLI: python -m ingest.run_ingest ./pdfs
  ├── data/
  │   ├── raw/               # original PDFs (gitignored)
  │   └── processed/         # cleaned .jsonl per thesis (inspectable!)
  ├── .env.example
  ├── requirements.txt
  └── README.md
  ```
- [x] Set up a Python 3.12 virtual environment.
- [x] Create `requirements.txt`:
  ```txt
  fastapi
  uvicorn[standard]
  pinecone
  litellm
  pymupdf
  python-dotenv
  ```
- [x] Install dependencies (`pip install -r requirements.txt`).
- [x] Create `.env.example` (see Task 4 for full contents) and a local `.env` (gitignored).
- [x] Add `data/raw/` and `.env` to `.gitignore`.

## Done when
`pip install -r requirements.txt` succeeds and the folder structure above exists (empty stub files are fine).
