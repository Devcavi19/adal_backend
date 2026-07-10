# Adal Backend

Lightweight RAG backend for Adal, CSPC's AI academic librarian. Turns undergraduate thesis PDFs into a searchable Pinecone knowledge base and exposes a small HTTP API consumed by the React frontend.

See `docs/backend-setup-plan.md` for the full design and build order.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```
