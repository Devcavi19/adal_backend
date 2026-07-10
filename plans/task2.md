# Task 2 — Ingest pipeline (extract → clean → chunk)

Source: `docs/backend-setup-plan.md` — Phase 1, Build order item 2.

## Goal
Convert one sample thesis PDF into clean, chunked `.jsonl` and hand-inspect quality before touching Pinecone.

## 2.1 Extract (`ingest/extract.py`)
```python
import pymupdf  # PyMuPDF

def extract_pages(pdf_path: str) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text", sort=True) for page in doc]
```
- `sort=True` fixes reading order on two-column layouts.
- If a thesis is a **scanned image PDF** (extraction returns near-empty text), OCR it once with `ocrmypdf input.pdf output.pdf` and re-extract. Don't build OCR into the pipeline — treat it as a manual pre-step.

## 2.2 Clean (`ingest/clean.py`)
Apply in order — each rule targets a real thesis-PDF artifact:
1. **Strip repeating headers/footers** — lines appearing on >50% of pages (running titles, college name, page numbers). Detect by frequency, not position.
2. **Fix hyphenation** — join `word-\nbreaks` → `wordbreaks`; collapse single newlines inside paragraphs, keep blank lines as paragraph breaks.
3. **Drop boilerplate sections** — title page, approval sheet, certification, acknowledgment, table of contents, list of figures/tables, raw questionnaire appendices. Keep: abstract, chapters 1–5, conclusions, recommendations. Match on standard headings (`ABSTRACT`, `CHAPTER I`, `APPROVAL SHEET`, …).
4. **Normalize whitespace & unicode** — NFKC normalize, collapse space runs, remove form feeds/control chars.
5. **Handle the references list** — don't chunk as prose; drop it or store as one `references` chunk per thesis.
6. **Filter junk chunks** — after chunking, drop chunks <100 chars, mostly digits/symbols, or duplicated within the same thesis.

Write output to `data/processed/<thesis-slug>.jsonl` — one JSON object per chunk — so cleaning quality can be eyeballed before paying for embeddings.

## 2.3 Chunk (`ingest/chunk.py`)
- Split on section headings first (chapters, numbered headings), then split long sections into **~800-token windows with 100-token overlap**. Section-aware beats fixed-size — a chunk should never straddle "Methodology" and "Results".
- Prefix each chunk with a one-line context header:
  ```
  [Thesis: "IoT-Based Fish Feeder", BSIT 2023 — Chapter 3: Methodology]
  <chunk text…>
  ```
- Chunk metadata shape (stored in Pinecone alongside the vector):
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
- Capture title/authors/year/program once per PDF via `manifest.csv` (see Task 6) rather than parsing title pages automatically.

## Steps
- [ ] Implement `extract_pages()` in `ingest/extract.py`.
- [ ] Implement the 6 cleaning rules in `ingest/clean.py`.
- [ ] Implement section-aware chunking with overlap in `ingest/chunk.py`.
- [ ] Run the pipeline on **one sample thesis PDF**.
- [ ] Manually inspect `data/processed/<slug>.jsonl` until chunks read cleanly (no boilerplate, no broken hyphenation, sensible chunk boundaries).

## Done when
One sample thesis produces a clean, readable `.jsonl` file with correctly bounded chunks and context headers.
