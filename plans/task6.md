# Task 6 — Full corpus ingestion

Source: `docs/backend-setup-plan.md` — §1.4 Metadata manifest, Build order item 6.

## Goal
Fill in `manifest.csv` for every thesis PDF and ingest the full corpus (not just the one sample from Task 2/3).

## Metadata manifest (`data/manifest.csv`)
One row per PDF, keyed by filename. Committed to git — it's the human-curated part of the pipeline.

```csv
source_file,title,authors,year,program,adviser,keywords
iot-fish-feeder-2023.pdf,"IoT-Based Fish Feeder with Water Quality Monitoring","Dela Cruz, Juan; Santos, Maria",2023,BSIT,"Engr. R. Reyes","iot; aquaculture; sensors"
enrollment-system-2022.pdf,"Web-Based Enrollment System for CSPC","Aquino, Pedro",2022,BSIS,"Prof. L. Cruz","enrollment; web app"
grade-predictor-2024.pdf,"Student Grade Prediction Using Machine Learning","Ramos, Ana; Villanueva, Ben; Ocampo, Cy",2024,BSCS,,"machine learning; education"
```

### Column rules
| Column | Required | Notes |
|---|---|---|
| `source_file` | yes | Exact filename in `data/raw/` — this is the join key |
| `title` | yes | Official thesis title; quote it (titles contain commas) |
| `authors` | yes | **Semicolon-separated** (`Last, First; Last, First`) so commas inside names don't break the CSV |
| `year` | yes | Defense/publication year, plain integer |
| `program` | yes | Short code (`BSIT`, `BSCS`, `BSIS`, …) — used for Pinecone metadata filters |
| `adviser` | no | Leave empty if unknown |
| `keywords` | no | Semicolon-separated; useful extra retrieval signal in the chunk header |

### Conventions
- The chunk-ID **slug is derived from `source_file`** (filename minus `.pdf`), so name PDFs meaningfully up front: `<short-title>-<year>.pdf`, lowercase, hyphens, no spaces. Renaming a PDF later changes its IDs and orphans old vectors.
- Easiest way to author it: Google Sheets/Excel with these headers, then **File → Download → CSV (UTF-8)** — the spreadsheet handles quoting for you.
- Any field containing a comma must be wrapped in double quotes — automatic from a spreadsheet export; only a concern if editing by hand.

### Fail-fast loader (already part of `run_ingest.py`)
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
During ingestion each chunk merges its manifest row into the Pinecone metadata and the context header, e.g. `[Thesis: "IoT-Based Fish Feeder…", BSIT 2023 — Chapter 3: Methodology]`.

## Steps
- [ ] Collect all remaining thesis PDFs into `data/raw/`, renamed to `<short-title>-<year>.pdf` (lowercase, hyphens).
- [ ] Author `data/manifest.csv` with one row per PDF (title, authors, year, program required; adviser/keywords optional).
- [ ] Run the full extract → clean → chunk pipeline (Task 2) across every PDF in `data/raw/`.
- [ ] Spot-check a few `data/processed/*.jsonl` files for cleaning quality.
- [ ] Run `run_ingest.py` to embed and upsert the full corpus into Pinecone (Task 3), confirming the manifest loader's fail-fast checks pass.
- [ ] Verify `index.describe_index_stats()` reflects the full corpus's vector count.

## Done when
Every thesis PDF in `data/raw/` has a corresponding manifest row and is searchable in Pinecone.
