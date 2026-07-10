"""Ingest CLI.

Phase 1 (this task): extract -> clean -> chunk each PDF into an inspectable
``data/processed/<slug>.jsonl`` -- one JSON object per chunk -- so cleaning
quality can be eyeballed before paying for embeddings.

Phase 2 (Task 3): ``--upsert`` embeds the processed ``.jsonl`` chunks with the
local Ollama model (via LiteLLM) and upserts them into Pinecone. The offline
text stage above stays unchanged.

Usage:
    python -m ingest.run_ingest data/raw                       # every PDF -> JSONL
    python -m ingest.run_ingest "data/raw/one thesis.pdf"      # a single PDF -> JSONL
    python -m ingest.run_ingest "data/raw/x.pdf" --title "…" --program BSIT --year 2023
    python -m ingest.run_ingest --upsert                       # JSONL -> Pinecone
"""

import argparse
import csv
import json
import pathlib
import re
import sys

from .chunk import chunk_sections, segment_sections
from .clean import dehyphenate, normalize_text, strip_leader_lines, strip_repeating_lines
from .extract import extract_pages, looks_scanned

RAW_DIR = pathlib.Path("data/raw")
PROCESSED_DIR = pathlib.Path("data/processed")
MANIFEST_PATH = pathlib.Path("data/manifest.csv")

# fields written per chunk (internal keys are dropped)
_OUTPUT_FIELDS = ("id", "text", "title", "authors", "year", "program", "section", "source_file")


def slugify(name: str) -> str:
    """Filename stem -> id slug: lowercase, alnum runs joined by hyphens."""
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s) or "thesis"


def _manifest_lookup(source_file: str) -> dict | None:
    """Best-effort read of a manifest row (the fail-fast loader is Task 6)."""
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("source_file") == source_file:
                return row
    return None


def build_meta(pdf_path: pathlib.Path, overrides: dict) -> dict:
    """Assemble chunk metadata: manifest row if present, else derive from the
    filename, with any CLI overrides applied last."""
    source_file = pdf_path.name
    stem = pdf_path.stem
    meta: dict = {
        "slug": slugify(stem),
        "source_file": source_file,
        "title": stem,
        "authors": "",
        "year": None,
        "program": "",
    }

    row = _manifest_lookup(source_file)
    if row:
        year = row.get("year", "")
        meta.update(
            title=row.get("title") or meta["title"],
            authors=row.get("authors", ""),
            year=int(year) if str(year).isdigit() else None,
            program=row.get("program", ""),
        )
    else:
        # derive a best guess so a single sample runs before the manifest exists
        m = re.search(r"\b(19|20)\d{2}\b", stem)
        if m:
            meta["year"] = int(m.group())
        prog = re.match(r"\s*([A-Za-z]{2,6}(?:[- ]?\d[A-Za-z]?)?)", stem)
        if prog:
            meta["program"] = prog.group(1).upper().replace(" ", "")

    for key, val in overrides.items():
        if val is not None:
            meta[key] = val
    return meta


def process_pdf(pdf_path: pathlib.Path, meta: dict) -> list[dict]:
    """Run the full extract -> clean -> chunk pipeline for one PDF."""
    pages = extract_pages(str(pdf_path))
    if looks_scanned(pages):
        raise ValueError(
            f"{pdf_path.name}: extracted almost no text -- looks scanned. "
            f"OCR it once (`ocrmypdf in.pdf out.pdf`) and re-run."
        )
    pages = strip_repeating_lines(pages)                    # rule 1
    text = normalize_text("\n".join(pages))                 # rule 4
    text = strip_leader_lines(text)                         # drop TOC dot-leaders
    text = dehyphenate(text)                                # rule 2 (line joins)
    sections = segment_sections(text)                       # rules 3/5
    return chunk_sections(sections, meta)                   # rules 2/5/6 + windows


def write_jsonl(chunks: list[dict], slug: str) -> pathlib.Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{slug}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({k: c[k] for k in _OUTPUT_FIELDS}, ensure_ascii=False) + "\n")
    return out


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def upsert_processed(jsonl_dir: pathlib.Path, namespace: str) -> int:
    """Embed + upsert every processed ``.jsonl`` in ``jsonl_dir`` into Pinecone."""
    from app import vectorstore  # lazy: pinecone/litellm only needed for this stage

    files = sorted(jsonl_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files in {jsonl_dir}/ -- run the PDF stage first.")

    vectorstore.ensure_index()
    total = 0
    for jf in files:
        chunks = _load_jsonl(jf)
        n = vectorstore.upsert_chunks(chunks, namespace=namespace)
        total += n
        print(f"  upsert  {jf.name}  ->  {n} vectors  (ns={namespace})")

    stats = vectorstore.get_index().describe_index_stats()
    print(f"Done. {total} vectors upserted. Index stats: {stats}")
    return total


def _iter_pdfs(target: pathlib.Path):
    if target.is_dir():
        yield from sorted(target.glob("*.pdf"))
    elif target.suffix.lower() == ".pdf":
        yield target
    else:
        sys.exit(f"Not a PDF or directory: {target}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract -> clean -> chunk thesis PDFs to JSONL.")
    ap.add_argument("target", nargs="?", default=str(RAW_DIR),
                    help="a PDF file or a directory of PDFs (default: data/raw)")
    ap.add_argument("--title")
    ap.add_argument("--authors")
    ap.add_argument("--program")
    ap.add_argument("--year", type=int)
    ap.add_argument("--upsert", action="store_true",
                    help="embed processed JSONL and upsert to Pinecone (skips PDF stage)")
    ap.add_argument("--jsonl-dir", default=str(PROCESSED_DIR),
                    help="directory of processed .jsonl to upsert (default: data/processed)")
    ap.add_argument("--namespace", default="theses",
                    help="Pinecone namespace to upsert into (default: theses)")
    args = ap.parse_args(argv)

    if args.upsert:
        upsert_processed(pathlib.Path(args.jsonl_dir), args.namespace)
        return 0

    overrides = {
        "title": args.title, "authors": args.authors,
        "program": args.program, "year": args.year,
    }

    total = 0
    for pdf in _iter_pdfs(pathlib.Path(args.target)):
        meta = build_meta(pdf, overrides)
        try:
            chunks = process_pdf(pdf, meta)
        except ValueError as e:
            print(f"  skip  {e}", file=sys.stderr)
            continue
        out = write_jsonl(chunks, meta["slug"])
        total += len(chunks)
        print(f"  ok    {pdf.name}  ->  {out}  ({len(chunks)} chunks)")

    print(f"Done. {total} chunks written to {PROCESSED_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
