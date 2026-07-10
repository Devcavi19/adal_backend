"""PDF -> raw text (PyMuPDF).

Pure extraction only: no cleaning, no chunking. `sort=True` reconstructs a
sensible reading order on two-column thesis layouts.
"""

import pymupdf  # PyMuPDF


def extract_pages(pdf_path: str) -> list[str]:
    """Return the text of each page, in reading order."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text", sort=True) for page in doc]


def looks_scanned(pages: list[str], min_chars_per_page: float = 100.0) -> bool:
    """Heuristic: an image-only (scanned) PDF extracts to near-empty text.

    Such a PDF must be OCR'd once as a manual pre-step
    (`ocrmypdf input.pdf output.pdf`) and re-extracted -- OCR is deliberately
    kept out of the pipeline.
    """
    if not pages:
        return True
    total = sum(len(p.strip()) for p in pages)
    return total / len(pages) < min_chars_per_page
