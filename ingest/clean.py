"""Cleaning rules for extracted thesis text.

Each function targets a real artifact of exported/scanned thesis PDFs. They are
applied in the order documented in `docs/backend-setup-plan.md` (§1.2):

1. strip repeating headers/footers   -> strip_repeating_lines(pages)
2. fix hyphenation                    -> dehyphenate()  +  reflow()
3. drop boilerplate sections          -> heading predicates, applied in chunk.py
4. normalize whitespace & unicode     -> normalize_text()
5. handle the references list         -> is_references_heading() (kept as one chunk)
6. filter junk chunks                 -> is_junk_chunk()  (post-chunking)

Rules 1/2/4 run on raw page text; rules 3/5 are heading-driven and are applied
while sectioning in `chunk.py`; rule 6 runs on the finished chunks.
"""

import re
import unicodedata

# --- Rule 1: repeating headers / footers ------------------------------------

_PAGE_NUM_RE = re.compile(r"^[\s\-–—]*\d{1,4}[\s\-–—]*$")


def _norm_line(line: str) -> str:
    """Normalize a line for frequency comparison (digits -> #, so page numbers
    embedded in running footers still collide across pages)."""
    s = " ".join(line.split()).lower()
    return re.sub(r"\d+", "#", s)


def strip_repeating_lines(pages: list[str], threshold: float = 0.5) -> list[str]:
    """Remove lines that recur on more than `threshold` of pages.

    Running titles ("CAMARINES SUR POLYTECHNIC COLLEGES"), college names and
    page numbers repeat on most pages; detect them by frequency, not position.
    """
    if not pages:
        return pages

    from collections import Counter

    freq: Counter[str] = Counter()
    for page in pages:
        seen = {_norm_line(ln) for ln in page.splitlines() if ln.strip()}
        freq.update(seen)

    cutoff = max(2, int(len(pages) * threshold))
    repeating = {key for key, n in freq.items() if n > cutoff and key}

    cleaned: list[str] = []
    for page in pages:
        kept = []
        for ln in page.splitlines():
            if not ln.strip():
                kept.append(ln)
                continue
            if _PAGE_NUM_RE.match(ln):  # bare page numbers
                continue
            if _norm_line(ln) in repeating:
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned


# --- Rule 4: unicode & whitespace normalization -----------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # keep \t \n \r


def normalize_text(text: str) -> str:
    """NFKC-normalize, drop form-feeds/control chars, collapse intra-line space
    runs. Newlines are preserved -- section detection depends on them."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of spaces/tabs without touching newlines
    text = re.sub(r"[ \t]+", " ", text)
    # strip trailing spaces per line
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    return text


# Table-of-contents / list entries, which must be removed *before* sectioning:
# otherwise the TOC's own "Chapter 1" / "Appendix A" / "References" entries are
# mistaken for real section headings and hijack the section boundaries.
#   _LEADER_RE   -> dotted leaders:      "Chapter 3 .............. 31"
#   _TOC_TAIL_RE -> space-padded page #: "Appendix A            120"
# _TOC_TAIL_RE requires a 2+ space gap before the trailing page number, so a
# real heading line ("Chapter 1", "1.2 Years in Service?") is never matched.
_LEADER_RE = re.compile(r"(?:\.\s*){4,}")
_TOC_TAIL_RE = re.compile(r"\S\s{2,}[ivxlcdm\d]{1,5}\s*$", re.I)


def strip_leader_lines(text: str) -> str:
    return "\n".join(
        ln for ln in text.split("\n")
        if not (_LEADER_RE.search(ln) or _TOC_TAIL_RE.search(ln))
    )


# --- Rule 2: hyphenation & paragraph reflow ---------------------------------

_HYPHEN_BREAK_RE = re.compile(r"(\w)[­\-]\n[ \t]*(\w)")


def dehyphenate(text: str) -> str:
    """Join words split across a line break: ``compre-\\nhension`` -> ``comprehension``.

    Runs before sectioning; only fuses an alphanumeric-hyphen-newline-alphanumeric
    sequence, so genuine hyphenated compounds on one line are untouched.
    """
    prev = None
    while prev != text:  # repeat for consecutive breaks
        prev = text
        text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return text


_SENTENCE_END_RE = re.compile(r"""[.!?;:]["'”’)\]]?$""")


def reflow(text: str) -> str:
    """Collapse single newlines inside a paragraph into spaces while keeping
    blank lines as paragraph breaks. Applied to a section body *after* its
    headings have been detected (headings need line boundaries intact).

    Many theses are double-spaced, so PDF extraction leaves a blank line between
    every *wrapped* line. A second pass re-joins such fragments: a block that is
    long-ish and does not end in sentence punctuation is a soft-wrapped line, so
    it is merged with the next block. Short unpunctuated blocks (sub-headings
    like "Statement of the Problem") are left standing.
    """
    blocks = []
    for para in re.split(r"\n[ \t]*\n+", text):
        joined = " ".join(seg.strip() for seg in para.split("\n") if seg.strip())
        joined = re.sub(r"[ \t]+", " ", joined).strip()
        if joined:
            blocks.append(joined)

    merged: list[str] = []
    for block in blocks:
        if (merged and len(merged[-1]) >= 50
                and not _SENTENCE_END_RE.search(merged[-1])):
            merged[-1] += " " + block
        else:
            merged.append(block)
    return "\n\n".join(merged)


# --- Rule 3: boilerplate vs. keep -------------------------------------------
# Front/back matter to drop entirely. Matched against a heading line.

_BOILERPLATE_RE = re.compile(
    r"^(title page|approval sheet|approval|certification|certificate\b.*|"
    r"acknowledge?ment[s]?|dedication|table of contents|list of (figures|tables|"
    r"appendices|plates|abbreviations)|curriculum vitae|biographical (sketch|data)|"
    r"vita)\s*$",
    re.I,
)

_REFERENCES_RE = re.compile(r"^(references|bibliography|works cited|literature cited)\s*$", re.I)

_APPENDIX_RE = re.compile(r"^appendix\b|^appendices\s*$", re.I)

_ABSTRACT_RE = re.compile(r"^abstract\s*$", re.I)


def is_boilerplate_heading(line: str) -> bool:
    return bool(_BOILERPLATE_RE.match(line.strip()))


def is_references_heading(line: str) -> bool:
    return bool(_REFERENCES_RE.match(line.strip()))


def is_appendix_heading(line: str) -> bool:
    return bool(_APPENDIX_RE.match(line.strip()))


def is_abstract_heading(line: str) -> bool:
    return bool(_ABSTRACT_RE.match(line.strip()))


# --- Rule 6: junk-chunk filter ----------------------------------------------


def is_junk_chunk(text: str, min_chars: int = 100) -> bool:
    """A chunk is junk if it is too short or mostly digits/symbols -- typically a
    table or a figure caption that extracted badly."""
    body = text.strip()
    if len(body) < min_chars:
        return True
    letters = sum(c.isalpha() for c in body)
    if letters / max(len(body), 1) < 0.5:  # <50% letters -> tables/gibberish
        return True
    return False
