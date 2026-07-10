"""Section-aware chunking.

Split cleaned thesis text on section headings first, then window long sections
into ~800-token / 100-token-overlap chunks so a chunk never straddles two
sections (e.g. "Methodology" and "Results"). Each chunk is prefixed with a
one-line context header and carries the thesis metadata that lives alongside the
vector in Pinecone.
"""

import re

from . import clean

# ~800-token windows with 100-token overlap. We approximate tokens from words
# (~1.3 tokens/word for English prose) to avoid a tokenizer dependency in the
# offline pipeline; embedding/upsert stays byte-for-byte deterministic.
TOKENS_PER_WORD = 1.3
TARGET_TOKENS = 800
OVERLAP_TOKENS = 100

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# "Chapter 3", "CHAPTER III", "Chapter Three" -- optionally with the title
# trailing on the same line ("CHAPTER I  THE PROBLEM").
_CHAPTER_RE = re.compile(
    r"^\s*chapter\s+(\d{1,2}|[ivx]{1,4}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\b\.?\s*(.*)$",
    re.I,
)


def _chapter_number(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _ROMAN.get(token) or _WORDS.get(token)


def approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * TOKENS_PER_WORD))


def _looks_like_title(line: str) -> bool:
    """A chapter-title line: short and dominated by uppercase letters, e.g.
    'REVIEW OF RELATED LITERATURE AND STUDIES'."""
    s = line.strip()
    if not s or len(s) > 90:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(c.isupper() for c in letters)
    return upper / len(letters) > 0.7


def segment_sections(text: str) -> list[dict]:
    """Split cleaned text into ordered sections.

    Returns dicts ``{"kind", "num", "label", "body"}`` where ``kind`` is one of
    ``frontmatter`` (pre-abstract, dropped), ``abstract``, ``chapter``,
    ``references`` (kept as a single chunk) or ``appendix`` (dropped). Everything
    from the first appendix heading to EOF is folded into one ``appendix``
    section, since trailing appendices are raw questionnaires we discard.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current = {"kind": "frontmatter", "num": None, "label": "Front matter", "body": []}
    in_appendix = False
    seen_real_chapter = False  # a chapter section carrying real prose

    # An appendix heading is the terminal back-matter boundary only once a real
    # chapter body has appeared. Table-of-contents entries ("CHAPTER 1: …",
    # "Appendix A") are near-empty stubs, and the abstract/front matter precede
    # the TOC -- so only a substantial *chapter* body flips this flag, which the
    # real chapters (after the TOC) always do before the real appendices.
    REAL_BODY_CHARS = 1500

    def flush():
        nonlocal seen_real_chapter
        current["body"] = "\n".join(current["body"]).strip()
        if current["kind"] == "chapter" and len(current["body"]) >= REAL_BODY_CHARS:
            seen_real_chapter = True
        sections.append(current)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_appendix:
            i += 1
            continue

        chap = _CHAPTER_RE.match(stripped)
        if chap and _chapter_number(chap.group(1)) is not None:
            flush()
            num = _chapter_number(chap.group(1))
            title = chap.group(2).strip()
            if not title:  # title sits on the following non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and _looks_like_title(lines[j]):
                    title = lines[j].strip()
                    i = j
            title = title.strip(" :-–—")  # inline "CHAPTER 1: INTRODUCTION"
            label = f"Chapter {num}: {title.title()}" if title else f"Chapter {num}"
            current = {"kind": "chapter", "num": num, "label": label, "body": []}
            i += 1
            continue

        if clean.is_abstract_heading(stripped):
            flush()
            current = {"kind": "abstract", "num": None, "label": "Abstract", "body": []}
            i += 1
            continue

        if clean.is_references_heading(stripped):
            flush()
            current = {"kind": "references", "num": None, "label": "References", "body": []}
            i += 1
            continue

        if clean.is_appendix_heading(stripped):
            # Before any real chapter body this is a stray TOC entry, not the
            # back-matter appendix: drop the line but keep scanning for chapters.
            if not seen_real_chapter:
                i += 1
                continue
            flush()
            current = {"kind": "appendix", "num": None, "label": "Appendix", "body": []}
            in_appendix = True
            i += 1
            continue

        if clean.is_boilerplate_heading(stripped):
            flush()
            current = {"kind": "frontmatter", "num": None, "label": stripped.title(), "body": []}
            i += 1
            continue

        current["body"].append(line)
        i += 1

    flush()
    return sections


def _section_tag(section: dict) -> str:
    if section["kind"] == "chapter":
        return f"ch{section['num']}"
    return section["kind"]  # abstract | references


def window_text(text: str, target_tokens: int = TARGET_TOKENS,
                overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """Greedily pack paragraphs into ~target_tokens windows with overlap.

    Paragraph boundaries are respected; an oversized single paragraph is
    hard-split on word boundaries so no window blows past the target."""
    paragraphs = [p for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return []

    # hard-split any paragraph that alone exceeds the target
    units: list[str] = []
    for para in paragraphs:
        if approx_tokens(para) <= target_tokens:
            units.append(para)
            continue
        words = para.split()
        step = max(1, int(target_tokens / TOKENS_PER_WORD))
        for k in range(0, len(words), step):
            units.append(" ".join(words[k:k + step]))

    windows: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for unit in units:
        ut = approx_tokens(unit)
        if cur and cur_tok + ut > target_tokens:
            windows.append("\n\n".join(cur))
            # carry trailing units as overlap for the next window
            carry: list[str] = []
            carry_tok = 0
            for prev in reversed(cur):
                pt = approx_tokens(prev)
                if carry_tok + pt > overlap_tokens:
                    break
                carry.insert(0, prev)
                carry_tok += pt
            cur = carry
            cur_tok = carry_tok
        cur.append(unit)
        cur_tok += ut
    if cur:
        windows.append("\n\n".join(cur))
    return windows


def _context_header(meta: dict, section_label: str) -> str:
    title = meta.get("title") or meta.get("source_file", "Untitled")
    program = meta.get("program", "")
    year = meta.get("year", "")
    tag = " ".join(str(x) for x in (program, year) if x)
    prefix = f'[Thesis: "{title}"'
    if tag:
        prefix += f", {tag}"
    return f"{prefix} — {section_label}]"


def chunk_sections(sections: list[dict], meta: dict) -> list[dict]:
    """Turn kept sections into chunk records with context headers, deterministic
    IDs and junk/duplicate filtering."""
    slug = meta["slug"]
    chunks: list[dict] = []
    seen: set[str] = set()

    for section in sections:
        kind = section["kind"]
        if kind in ("frontmatter", "appendix") or not section["body"]:
            continue  # rule 3: drop boilerplate & appendices

        body = clean.reflow(section["body"])  # rule 2: paragraph reflow
        if not body:
            continue

        # rule 5: keep the reference list as a single chunk, not windowed prose
        pieces = [body] if kind == "references" else window_text(body)
        tag = _section_tag(section)

        for piece in pieces:
            if clean.is_junk_chunk(piece):  # rule 6
                continue
            key = re.sub(r"\s+", " ", piece).strip().lower()
            if key in seen:  # rule 6: dedupe within the same thesis
                continue
            seen.add(key)

            n = sum(1 for c in chunks if c["_tag"] == tag)
            header = _context_header(meta, section["label"])
            record = {
                "id": f"{slug}__{tag}__{n:03d}",
                "text": f"{header}\n\n{piece}",
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year"),
                "program": meta.get("program", ""),
                "section": section["label"],
                "source_file": meta.get("source_file", ""),
                "_tag": tag,  # internal, stripped before writing
            }
            chunks.append(record)

    for c in chunks:
        c.pop("_tag", None)
    return chunks
