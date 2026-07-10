"""retrieve -> prompt -> LLM answer"""

import re

from . import llm, vectorstore

SYSTEM_PROMPT = """You are Adal, the academic librarian assistant of Camarines Sur Polytechnic Colleges (CSPC) in Nabua, Camarines Sur.

Grounding rules:
- For questions about CSPC research (theses/capstones), answer ONLY from the thesis
  context below. Never invent titles, authors, findings, or figures.
- Cite thesis titles inline as *Title* (Year), and only cite titles that appear in the
  context. The context holds short excerpts, not full theses -- report what the excerpts
  say and do not extrapolate beyond them. If an excerpt shows little more than a thesis
  title, list the title without describing the study; never guess what it "likely" covers.
- If the context does not cover the question (or none was found), say so plainly --
  e.g. "I couldn't find CSPC theses on that topic." -- and suggest rephrasing or a
  related topic. Do not substitute general knowledge for missing local research.
- For general librarian help that is NOT about local research -- citation formatting
  (APA, IEEE, MLA, ...), literature-search strategy, database navigation, study tips --
  answer from your own knowledge; no thesis context is needed.
- For greetings or small talk, reply briefly and offer to help find CSPC research.
- Keep answers concise and written in markdown."""

# Program codes present in the corpus metadata (see data/manifest.csv), plus
# unambiguous full names users are likely to type. Used to scope retrieval.
_PROGRAM_CODES = ("ABELS", "BLIS", "BSENTREP", "BSHM", "BSIS", "BSIT", "BSM", "BSN", "BSOA", "BSTM")
_PROGRAM_PHRASES = {
    "nursing": "BSN",
    "midwifery": "BSM",
    "hospitality management": "BSHM",
    "tourism management": "BSTM",
    "information technology": "BSIT",
    "information systems": "BSIS",
    "library and information science": "BLIS",
    "entrepreneurship": "BSENTREP",
    "office administration": "BSOA",
    "english language studies": "ABELS",
}

_YEAR = r"(20\d{2})"


def infer_filter(message: str) -> dict | None:
    """Metadata filter for queries that imply a scope (program and/or year).

    Conservative by design: only explicit program codes/names and year phrases
    trigger a filter; anything ambiguous retrieves unfiltered.
    """
    clauses: dict = {}

    programs = {p for p in _PROGRAM_CODES if re.search(rf"\b{p}\b", message, re.IGNORECASE)}
    lowered = message.lower()
    programs.update(code for phrase, code in _PROGRAM_PHRASES.items() if phrase in lowered)
    if len(programs) == 1:
        clauses["program"] = {"$eq": programs.pop()}
    elif programs:
        clauses["program"] = {"$in": sorted(programs)}

    if m := re.search(rf"\b(?:between|from)\s+{_YEAR}\s+(?:and|to|-)\s*{_YEAR}\b", message, re.IGNORECASE):
        clauses["year"] = {"$gte": int(m.group(1)), "$lte": int(m.group(2))}
    elif m := re.search(rf"\b(?:since|from)\s+{_YEAR}\b|\b{_YEAR}\s+onwards?\b", message, re.IGNORECASE):
        clauses["year"] = {"$gte": int(m.group(1) or m.group(2))}
    elif m := re.search(rf"\bafter\s+{_YEAR}\b", message, re.IGNORECASE):
        clauses["year"] = {"$gt": int(m.group(1))}
    elif m := re.search(rf"\bbefore\s+{_YEAR}\b", message, re.IGNORECASE):
        clauses["year"] = {"$lt": int(m.group(1))}
    elif m := re.search(rf"\b(?:until|up to)\s+{_YEAR}\b", message, re.IGNORECASE):
        clauses["year"] = {"$lte": int(m.group(1))}
    elif years := re.findall(rf"\b{_YEAR}\b", message):
        unique = sorted({int(y) for y in years})
        clauses["year"] = {"$eq": unique[0]} if len(unique) == 1 else {"$in": unique}

    return clauses or None


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "\n\nNo relevant thesis context was found for this question."
    parts = ["\n\nThesis context:"]
    for c in chunks:
        title = c.get("title", "Untitled")
        year = c.get("year", "n.d.")
        section = c.get("section")
        header = f"### {title} ({year})" + (f" -- {section}" if section else "")
        parts.append(f"{header}\n{c.get('text', '')}")
    return "\n\n".join(parts)


def dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c.get("title"), c.get("year"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"title": c.get("title"), "year": c.get("year"), "section": c.get("section")})
    return sources


def answer(message: str, history: list[dict], *, model: str | None = None) -> dict:
    chunks = vectorstore.query(message, metadata_filter=infer_filter(message))
    system = SYSTEM_PROMPT + format_context(chunks)
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": message}]
    reply = llm.chat(messages, model=model)
    return {"text": reply, "sources": dedupe_sources(chunks)}
