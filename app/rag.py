"""retrieve -> prompt -> LLM answer"""

import re

from . import llm, vectorstore

SYSTEM_PROMPT = """You are Adal, the academic librarian assistant of Camarines Sur Polytechnic Colleges (CSPC) in Nabua, Camarines Sur.

Grounding rules:
- For questions about CSPC research (theses/capstones), answer ONLY from the thesis
  context provided with the question. Never invent titles, authors, findings, or figures.
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

CONDENSE_PROMPT = """Rewrite the user's latest message as a short standalone search query \
for a thesis database, resolving references like "it", "the second one", or "those studies" \
from the conversation into the actual topics or titles they point to. Keep any program \
names/codes and years. If the message is already self-contained (or is a greeting), return \
it unchanged. Output ONLY the query text, nothing else."""

# Per-turn cap when embedding history into the condense prompt: previous answers
# can be long, and only their gist is needed to resolve references.
_CONDENSE_TURN_CHARS = 500
_CONDENSE_NUM_PREDICT = 64

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


def condense_query(message: str, history: list[dict]) -> str:
    """``message`` rewritten as a standalone retrieval query, given ``history``.

    Follow-ups like "tell me more about the second one" embed as meaningless
    vectors and match nothing, so a small LLM call resolves them against the
    conversation first. First turns (no history) and any condensation failure
    fall back to the original message -- retrieval must never break over this.
    """
    if not history:
        return message
    transcript = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')[:_CONDENSE_TURN_CHARS]}"
        for m in history
    )
    prompt = f"Conversation so far:\n{transcript}\n\nLatest user message: {message}"
    try:
        reply = llm.chat(
            [{"role": "system", "content": CONDENSE_PROMPT},
             {"role": "user", "content": prompt}],
            num_predict=_CONDENSE_NUM_PREDICT,
        )
    except Exception:  # noqa: BLE001 -- condensation is best-effort
        return message
    # First non-empty line only: drops any trailing chatter or the truncation
    # notice appended when the 64-token cap is hit.
    lines = [ln.strip().strip('"') for ln in reply.splitlines()]
    condensed = next((ln for ln in lines if ln), "")
    return condensed or message


def retrieve(message: str, history: list[dict] | None = None) -> list[dict]:
    """The context chunks for ``message`` (condensed against ``history`` and
    scoped by any inferred filter)."""
    query = condense_query(message, history or [])
    return vectorstore.query(query, metadata_filter=infer_filter(query))


def build_messages(message: str, history: list[dict], chunks: list[dict]) -> list[dict]:
    """Chat messages with the retrieved context appended to the *user* message.

    Keeping the system prompt and history byte-identical across turns lets
    Ollama reuse its KV cache for that prefix; only the new user message (with
    its per-query context) is prefilled. Context in the system prompt would
    invalidate the cache from token zero on every turn.
    """
    user = message + format_context(chunks)
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user}]


def answer(message: str, history: list[dict], *, model: str | None = None) -> dict:
    chunks = retrieve(message, history)
    reply = llm.chat(build_messages(message, history, chunks), model=model)
    return {"text": reply, "sources": dedupe_sources(chunks)}


def answer_stream(message: str, history: list[dict], *, chunks: list[dict],
                  model: str | None = None):
    """Yield answer pieces for pre-retrieved ``chunks`` (see ``retrieve``).

    Retrieval stays with the caller so it can send the sources to the client
    before generation starts.
    """
    yield from llm.chat_stream(build_messages(message, history, chunks), model=model)
