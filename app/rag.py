"""retrieve -> prompt -> LLM answer"""

from . import llm, vectorstore

SYSTEM_PROMPT = """You are Adal, the academic librarian assistant for the College of Saint Peter College (CSPC).

Grounding rules:
- When the question is about local research (CSPC theses/capstones), answer ONLY using the
  thesis context provided below. Do not invent findings, authors, or figures.
- If the provided context does not cover the question, say so plainly instead of guessing.
- Cite thesis titles inline when you use them (e.g. "According to *Title* (Year), ...").
- For general librarian skills that are not about local research -- citation formatting
  (APA, IEEE, MLA, etc.), search strategy, database navigation, study tips -- you may answer
  from your own knowledge, since no thesis context is needed for these.
- Keep answers concise and written in markdown."""


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
    chunks = vectorstore.query(message)
    system = SYSTEM_PROMPT + format_context(chunks)
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": message}]
    reply = llm.chat(messages, model=model)
    return {"text": reply, "sources": dedupe_sources(chunks)}
