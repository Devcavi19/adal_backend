"""Task 7 eval harness: run representative questions through the RAG pipeline.

Retrieval-only by default (fast; shows scores, inferred filters, and matched
titles so precision/recall can be judged). ``--answers`` also generates the
full LLM answer per question; ``--model`` picks the chat model.

    python -m scripts.eval_rag
    python -m scripts.eval_rag --answers
    python -m scripts.eval_rag --answers --model mistral/mistral-small-latest
"""

import argparse

from app import rag, vectorstore

# (label, question) -- covers in-scope, scoped (program/year), out-of-scope,
# general librarian skills, and smalltalk.
QUESTIONS = [
    ("in-scope",       "What theses are about library management systems?"),
    ("in-scope",       "What did CSPC students find about time management and academic performance of nursing students?"),
    ("in-scope",       "Are there studies about taro-based food products?"),
    ("scoped program", "What BSIT capstone projects used SMS notifications?"),
    ("scoped prog+yr", "Show me nursing theses from 2024 about maternal healthcare."),
    ("scoped year",    "What capstone projects about web systems were made in 2023?"),
    ("scoped range",   "What entrepreneurship studies were done between 2023 and 2024?"),
    ("out-of-scope",   "Do you have theses about quantum computing?"),
    ("out-of-scope",   "What research is there about marine biology at CSPC?"),
    ("librarian",      "How do I cite a journal article in APA 7th edition?"),
    ("smalltalk",      "Hello! Who are you?"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", action="store_true", help="also generate LLM answers")
    ap.add_argument("--model", default=None, help="chat model id (default: config default)")
    args = ap.parse_args()

    for label, q in QUESTIONS:
        flt = rag.infer_filter(q)
        chunks = vectorstore.query(q, metadata_filter=flt)
        print(f"\n=== [{label}] {q}")
        print(f"    filter: {flt}")
        if not chunks:
            print("    (no chunks above threshold)")
        for c in chunks:
            print(f"    {c['score']:.3f}  {c.get('program', '?'):9} {c.get('year', '?')}  {c.get('title', '')[:70]}")
        if args.answers:
            result = rag.answer(q, [], model=args.model)
            print(f"    --- answer ---\n{result['text']}\n    --- sources: {[s['title'][:50] for s in result['sources']]}")


if __name__ == "__main__":
    main()
