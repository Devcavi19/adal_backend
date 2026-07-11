"""Measured evaluation of the RAG pipeline against a labeled ground-truth set.

Three layers, cheapest first:

1. Retrieval metrics (default; no chat LLM needed): hit@k and expected-file
   recall against labeled relevant theses, out-of-scope rejection rate, and
   metadata-filter checks.
2. ``--answers``: also generate the full LLM answer per question.
3. ``--judge``: LLM-as-judge grading of context precision (is each retrieved
   excerpt actually relevant?), faithfulness (is every claim in the answer
   supported by the excerpts?) and answer relevance. Implies ``--answers``.
   The judge defaults to a *different* model than the answerer so the answerer
   never grades its own homework.

    python -m scripts.eval_rag                       # retrieval metrics only
    python -m scripts.eval_rag --judge               # full evaluation
    python -m scripts.eval_rag --judge --json        # + save a timestamped report
    python -m scripts.eval_rag --judge --strict      # non-zero exit below thresholds

``--strict`` makes the run gate-able (CI, pre-deploy): it exits 1 when hit
rate, rejection rate, faithfulness, or context precision fall below the
``STRICT_*`` thresholds.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from app import config, llm, rag, vectorstore

# --- Labeled eval set ---------------------------------------------------------
# ``expected`` lists manifest ``source_file``s a correct retrieval MUST surface.
# It is the *known*-relevant set, not an exhaustive one, so precision is graded
# by the judge rather than computed from these labels. With RETRIEVAL_TOP_K=5
# and RETRIEVAL_MAX_PER_THESIS=2 at most 3 distinct theses fit in one result,
# so keep ``expected`` lists to 3 files or fewer or recall can never reach 1.0.
#
# ``kind``:
#   retrieval    -- must surface the expected files and answer from them
#   out_of_scope -- corpus has nothing; must retrieve nothing and say "not found"
#   general      -- librarian knowledge; correct behavior is NO thesis context
#   smalltalk    -- greeting; correct behavior is NO thesis context
#
# ``expect_filter``: metadata keys ``infer_filter`` must produce for the query.
# ``any_program``: browse-style fallback -- retrieving anything from this
#   program also counts as a hit (used where "all relevant theses" is the
#   whole program's output and listing files would be arbitrary).
CASES = [
    dict(id="time-mgmt-nursing", kind="retrieval",
         question="What did CSPC students find about time management and academic performance of nursing students?",
         expected=["effect-time-management-study-habits-academic-2023.pdf"]),
    dict(id="library-mgmt-systems", kind="retrieval",
         question="What theses are about library management systems?",
         expected=["web-based-iriga-city-public-library-management-2023.pdf"]),
    dict(id="taro-products", kind="retrieval",
         question="Are there studies about taro-based food products?",
         expected=["taro-ice-cream-2023.pdf", "taro-delicacies-2023.pdf",
                   "taro-tart-desiccated-coconut-2024.pdf"]),
    dict(id="bsit-sms", kind="retrieval",
         question="What BSIT capstone projects used SMS notifications?",
         expected=["boardal-web-based-audal-boarding-house-management-2023.pdf",
                   "petcare-rayna-animal-care-clinic-information-2024.pdf"],
         expect_filter={"program"}),
    dict(id="bsn-2024-maternal", kind="retrieval",
         question="Show me nursing theses from 2024 about maternal healthcare.",
         expected=["utilization-maternal-healthcare-services-among-women-2024.pdf"],
         expect_filter={"program", "year"}),
    dict(id="dengue", kind="retrieval",
         question="What research did CSPC students do about dengue?",
         expected=["knowledge-practices-dengue-fever-among-residents-2024.pdf",
                   "implementation-dengue-prevention-control-program-barangay-2023.pdf"]),
    dict(id="bls-competence", kind="retrieval",
         question="Are there studies on basic life support competence?",
         expected=["competence-level-basic-life-support-bls-2023.pdf",
                   "level-competence-basic-life-support-first-2023.pdf"]),
    dict(id="coffee-shops", kind="retrieval",
         question="What studies looked at coffee shops?",
         expected=["customer-satisfaction-coffee-shops-iriga-city-2024.pdf",
                   "social-media-marketing-selected-coffee-shops-2024.pdf",
                   "054-288-4421-loc-128-college-tourism-2023.pdf"]),
    dict(id="heritage-sites", kind="retrieval",
         question="What theses tackle the conservation of heritage sites?",
         expected=["conservation-preservation-heritage-sites-ligao-city-2024.pdf",
                   "safeguarding-heritage-sites-rinconada-area-2023.pdf"]),
    dict(id="nursing-tracer", kind="retrieval",
         question="Is there a tracer study of CSPC nursing graduates?",
         expected=["tracer-study-bachelor-science-nursing-graduates-2023.pdf"]),
    dict(id="shorthand-bsoa", kind="retrieval",
         question="What did office administration researchers find about learning shorthand?",
         expected=["factors-affecting-comprehension-bachelor-science-office-2023.pdf",
                   "exploring-machine-shorthand-learning-experiences-bsoa-2023.pdf",
                   "factors-affecting-academic-performance-machine-shorthand-2022.pdf"],
         expect_filter={"program"}),
    dict(id="chatgpt-language", kind="retrieval",
         question="Do we have research about students using ChatGPT for language learning?",
         expected=["exploring-language-studies-students-perceptions-chatgpt-2024.pdf"]),
    dict(id="taylor-swift", kind="retrieval",
         question="Is there a thesis analyzing a Taylor Swift album?",
         expected=["anatomizing-taylor-swift-s-1989-taylor-2024.pdf"]),
    dict(id="library-noise", kind="retrieval",
         question="What research exists on library noise and its effect on learning?",
         expected=["factors-affecting-library-noise-pollution-students-2025.pdf"]),
    dict(id="blis-employability", kind="retrieval",
         question="How employable are the library and information science graduates of CSPC?",
         expected=["employability-bachelor-library-information-science-graduates-2025.pdf"],
         expect_filter={"program"}),
    dict(id="entrep-2023-2024", kind="retrieval",
         question="What entrepreneurship studies were done between 2023 and 2024?",
         expected=[], any_program="BSENTREP",
         expect_filter={"program", "year"}),
    dict(id="quantum", kind="out_of_scope",
         question="Do you have theses about quantum computing?"),
    dict(id="marine-bio", kind="out_of_scope",
         question="What research is there about marine biology at CSPC?"),
    dict(id="apa", kind="general",
         question="How do I cite a journal article in APA 7th edition?"),
    dict(id="hello", kind="smalltalk",
         question="Hello! Who are you?"),
]

# ``--strict`` gates. Faithfulness is 1.0 on purpose: a single hallucinated
# citation is a trust failure, not a rounding error. Out-of-scope *retrieval*
# rejection is reported but not gated -- the score floor lets some junk through
# by design, and the system prompt is the defense; what IS gated is that the
# judged answers stay faithful (an out-of-scope answer citing theses fails).
STRICT_HIT_RATE = 0.80
STRICT_FAITHFULNESS = 1.00
STRICT_CONTEXT_PRECISION = 0.70

_JUDGE_NUM_PREDICT = 512

CONTEXT_JUDGE_PROMPT = """You grade a retrieval system for a college thesis database.
Given a user question and numbered thesis excerpts, decide for each excerpt whether it is
RELEVANT: it is genuinely about the topic asked, not merely sharing keywords with it.
Reply with ONLY a JSON object, no other text:
{"relevant": [<numbers of the relevant excerpts>]}"""

ANSWER_JUDGE_PROMPT = """You grade an academic librarian chatbot that must answer questions about
CSPC theses ONLY from the thesis excerpts provided (general librarian advice like citation
formatting may come from its own knowledge). Given the question, the excerpts, and the bot's
answer, reply with ONLY a JSON object, no other text:
{
  "faithful": true or false,
  "unsupported_claims": ["<claims about theses not supported by the excerpts>"],
  "relevance": <1-5, how well the answer addresses the user's question>
}
Rules for "faithful":
- A claim is unsupported ONLY if the excerpts contain nothing that supports it. Paraphrasing,
  summarizing, or reorganizing excerpt content is fully faithful -- do not demand exact wording.
- Before flagging a claim, re-scan every excerpt for the fact (numbers, rankings, findings);
  flag it only if it is genuinely absent.
- An excerpt's presence itself establishes that the thesis exists in the collection -- never
  flag claims that a thesis exists, is local, or is part of the collection.
- Reporting what an excerpt's literature review says other studies found is faithful.
- Unfaithful means: citing a thesis title that appears in NO excerpt, or stating findings,
  figures, or authors that appear in NO excerpt.
- If no excerpts were provided, a faithful answer cites no theses at all; saying nothing was
  found is faithful and, for an unanswerable question, highly relevant."""


def _parse_json(reply: str) -> dict | None:
    """First JSON object in ``reply``, or None -- small models add chatter."""
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _judge(system: str, user: str, model: str) -> dict | None:
    try:
        reply = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model, num_predict=_JUDGE_NUM_PREDICT,
        )
    except llm.ChatError:
        return None
    return _parse_json(reply)


def _default_judge_model(answer_model: str | None) -> str:
    """A configured model different from the answerer, if there is one."""
    answering = (answer_model or config.DEFAULT_CHAT_MODEL).strip()
    others = [m for m in config.LLM_MODELS if m != answering]
    return others[0] if others else answering


def _numbered_excerpts(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {c.get('title', 'Untitled')} ({c.get('year', 'n.d.')})\n{c.get('text', '')}"
        for i, c in enumerate(chunks, start=1)
    )


def evaluate_case(case: dict, *, answers: bool, judge: bool,
                  model: str | None, judge_model: str) -> dict:
    q = case["question"]
    flt = rag.infer_filter(q)
    chunks = vectorstore.query(q, metadata_filter=flt)
    retrieved_files = [c.get("source_file", "") for c in chunks]

    row: dict = {
        "id": case["id"], "kind": case["kind"], "question": q, "filter": flt,
        "retrieved": [
            {"score": round(c["score"], 3), "source_file": c.get("source_file"),
             "title": (c.get("title") or "")[:80]} for c in chunks
        ],
    }

    if want := case.get("expect_filter"):
        row["filter_ok"] = want <= set(flt or {})

    if case["kind"] == "retrieval":
        expected = case.get("expected", [])
        hit = any(f in expected for f in retrieved_files)
        if not hit and (prog := case.get("any_program")):
            hit = any(c.get("program") == prog for c in chunks)
        row["hit"] = hit
        if expected:
            found = {f for f in retrieved_files if f in expected}
            row["recall"] = len(found) / len(expected)
    else:
        # Correct behavior for everything non-retrieval is empty context.
        row["rejected"] = not chunks

    if answers:
        try:
            row["answer"] = llm.chat(rag.build_messages(q, [], chunks), model=model)
        except llm.ChatError as e:
            row["answer_error"] = str(e)

    if judge and chunks and case["kind"] == "retrieval":
        verdict = _judge(
            CONTEXT_JUDGE_PROMPT,
            f"Question: {q}\n\nExcerpts:\n\n{_numbered_excerpts(chunks)}",
            judge_model,
        )
        if verdict and isinstance(verdict.get("relevant"), list):
            relevant = {n for n in verdict["relevant"] if isinstance(n, int)}
            row["context_precision"] = len(relevant & set(range(1, len(chunks) + 1))) / len(chunks)

    if judge and row.get("answer"):
        excerpts = _numbered_excerpts(chunks) if chunks else "(no excerpts were provided)"
        verdict = _judge(
            ANSWER_JUDGE_PROMPT,
            f"Question: {q}\n\nExcerpts:\n\n{excerpts}\n\nBot's answer:\n{row['answer']}",
            judge_model,
        )
        if verdict:
            row["faithful"] = bool(verdict.get("faithful"))
            row["unsupported_claims"] = verdict.get("unsupported_claims") or []
            if isinstance(verdict.get("relevance"), (int, float)):
                row["answer_relevance"] = float(verdict["relevance"])

    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    hits = [r["hit"] for r in rows if "hit" in r]
    recalls = [r["recall"] for r in rows if "recall" in r]
    oos = [r["rejected"] for r in rows if r["kind"] == "out_of_scope"]
    clean = [r["rejected"] for r in rows if r["kind"] in ("general", "smalltalk")]
    filter_oks = [r["filter_ok"] for r in rows if "filter_ok" in r]
    precisions = [r["context_precision"] for r in rows if "context_precision" in r]
    faithfuls = [r["faithful"] for r in rows if "faithful" in r]
    relevances = [r["answer_relevance"] for r in rows if "answer_relevance" in r]
    return {
        "hit_rate": _mean([float(h) for h in hits]),
        "mean_recall": _mean(recalls),
        "oos_rejection_rate": _mean([float(x) for x in oos]),
        "clean_context_rate": _mean([float(x) for x in clean]),
        "filter_accuracy": _mean([float(x) for x in filter_oks]),
        "context_precision": _mean(precisions),
        "faithfulness": _mean([float(x) for x in faithfuls]),
        "mean_answer_relevance": _mean(relevances),
        "counts": {
            "retrieval": len(hits), "out_of_scope": len(oos),
            "general_smalltalk": len(clean), "judged_answers": len(faithfuls),
        },
    }


def _fmt(value: float | None, *, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def print_report(rows: list[dict], summary: dict) -> None:
    for r in rows:
        marks = []
        if "hit" in r:
            marks.append("hit " + ("Y" if r["hit"] else "N"))
        if "recall" in r:
            marks.append(f"recall {r['recall']:.2f}")
        if "rejected" in r:
            marks.append("no-context " + ("Y" if r["rejected"] else "N"))
        if "filter_ok" in r:
            marks.append("filter " + ("Y" if r["filter_ok"] else "N"))
        if "context_precision" in r:
            marks.append(f"ctx-prec {r['context_precision']:.2f}")
        if "faithful" in r:
            marks.append("faithful " + ("Y" if r["faithful"] else "N"))
        if "answer_relevance" in r:
            marks.append(f"rel {r['answer_relevance']:.0f}/5")
        print(f"\n=== [{r['kind']}] {r['id']}: {r['question']}")
        print(f"    filter: {r['filter']}")
        for c in r["retrieved"]:
            print(f"    {c['score']:.3f}  {c['source_file']}")
        if r.get("answer_error"):
            print(f"    !! answer failed: {r['answer_error']}")
        if r.get("unsupported_claims"):
            for claim in r["unsupported_claims"]:
                print(f"    !! unsupported claim: {claim}")
        print(f"    -> {', '.join(marks) if marks else '(no metrics)'}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  retrieval hit rate      {_fmt(summary['hit_rate'])}"
          f"  ({summary['counts']['retrieval']} labeled questions)")
    print(f"  expected-file recall    {_fmt(summary['mean_recall'], pct=False)}")
    print(f"  out-of-scope rejection  {_fmt(summary['oos_rejection_rate'])}"
          "  (retrieval-level; prompt is the final defense)")
    print(f"  clean context rate      {_fmt(summary['clean_context_rate'])}"
          "  (general/smalltalk got no thesis context)")
    print(f"  filter accuracy         {_fmt(summary['filter_accuracy'])}")
    print(f"  context precision       {_fmt(summary['context_precision'])}   (LLM-judged)")
    print(f"  faithfulness            {_fmt(summary['faithfulness'])}   (LLM-judged)")
    print(f"  answer relevance        {_fmt(summary['mean_answer_relevance'], pct=False)}/5 (LLM-judged)")


def strict_failures(summary: dict) -> list[str]:
    gates = [
        ("hit_rate", STRICT_HIT_RATE),
        ("faithfulness", STRICT_FAITHFULNESS),
        ("context_precision", STRICT_CONTEXT_PRECISION),
    ]
    return [
        f"{name} {summary[name]:.2f} < {floor:.2f}"
        for name, floor in gates
        if summary[name] is not None and summary[name] < floor
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", action="store_true", help="also generate LLM answers")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judge context precision, faithfulness, relevance (implies --answers)")
    ap.add_argument("--model", default=None, help="answering model (default: config default)")
    ap.add_argument("--judge-model", default=None,
                    help="judging model (default: first configured model that isn't the answerer)")
    ap.add_argument("--json", nargs="?", const="", default=None, metavar="PATH",
                    help="write a JSON report (default path: eval_results/eval-<timestamp>.json)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any STRICT_* threshold is missed (for CI / pre-deploy)")
    args = ap.parse_args()
    answers = args.answers or args.judge
    judge_model = args.judge_model or _default_judge_model(args.model)

    if args.judge:
        print(f"answering with {args.model or config.DEFAULT_CHAT_MODEL}, judging with {judge_model}")

    rows = [
        evaluate_case(case, answers=answers, judge=args.judge,
                      model=args.model, judge_model=judge_model)
        for case in CASES
    ]
    summary = summarize(rows)
    print_report(rows, summary)

    if args.json is not None:
        path = Path(args.json) if args.json else (
            Path("eval_results") / f"eval-{datetime.now():%Y%m%d-%H%M%S}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "timestamp": datetime.now().astimezone().isoformat(),
            "answer_model": args.model or config.DEFAULT_CHAT_MODEL,
            "judge_model": judge_model if args.judge else None,
            "retrieval_config": {
                "top_k": config.RETRIEVAL_TOP_K,
                "min_score": config.RETRIEVAL_MIN_SCORE,
                "min_score_filtered": config.RETRIEVAL_MIN_SCORE_FILTERED,
                "max_per_thesis": config.RETRIEVAL_MAX_PER_THESIS,
            },
            "summary": summary,
            "cases": rows,
        }, indent=2, ensure_ascii=False))
        print(f"\nreport written to {path}")

    if args.strict:
        if failures := strict_failures(summary):
            print("\nSTRICT GATE FAILED: " + "; ".join(failures))
            sys.exit(1)
        print("\nstrict gate passed")


if __name__ == "__main__":
    main()
