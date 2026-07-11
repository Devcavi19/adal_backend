"""LLM wrappers: embeddings + chat, local (Ollama) and cloud.

Embeddings call Ollama's native ``/api/embed`` endpoint directly (via
``requests``) rather than going through LiteLLM: LiteLLM's sync ``embedding()``
reuses a module-level async client whose event loop gets closed between calls, so
a multi-batch upsert fails intermittently with "Event loop is closed". The native
endpoint batches and is more reliable.

Chat (``chat()``) walks ``config.LLM_MODELS`` as a fallback chain. Local
``ollama/*`` models go straight to Ollama's native ``/api/chat`` (with thinking
disabled, so RAG answers come back promptly instead of burning the token budget
on a reasoning trace); cloud models (``gemini/...``, ``gpt-...``) go via LiteLLM.
"""

import json
import re
from collections.abc import Iterator

import requests

from . import config


class ChatError(RuntimeError):
    """All configured chat models failed."""


# Appended when generation stops at the token budget (Ollama done_reason /
# LiteLLM finish_reason == "length") so truncation is visible instead of the
# answer just ending mid-sentence. With the truncated turn in history, asking
# to continue picks up where it stopped.
TRUNCATION_NOTICE = '\n\n*…answer hit the length limit — say "continue" for the rest.*'


def _ollama_url(path: str) -> str:
    """Full Ollama URL for ``path``, tolerating a base with or without a scheme."""
    base = config.OLLAMA_BASE_URL
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return base.rstrip("/") + path


def _bare(model: str) -> str:
    """Strip the optional ``ollama/`` prefix to get the raw Ollama model name."""
    return model[len("ollama/"):] if model.startswith("ollama/") else model


def _model_name() -> str:
    """Bare Ollama model name for the configured embedding model."""
    return _bare(config.EMBEDDING_MODEL)


# Ollama's bge-m3 deterministically produces NaN vectors for rare specific
# inputs (e.g. some punctuation/word combinations); the server then 500s with
# this marker because NaN is not encodable as JSON. One bad input fails the
# whole batch, so the fallback isolates per item and sanitizes (Ollama 0.21.0).
_EMBED_NAN_ERROR = "unsupported value: NaN"


def _embed_call(batch: list[str]) -> list[list[float]] | None:
    """One native ``/api/embed`` call for ``batch``.

    Returns ``None`` when Ollama trips its NaN bug (see ``_EMBED_NAN_ERROR``);
    raises for any other failure.
    """
    resp = requests.post(
        _ollama_url("/api/embed"),
        json={"model": _model_name(), "input": batch, "keep_alive": config.OLLAMA_KEEP_ALIVE},
        timeout=120,
    )
    if resp.status_code == 500 and _EMBED_NAN_ERROR in resp.text:
        return None
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) != len(batch):
        raise RuntimeError(
            f"Ollama embed returned {len(embeddings or [])} vectors for "
            f"{len(batch)} inputs (model {_model_name()}). Response: {str(data)[:200]}"
        )
    return embeddings


def _desensitize(text: str) -> str:
    """Punctuation stripped + whitespace collapsed.

    Empirically sidesteps the NaN bug while keeping the embedding semantically
    equivalent for retrieval (whitespace/punctuation perturbations do NOT help;
    removing punctuation does).
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _embed_one(text: str) -> list[float]:
    """Embed one text, sanitizing progressively around Ollama's NaN bug."""
    tried: list[str] = []
    for variant in (text, _desensitize(text), _desensitize(text).lower()):
        if not variant or variant in tried:
            continue
        tried.append(variant)
        vecs = _embed_call([variant])
        if vecs is not None:
            return vecs[0]
    raise RuntimeError(
        f"Ollama embed (model {_model_name()}) produced NaN vectors for input "
        f"{text[:100]!r} even after sanitizing."
    )


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of texts with the configured model, preserving order.

    Calls the local Ollama model in batches. Returns one vector per input text,
    each of length ``config.EMBED_DIM``. A batch that trips Ollama's NaN bug is
    retried per item with sanitized fallbacks (see ``_embed_one``).
    """
    batch_size = batch_size or config.EMBED_BATCH
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = _embed_call(batch)
        if embeddings is None:
            embeddings = [_embed_one(t) for t in batch]
        vectors.extend(embeddings)

    if vectors and len(vectors[0]) != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding width {len(vectors[0])} != configured EMBED_DIM "
            f"{config.EMBED_DIM} for model {config.EMBEDDING_MODEL}. "
            f"Fix EMBED_DIM (and the Pinecone index dimension) to match."
        )
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]


# --- Chat -------------------------------------------------------------------

def _ollama_payload(model: str, messages: list[dict], *, stream: bool,
                    num_predict: int | None = None) -> dict:
    """Request body for Ollama's ``/api/chat``.

    Thinking is disabled (``config.OLLAMA_THINK``): for RAG we want the answer,
    not a reasoning trace that eats the token budget. ``num_ctx`` caps the
    context window Ollama allocates (its default is the model maximum -- 262K
    for qwen3.5, which costs ~14 GB of KV cache and a slow load), and
    ``keep_alive`` keeps the model resident so idle users don't pay a reload.
    """
    return {
        "model": _bare(model),
        "messages": messages,
        "stream": stream,
        "think": config.OLLAMA_THINK,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"num_predict": num_predict or config.LLM_NUM_PREDICT,
                    "num_ctx": config.OLLAMA_NUM_CTX},
    }


def _ollama_chat(model: str, messages: list[dict], *, num_predict: int | None = None) -> str:
    """One chat completion from a local Ollama model via native ``/api/chat``."""
    resp = requests.post(
        _ollama_url("/api/chat"),
        json=_ollama_payload(model, messages, stream=False, num_predict=num_predict),
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    if not content.strip():
        raise RuntimeError(f"Ollama model {model} returned an empty answer.")
    if data.get("done_reason") == "length":
        content += TRUNCATION_NOTICE
    return content


def _ollama_chat_stream(model: str, messages: list[dict]) -> Iterator[str]:
    """Yield answer pieces from a local Ollama model as they are generated."""
    with requests.post(
        _ollama_url("/api/chat"),
        json=_ollama_payload(model, messages, stream=True),
        timeout=300,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if error := chunk.get("error"):
                raise RuntimeError(f"Ollama model {model} stream error: {error}")
            if piece := (chunk.get("message") or {}).get("content"):
                yield piece
            if chunk.get("done") and chunk.get("done_reason") == "length":
                yield TRUNCATION_NOTICE


def preload() -> None:
    """Ask Ollama to load the default chat model into memory (best-effort).

    An ``/api/chat`` call with no messages loads the model without generating,
    so the first real user doesn't pay the multi-second cold load.
    """
    model = config.DEFAULT_CHAT_MODEL
    if not model.startswith("ollama/"):
        return
    requests.post(
        _ollama_url("/api/chat"),
        json={
            "model": _bare(model),
            "messages": [],
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": config.OLLAMA_NUM_CTX},
        },
        timeout=120,
    )


def _litellm_chat(model: str, messages: list[dict], *, num_predict: int | None = None) -> str:
    """One chat completion from a cloud provider via LiteLLM."""
    import litellm  # local import: only cloud models pull in litellm

    resp = litellm.completion(
        model=model, messages=messages, max_tokens=num_predict or config.LLM_NUM_PREDICT,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    if choice.finish_reason == "length":
        content += TRUNCATION_NOTICE
    return content


def _litellm_chat_stream(model: str, messages: list[dict]) -> Iterator[str]:
    """Yield answer pieces from a cloud provider via LiteLLM."""
    import litellm  # local import: only cloud models pull in litellm

    resp = litellm.completion(
        model=model, messages=messages, max_tokens=config.LLM_NUM_PREDICT, stream=True,
    )
    for chunk in resp:
        choice = chunk.choices[0]
        if piece := choice.delta.content:
            yield piece
        if choice.finish_reason == "length":
            yield TRUNCATION_NOTICE


def available_models() -> list[dict]:
    """The selectable chat models for the frontend: ``[{"id", "label"}, ...]``."""
    return [{"id": m, "label": config.MODEL_LABELS.get(m, m)} for m in config.LLM_MODELS]


def _one(model: str, messages: list[dict], *, num_predict: int | None = None) -> str:
    """Dispatch a single completion: ``ollama/*`` local, everything else LiteLLM."""
    if model.startswith("ollama/"):
        return _ollama_chat(model, messages, num_predict=num_predict)
    return _litellm_chat(model, messages, num_predict=num_predict)


def _one_stream(model: str, messages: list[dict]) -> Iterator[str]:
    """Dispatch a single streaming completion, same routing as ``_one``."""
    if model.startswith("ollama/"):
        return _ollama_chat_stream(model, messages)
    return _litellm_chat_stream(model, messages)


def _fallback_chain(model: str | None) -> list[str]:
    """The models to try, in order: the user's validated pick, then the fallback.

    ``model`` is the user's pick from the frontend; it must be one of
    ``config.LLM_MODELS`` -- an unlisted model is rejected so the client can't run
    an arbitrary, possibly costly, model.
    """
    chosen = (model or config.DEFAULT_CHAT_MODEL).strip()
    if chosen not in config.LLM_MODELS:
        raise ChatError(
            f"Model {chosen!r} is not a selectable option "
            f"(allowed: {', '.join(config.LLM_MODELS)})."
        )
    chain = [chosen]
    if config.CHAT_FALLBACK_MODEL and config.CHAT_FALLBACK_MODEL != chosen:
        chain.append(config.CHAT_FALLBACK_MODEL)
    return chain


def chat(messages: list[dict], *, model: str | None = None,
         num_predict: int | None = None) -> str:
    """Return an answer from ``model`` (or the default), falling back on failure.

    If the chosen model errors, the configured local fallback is tried so the
    user still gets an answer. Raises ``ChatError`` only if every attempt fails.
    ``num_predict`` overrides the configured answer-length cap -- used for
    short auxiliary generations like retrieval-query condensation.
    """
    errors: list[str] = []
    for m in _fallback_chain(model):
        try:
            return _one(m, messages, num_predict=num_predict)
        except Exception as e:  # noqa: BLE001 -- fall through to the fallback
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
    raise ChatError("All chat models failed -> " + " | ".join(errors))


def chat_stream(messages: list[dict], *, model: str | None = None) -> Iterator[str]:
    """Yield answer pieces from ``model`` (or the default), falling back on failure.

    Fallback only happens before the first piece is produced: once a model has
    started answering, its output has already been streamed to the client, so a
    mid-stream error propagates instead of restarting on another model.
    """
    errors: list[str] = []
    for m in _fallback_chain(model):
        stream = _one_stream(m, messages)
        try:
            first = next(stream)
        except StopIteration:
            errors.append(f"{m}: returned an empty answer")
            continue
        except Exception as e:  # noqa: BLE001 -- fall through to the fallback
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
            continue
        yield first
        yield from stream
        return
    raise ChatError("All chat models failed -> " + " | ".join(errors))
