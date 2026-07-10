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

import requests

from . import config


class ChatError(RuntimeError):
    """All configured chat models failed."""


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


def embed_texts(texts: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of texts with the configured model, preserving order.

    Calls the local Ollama model in batches. Returns one vector per input text,
    each of length ``config.EMBED_DIM``.
    """
    batch_size = batch_size or config.EMBED_BATCH
    url = _ollama_url("/api/embed")
    model = _model_name()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = requests.post(url, json={"model": model, "input": batch}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None or len(embeddings) != len(batch):
            raise RuntimeError(
                f"Ollama embed returned {len(embeddings or [])} vectors for "
                f"{len(batch)} inputs (model {model}). Response: {str(data)[:200]}"
            )
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

def _ollama_chat(model: str, messages: list[dict]) -> str:
    """One chat completion from a local Ollama model via native ``/api/chat``.

    Thinking is disabled (``config.OLLAMA_THINK``): for RAG we want the answer,
    not a reasoning trace that eats the token budget.
    """
    resp = requests.post(
        _ollama_url("/api/chat"),
        json={
            "model": _bare(model),
            "messages": messages,
            "stream": False,
            "think": config.OLLAMA_THINK,
            "options": {"num_predict": config.LLM_NUM_PREDICT},
        },
        timeout=300,
    )
    resp.raise_for_status()
    content = (resp.json().get("message") or {}).get("content", "")
    if not content.strip():
        raise RuntimeError(f"Ollama model {model} returned an empty answer.")
    return content


def _litellm_chat(model: str, messages: list[dict]) -> str:
    """One chat completion from a cloud provider via LiteLLM."""
    import litellm  # local import: only cloud models pull in litellm

    resp = litellm.completion(
        model=model, messages=messages, max_tokens=config.LLM_NUM_PREDICT,
    )
    return resp.choices[0].message.content or ""


def available_models() -> list[dict]:
    """The selectable chat models for the frontend: ``[{"id", "label"}, ...]``."""
    return [{"id": m, "label": config.MODEL_LABELS.get(m, m)} for m in config.LLM_MODELS]


def _one(model: str, messages: list[dict]) -> str:
    """Dispatch a single completion: ``ollama/*`` local, everything else LiteLLM."""
    if model.startswith("ollama/"):
        return _ollama_chat(model, messages)
    return _litellm_chat(model, messages)


def chat(messages: list[dict], *, model: str | None = None) -> str:
    """Return an answer from ``model`` (or the default), falling back on failure.

    ``model`` is the user's pick from the frontend; it must be one of
    ``config.LLM_MODELS`` -- an unlisted model is rejected so the client can't run
    an arbitrary, possibly costly, model. If the chosen model errors, the
    configured local fallback is tried so the user still gets an answer. Raises
    ``ChatError`` only if every attempt fails.
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

    errors: list[str] = []
    for m in chain:
        try:
            return _one(m, messages)
        except Exception as e:  # noqa: BLE001 -- fall through to the fallback
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
    raise ChatError("All chat models failed -> " + " | ".join(errors))
