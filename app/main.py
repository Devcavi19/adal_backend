"""FastAPI app, CORS, routes"""

import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config, llm, rag, vectorstore


def _warmup() -> None:
    """Best-effort: open the Pinecone connection and load both Ollama models.

    The first Pinecone query pays ~1-3s of TLS/host resolution and a cold
    Ollama model pays a multi-second load; doing both here means no user does.
    Failures are ignored -- real requests surface real errors.
    """
    try:
        vectorstore.query("warmup", top_k=1)
    except Exception:
        pass
    try:
        llm.preload()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(title="Adal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversationId: str | None = None
    message: str
    history: list[ChatMessage] = []
    model: str | None = None


class Source(BaseModel):
    title: str | None = None
    year: int | str | None = None
    section: str | None = None


class ChatResponse(BaseModel):
    text: str
    sources: list[Source] = []


@app.get("/api/health")
def health():
    return {"status": "ok", "default_model": config.DEFAULT_CHAT_MODEL}


@app.get("/api/models")
def models():
    return llm.available_models()


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.model is not None and req.model not in config.LLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {req.model!r}.")
    history = [m.model_dump() for m in req.history]
    try:
        return rag.answer(req.message, history, model=req.model)
    except llm.ChatError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-sent events: one ``sources`` event, then ``data`` events with
    answer text as it is generated, then a ``done`` event. Retrieval runs
    before the response starts so its errors still surface as HTTP errors."""
    if req.model is not None and req.model not in config.LLM_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model {req.model!r}.")
    history = [m.model_dump() for m in req.history]
    chunks = rag.retrieve(req.message, history)

    def events():
        yield f"event: sources\ndata: {json.dumps(rag.dedupe_sources(chunks))}\n\n"
        try:
            for piece in rag.answer_stream(req.message, history, chunks=chunks, model=req.model):
                yield f"data: {json.dumps(piece)}\n\n"
        except llm.ChatError as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
