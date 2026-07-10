"""FastAPI app, CORS, routes"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config, llm, rag

app = FastAPI(title="Adal API")

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
