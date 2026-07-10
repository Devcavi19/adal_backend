# Task 5 — Wire the frontend

Source: `docs/backend-setup-plan.md` — Phase 4, Build order item 5.

## Goal
Point the existing React frontend at the new backend through its single swap point, and chat end-to-end.

## Only file that changes: `src/services/chatService.ts`
Its signature is the contract the UI depends on. The backend (`app/main.py`, see [[task4]]) accepts an optional `history` and `model` on the request and always returns `sources` (possibly empty):
```ts
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatReply {
  text: string;
  sources?: { title: string; year: number | string; section: string }[];
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function sendMessage(
  conversationId: string,
  text: string,
  history: ChatMessage[] = [],   // last ~10 turns; optional but improves follow-up answers
  model?: string,                 // optional; must be one of GET /api/models, else 400
): Promise<ChatReply> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversationId, message: text, history, model }),
  });
  if (!res.ok) throw new Error(`Adal API error: ${res.status}`);
  return res.json();
}
```
`model` is optional — omit it (or leave it undefined) to get `config.DEFAULT_CHAT_MODEL`. A 502 means every LLM attempt (chosen model + fallback) failed; surface it as a chat error in the UI.

## Steps
- [X] Add `VITE_API_URL` to the frontend `.env` (e.g. `http://localhost:8000`).
- [X] Confirm the backend's `FRONTEND_ORIGIN` (`.env`, defaults to `http://localhost:5173`) matches the frontend dev server's actual origin, or CORS will block the request.
- [X] Update `src/services/chatService.ts` with the implementation above.
- [X] Run the backend (`uvicorn app.main:app --reload --port 8000`) and the frontend dev server together.
- [X] Chat end-to-end from the UI and confirm a grounded, cited reply renders correctly (markdown renders via `react-markdown`).
- [X] Optional: call `GET /api/models` to populate a model dropdown instead of hardcoding ids (see [[task4]] §4.3 for the model-menu design).

## Later upgrades (don't need to do now — don't break the contract)
- SSE streaming (`text/event-stream`) for token-by-token replies.
- Rendering `sources` as citation chips under assistant messages.
- Surfacing `GET /api/health`'s `default_model` in the UI (e.g. a status indicator).

## Done when
A message typed in the running frontend produces a real answer from the FastAPI backend, sourced from the sample thesis.
