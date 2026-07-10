# Task 5 — Wire the frontend

Source: `docs/backend-setup-plan.md` — Phase 4, Build order item 5.

## Goal
Point the existing React frontend at the new backend through its single swap point, and chat end-to-end.

## Only file that changes: `src/services/chatService.ts`
Its signature is the contract the UI depends on:
```ts
export interface ChatReply {
  text: string;
  sources?: { title: string; year: number; section: string }[];
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function sendMessage(
  conversationId: string,
  text: string,
): Promise<ChatReply> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversationId, message: text }),
  });
  if (!res.ok) throw new Error(`Adal API error: ${res.status}`);
  return res.json();
}
```

## Steps
- [ ] Add `VITE_API_URL` to the frontend `.env` (e.g. `http://localhost:8000`).
- [ ] Update `src/services/chatService.ts` with the implementation above.
- [ ] Run the backend (`uvicorn app.main:app --reload --port 8000`) and the frontend dev server together.
- [ ] Chat end-to-end from the UI and confirm a grounded, cited reply renders correctly (markdown renders via `react-markdown`).

## Later upgrades (don't need to do now — don't break the contract)
- SSE streaming (`text/event-stream`) for token-by-token replies.
- Rendering `sources` as citation chips under assistant messages.

## Done when
A message typed in the running frontend produces a real answer from the FastAPI backend, sourced from the sample thesis.
