# Task 8 — Deploy

Source: `docs/backend-setup-plan.md` — Phase 5, Build order item 8.

## Goal
Deploy the backend and point the frontend's `VITE_API_URL` at it, keeping the setup lightweight.

## Notes
- **Render / Railway / Fly.io** free-hobby tier runs `uvicorn` directly from the repo — no Docker needed (both auto-detect `requirements.txt`).
- Pinecone serverless + a cheap model (Gemini Flash / GPT-5 mini) keeps a thesis-defense-scale demo effectively free.
- Set `FRONTEND_ORIGIN` (backend CORS) to the deployed frontend URL.
- Add the `X-API-Key` header check before deploying publicly (see Task 4's "keep-it-lightweight rules" — no auth was added yet).

## Steps
- [ ] Choose a host (Render / Railway / Fly.io) and connect the `adal_backend` repo.
- [ ] Set production environment variables (`PINECONE_API_KEY`, `PINECONE_INDEX`, `LLM_MODELS`, provider API keys, `FRONTEND_ORIGIN`).
- [ ] Add a simple `X-API-Key` header check to `app/main.py` for public deployment.
- [ ] Deploy and confirm `GET /api/health` responds on the public URL.
- [ ] Update the frontend's `VITE_API_URL` to the deployed backend URL (in its production `.env` / hosting config).
- [ ] Deploy/redeploy the frontend and chat end-to-end against the live backend.

## Done when
The publicly deployed frontend can chat with Adal against the publicly deployed backend, with CORS and API-key auth in place.
