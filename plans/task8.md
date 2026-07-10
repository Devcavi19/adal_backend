# Task 8 — Deploy

Source: `docs/backend-setup-plan.md` — Phase 5, Build order item 8.

## Goal
Expose the backend (running locally / on the HPC) to the internet via an **ngrok tunnel** and point the frontend's `VITE_API_URL` at the tunnel URL, keeping the setup lightweight — no hosting provider needed.

## Notes
- Backend keeps running where it already runs (`uvicorn app.main:app`, local machine or HPC node) — ngrok just forwards a public URL to that local port. No Docker, no Render/Railway/Fly.io account needed.
- `ngrok http <port>` (default backend port — check `uvicorn` invocation, typically 8000) gives a public `https://<random>.ngrok-free.app` URL that forwards to the local server.
- Free ngrok URLs are ephemeral — they change every time the tunnel restarts. A paid plan or `ngrok config` static domain avoids having to update `VITE_API_URL` each session.
- On an HPC, run ngrok on the same node/session as `uvicorn` (or via an SSH tunnel to a node that has outbound internet — compute nodes sometimes don't). Confirm outbound internet access is allowed before relying on this for a live demo.
- Set `FRONTEND_ORIGIN` (backend CORS) to the deployed frontend's URL, and keep the ngrok URL itself out of CORS restrictions (CORS is about who calls the API, not where it's hosted).
- Add the `X-API-Key` header check before exposing publicly (see Task 4's "keep-it-lightweight rules" — no auth was added yet). This matters more with ngrok since the tunnel URL is otherwise wide open to anyone who finds it.
- `ngrok` free tier shows an interstitial warning page for browser visits unless the request includes the right headers — shouldn't affect the frontend's `fetch`/XHR calls, but worth checking if something looks broken.

## Steps
- [ ] Install/authenticate ngrok (`ngrok config add-authtoken <token>`) on the machine/HPC node running the backend.
- [ ] Start the backend locally (`uvicorn app.main:app --host 0.0.0.0 --port <port>`).
- [ ] Run `ngrok http <port>` and copy the public HTTPS forwarding URL.
- [ ] Add the `X-API-Key` header check to `app/main.py` before leaving the tunnel up.
- [ ] Confirm `GET /api/health` responds through the ngrok URL.
- [ ] Update the frontend's `VITE_API_URL` to the ngrok URL (local `.env` or hosting config, depending on where the frontend runs).
- [ ] Restart/redeploy the frontend and chat end-to-end against the tunneled backend.
- [ ] If the tunnel restarts (new random URL on free tier), repeat the `VITE_API_URL` update.

## Done when
The frontend can chat with Adal against the backend exposed through ngrok, with CORS and API-key auth in place.
