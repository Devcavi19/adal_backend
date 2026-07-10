# Adal UI/UX Redesign — Design Spec

**Date:** 2026-07-10
**Status:** Approved direction, pending final spec review
**Scope:** Full frontend — chat experience, sidebar, login, and marketing pages (About, Charter, Pricing)

## Goal

Bring the Adal frontend (AI-Driven Academic Librarian for CSPC) to the visual and interaction quality of modern AI products (ChatGPT, Claude, Perplexity), with an institutional identity built on CSPC's official colors.

## Design language

### Palette

Built from the CSPC logo colors, navy-primary with Computer Studies gold as a restrained accent. The logo's red/green/orange stay in the seal only.

| Token | Value | Role |
|---|---|---|
| Navy 900 (primary) | `#0f2e66` | Brand, buttons, links, user message bubbles, focus rings |
| Navy 600 | `#1a4494` | Hover states, links on light bg |
| Navy 500 | `#24457f` | Dark-mode primary surfaces (user bubbles, active buttons) |
| Gold (accent) | `#fcaf08` | Active-item indicator, send icon, highlights, small brand moments — used sparingly |
| White | `#ffffff` | Light-mode surfaces |
| Mist | `#f4f6f9` | Light-mode panels (sidebar, code blocks) |
| Neutrals | navy-tinted gray scale (e.g. text-secondary `#475269`, muted `#8a93a6`, borders `#d8dce4` / `#e3e7ee`) | Text and borders |
| Ink | `#0b1526` | Dark-mode background (sidebar `#0e1b33`, panels `#101d38`/`#1a2c4f`, borders `#1c2c4d`/`#26385c`) |

Semantic tokens (`--color-surface`, `--color-surface-panel`, `--color-text-primary`, `--color-text-secondary`, `--color-border`, `--color-accent`, …) are defined once and flip for dark mode.

### Typography

- **UI/body:** Inter.
- **Headlines (welcome greeting, marketing heroes, login value statement):** Source Serif 4 — collegiate, Claude-like warmth. Chosen over all-sans in review.
- Both loaded via **Fontsource packages** (self-hosted, works offline on campus). No Google Fonts CDN.

### Iconography

`lucide-react` SVG icons everywhere. Font Awesome (CDN) removed.

### Theming

- Light (default) + dark mode, with a `light | dark | system` preference.
- `ThemeProvider` React context persists choice to `localStorage` and sets `data-theme` on `<html>`.
- Inline script in `index.html` applies the saved theme before React mounts (no flash of wrong theme).
- Toggle appears in the sidebar footer and the marketing nav.

## Chat experience

### Welcome (empty) state

- Time-aware serif greeting ("Good morning/afternoon/evening, Student") under the Adal logo mark.
- Subtitle: "Your AI-Driven Academic Librarian. What are we researching today?"
- Four **suggested prompt chips** (find sources, citation help, summarize a paper, explain a concept); clicking one sends it as a message.
- Composer centered below the chips.
- Disclaimer under the composer: "Adal can make mistakes. Verify important sources with the CSPC Library."

### Conversation state

- Messages in a **centered readable column** (~44rem max width).
- **User messages:** navy bubbles, right-aligned (radius 16/16/4/16).
- **Adal replies:** open text with the Adal avatar mark, no bubble — better for long answers.
- **Markdown rendering** for bot replies via `react-markdown` + `remark-gfm` + `rehype-highlight` (code blocks get a copy button and syntax highlighting; light/dark highlight themes follow the app theme).
- **Hover actions** on bot messages: 👍 / 👎 feedback (toggleable, as today) and Copy. Timestamp on hover row.
- **Typing indicator:** animated three-dot pulse while the (mocked) reply is pending.
- Smooth auto-scroll to the newest message.

### Composer

- Auto-growing `textarea` (1→~8 rows), Enter sends, Shift+Enter newlines.
- Rounded 16px container with subtle navy-tinted shadow; send button navy with gold icon; disabled state when empty.
- Voice-input button remains as a placeholder (non-functional, as today).

### Sidebar

- Expanded (260px) ↔ collapsed icon rail (72px), animated; slide-over drawer on mobile (<768px), keeping the existing BottomNav.
- **Working chat history (local-first):** conversations persisted to `localStorage`, auto-titled from the first user message, grouped Today / Yesterday / Previous 7 days / Older, with rename and delete on hover. Active conversation marked with a gold left indicator.
- Footer: theme toggle, Feedback (opens existing modal, restyled), user profile row.

## Login & marketing pages

### Login

- Split panel: left = navy gradient brand panel with serif value statement ("Research smarter, with your library beside you.") and CSPC / Computer Studies attribution; right = clean form (email, password with visibility toggle, Sign in, Register link).
- Mobile: brand panel becomes a compact header above the form.
- Behavior unchanged (no real auth yet).

### Marketing layout (About, Charter, Pricing)

- Shared `MarketingLayout`: sticky top nav (logo, About/Charter/Pricing links with **gold underline on the active page**, "Open Adal" navy button), serif hero on a soft mist gradient, footer.
- **About:** hero + 3-feature card grid (source discovery, citation help, always open).
- **Charter:** same hero pattern + numbered principle cards.
- **Pricing:** card grid of tiers; recommended tier in navy with gold badge.
- All pages restyled with existing content preserved; dark mode supported.

## Technical architecture

### Stack changes

- **Add:** `tailwindcss` v4 + `@tailwindcss/vite`, `lucide-react`, `react-markdown`, `remark-gfm`, `rehype-highlight`, `@fontsource-variable/inter`, `@fontsource-variable/source-serif-4`.
- **Remove:** `bootstrap` dependency; CDN tags in `index.html` (Font Awesome, marked.js, highlight.js).
- **Delete:** all legacy CSS (`style.css`, `company.css`, `auth.css`, `marketing.css`, `premium.css`, `variables.css`, `react-adjustments.css`, `App.css`) — replaced by tokens + utilities. Dead code removed: `Hello.tsx`, `toSwitch/`.
- `clsx` + `tailwind-merge` (already installed) used via a `cn()` helper.

### Design tokens

Single source of truth in `src/index.css` using Tailwind v4 `@theme`: color scales, semantic surface/text/border tokens with dark-mode overrides under `[data-theme="dark"]`, font families, radii.

### Chat state

- `ConversationsContext` + `useChat` hook own all chat state: create conversation, send message, mock reply with typing delay, feedback toggle, rename, delete. Persisted to `localStorage`.
- Mock reply isolated in `src/services/chatService.ts` with an async, API-shaped signature (`sendMessage(conversationId, text) → Promise<Reply>`), so wiring the real backend later touches one file.
- Sends are treated as fallible: a failed send shows a retry affordance on the message (the mock never fails, but the UI pattern is ready).

### Component structure

```
src/
  components/
    ui/         Button, IconButton, Modal, Tooltip, ThemeToggle
    chat/       ChatInterface, WelcomeScreen, PromptChips, MessageList,
                Message, MarkdownContent, TypingIndicator, Composer
    sidebar/    Sidebar, HistoryList, HistoryItem, UserMenu
    layout/     MainLayout, MarketingLayout, MarketingNav, BottomNav
    feedback/   FeedbackModal (restyled)
  context/      ThemeProvider, ConversationsContext
  hooks/        useChat, useMediaQuery
  services/     chatService, FeedbackService
  pages/        LoginPage, AboutPage, CharterPage, PricingPage
```

## Out of scope

- Real backend/API integration, real authentication, voice input functionality.
- New marketing content (existing copy is preserved, restyled).
- The `dist/` folder (build artifact) and `.agent/` directory.

## Verification

- `npm run build` and `npm run lint` pass clean.
- Every screen driven in the browser: light + dark, desktop + mobile widths (<768px drawer/bottom-nav behavior), chat flow end-to-end (welcome → chips → send → typing → markdown reply → feedback/copy → history persistence across refresh).
