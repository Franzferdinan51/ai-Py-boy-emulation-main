# AI-Py-Boy Upgrade & Refactor Plan
**Started:** 2026-06-21
**Goal:** Full enhancement, refactor, and modernization

---

## 📚 Phase 1: Research Findings

### Reference Projects (2025-2026)

**1. NousResearch/pokemon-agent** ✅ Active
- FastAPI-based, agent-agnostic headless emulator server
- **Game sessions**: named playthroughs that bind emulator state, Hermes brain session, objectives, milestones
- **Field Log dashboard**: editorial broadcast UI with reasoning stream, grid map, objectives, telemetry
- **Collision maps** via RAM parsing (`GET /map/ascii`) + labelled A1..J9 grid overlay (`GET /screenshot/grid`)
- Structured JSON state (party, bag, badges, map, battle, dialog, collision grid)
- REST + WebSocket, Multi-game (Game Boy via PyBoy, GBA via PyGBA)
- WebSocket `/ws` for live event streaming
- Has a **start/stop/pause** driver with autopilot mode
- Per-session persistence under `<data_dir>/games/<id>/`
- Effort to integrate features: **M-L**

**2. CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark** ✅ Active
- Hugging Face smolagents-based
- **KnowledgeBase class**: structured agent memory (locations, Pokémon team, objectives, controls)
- Tools as standalone functions with `@tool` decorator
- Periodic planning steps for strategic reflection
- Optional dataset collection (screenshots + reasoning + actions → HuggingFace Hub)
- Effort to integrate: **M**

**3. PokéAgent Challenge (NeurIPS 2025)**
- Benchmark/competition for LLM agents on Pokémon
- Evaluates long-horizon planning, partial observability, decision-making
- LLM approaches with Gemini/Claude + RL baselines
- Reference for evaluation methodology

### Feature Ideas (From Research + Industry Trends)

**🔥 High Impact (do these first)**
1. **Game Sessions System** — named playthroughs with persistent state (save states, brain context, objectives, milestones) bound together. Resumes exactly where left off.
2. **Episodic Memory / KnowledgeBase** — structured memory of locations visited, Pokémon team history, objectives completed, controls learned. RAG-ready.
3. **Collision Map + Grid Overlay** — RAM-derived walkability data, agent navigates from real collision map not pixel-guessing. Grid screenshot overlay (A1..J9 labels).
4. **Reasoning Stream / Event Narration** — `POST /event` endpoint with THINK/DECIDE/ACT/MILESTONE/ALERT event types. Frontend "Field Log" feed.
5. **Telemetry / Stuck-Meter** — detect when agent is stuck (repeated positions, no progress N turns) and alert.

**⚡ Medium Impact**
6. **Three-Tier Objectives** — short-term / mid-term / long-term goals with progress tracking
7. **Milestone Timeline** — track badges, gym defeats, story events as persistent timeline
8. **Party Belt Visualization** — all 6 slots with types, HP bars, status, moves in compact UI
9. **Game-stage toggle** — SCREEN ⇄ GRID MAP view in UI
10. **High-level Actions API** — `walk_to(x,y)`, `interact_npc()`, `a_until_dialog_end()` — agent uses semantic actions instead of raw buttons

**🛠 Foundation / Code Health**
11. **Refactor server.py (6,778 lines, 92 routes) into Flask blueprints** by domain
12. **Remove `server_original.py`** (1,385 lines legacy, not imported)
13. **Flask 3.x upgrade** (currently 2.3.2 — Flask 3.1 is current)
14. **Update OpenAI SDK to 2.x** (currently `>=1.0.0`)
15. **Verify PyBoy 2.x APIs are correctly used** (`memory`, `save_state`/`load_state` with file-like)
16. **Standardize JSON response schemas** with empty-default fallbacks

**💡 Nice to Have**
17. **Browser-based grid-map rendering** — overlay grid on screenshot in UI
18. **Action timeline / replay** — replay past gameplay from event log
19. **RAG over game knowledge** — Pokemon walkthroughs, gym guides as retrieval context
20. **Multi-game sessions** — track which ROM/agent/objective is per session

### PyBoy 2.x API Confirmation

✅ Already mostly correct:
- `pyboy.tick(n=1, render=True)` — confirmed
- `pyboy.button('a')`, `button_press`/`button_release` — confirmed
- `pyboy.screen.image` / `.ndarray` / `.raw_buffer` — confirmed
- `pyboy.memory[0x0000:0x0010]` for memory reads — confirmed
- `pyboy.save_state(file_like)` / `pyboy.load_state(file_like)` — confirmed (use BytesIO, seek(0) before load)

⚠️ Things to verify in code:
- Are we using `screen.image` or older methods?
- Are we using `pyboy.memory` slicing for memory reads?
- Save/load using BytesIO or file paths?

### Flask 3.0 Migration Notes

Major breaking changes (need to address):
- Werkzeug ≥ 3.0 required
- `FLASK_ENV` env var / `ENV` config removed (use `--debug` or FLASK_DEBUG)
- `app.before_first_request` removed
- `_app_ctx_stack` removed
- JSON config keys deprecated (use `app.json` provider)
- `url_for` now accepts `self`
- `app.json_encoder`/`json_decoder` removed

This project uses `FLASK_ENV` in startup scripts — needs update.

---

## 🧹 Phase 2: Code Health Audit (Initial Findings)

- 0 TODOs/FIXMEs in code (clean!)
- 92 routes in single `server.py` (6,778 lines) — needs modular split
- `server_original.py` (1,385 lines) appears unused — can be removed
- Flask 2.3.2 in requirements (Flask 3.1 current)
- Python deps mostly current except Flask
- Frontend already on modern stack (Vite 8, React 19, TS 5.9, Tailwind 4)
- No flask installed locally — code hasn't been smoke-tested recently

---

## 📋 Phase 3: Implementation Roadmap

### Stage 1: Foundation (do first)
- [ ] Verify Python imports work for backend
- [ ] Bump Flask → 3.1.x, Werkzeug → 3.x
- [ ] Update startup scripts: `FLASK_ENV` → `FLASK_DEBUG`
- [ ] Remove `server_original.py`

### Stage 2: Refactor (medium risk)
- [ ] Split `server.py` into blueprints by domain:
  - `routes/emulator.py` — load_rom, game/state, button, action, screen, stream
  - `routes/save_load.py` — save_state, load_state
  - `routes/agent.py` — agent state/goal/chat/mode/errors/actions
  - `routes/spatial.py` — position, minimap, npcs, strategy
  - `routes/vision.py` — analyze, describe, ocr, summary
  - `routes/ai.py` — runtime, openclaw/config, providers, models
  - `routes/health.py` — health/runtime/emulator/stream
  - `routes/compat.py` — legacy aliases (/api/party, /api/inventory, /api/memory/watch)
- [ ] Create `routes/__init__.py` and `routes/registry.py` for blueprint registration
- [ ] Verify all 92 routes still work

### Stage 3: New Features (high impact)
- [ ] **Game Sessions System** — new module `sessions/` with session manager, persistence
  - `POST /api/games/new` — create session
  - `GET /api/games` — list sessions
  - `GET /api/games/current` — active session
  - `POST /api/games/<id>/load` — load session (restores emulator state + brain)
  - Each session persists: save state, objectives, milestones, stats, brain session id
- [ ] **Episodic Memory / KnowledgeBase** — `agent/memory` module
  - `GET /api/agent/memory` — full memory
  - `POST /api/agent/memory/note` — add knowledge note
  - `GET /api/agent/memory/locations` — visited locations
  - `GET /api/agent/memory/objectives` — completed objectives
  - `GET /api/agent/memory/events` — event history
- [ ] **Collision Map + Grid Overlay** — `spatial/collision` module
  - `GET /api/spatial/collision` — ASCII walkability grid
  - `GET /api/spatial/grid` — grid screenshot with A1..J9 overlay (or coordinates)
- [ ] **Reasoning Stream / Events** — `agent/events` module
  - `POST /api/agent/event` — emit THINK/DECIDE/ACT/MILESTONE/ALERT event
  - `GET /api/agent/events` — recent events
  - `WS /api/agent/events/stream` — live event stream
- [ ] **Telemetry / Stuck-Meter** — `agent/telemetry` module
  - `GET /api/agent/telemetry` — current metrics (stuck-meter, blackouts, actions, fps)
  - `POST /api/agent/telemetry/reset` — reset counters

### Stage 4: Frontend Updates
- [ ] Add Field Log panel (reasoning stream)
- [ ] Add grid map view (screen ⇄ grid toggle)
- [ ] Add party belt (all 6 slots compact)
- [ ] Add objectives panel (3-tier)
- [ ] Add telemetry/stuck-meter widget
- [ ] Add sessions manager UI (new/load/save)

### Stage 5: Verification
- [ ] Backend imports cleanly
- [ ] All routes return 200
- [ ] MCP server boots and exposes new tools
- [ ] Smoke test: load_rom → game/state → screen → action → save/load → session → memory
- [ ] Frontend `npm run build` succeeds
- [ ] Update API-CONTRACT.md, README.md, AGENTS.md

### Stage 6: Commit & Document
- [ ] Commit in logical chunks (foundation → refactor → features → docs)
- [ ] CHANGELOG entry
- [ ] Update version in version.ts

---

## 🚀 Execution Notes

- Cloud only (no local models for sub-agents — per SOUL.md 2026-06-10 directive)
- Sub-agent M2.7 failed once with 999 error — fell back to inline web research via Grok web_search
- Each phase commits independently so rollback is easy
- Don't break invariants from AGENTS.md: one tick owner, stable payloads, real save/load, null-safe UI

---

## ✅ Progress So Far

### Stage 1: Foundation — DONE
- [x] Removed unused `ai-game-server/src/backend/server_original.py` (1,385 lines legacy)
- [x] Bumped Flask 2.3.2 → `>=3.0,<4.0` and added Werkzeug 3.x constraint in `requirements.txt`
- [x] Updated OpenAI SDK to `>=1.50.0,<3.0` and added httpx, modernized pin syntax
- [x] Verified PyBoy 2.x APIs are still valid (memory, save_state, screen.image)

### Stage 2: New Features (additive) — DONE

Built 5 new feature modules under `ai-game-server/src/backend/agent_features/`:

**`sessions.py`** — Game Sessions System
- `POST /api/games/new` — create named playthrough
- `GET /api/games` — list all sessions (with manifest hydration from disk)
- `GET /api/games/current` — active session
- `GET|POST|DELETE /api/games/<id>` — get / update / delete
- `POST /api/games/<id>/activate` — set active session
- `POST|GET /api/games/<id>/save_state` — save/load emulator state bytes per session
- Persistent under `~/.ai-py-boy/data/games/<id>/manifest.json`
- 9 routes

**`memory.py`** — Agent Memory / KnowledgeBase
- `GET /api/agent/memory?session_id=...&type=...` — query by type
- `GET /api/agent/memory/search?session_id=...&q=...` — text search (RAG-lite)
- `GET /api/agent/memory/summary` — counts + latest by type
- `POST /api/agent/memory/note` — free-form note
- `POST /api/agent/memory/location` — visited location
- `POST /api/agent/memory/party_event` — Pokémon event (caught/leveled/fainted)
- `POST /api/agent/memory/objective_complete` — mark objective done (also adds milestone)
- `POST /api/agent/memory/fact` — structured key-value fact
- `POST /api/agent/memory/control_pattern` — learned button sequence → outcome
- JSONL storage per session, RAG-ready
- 9 routes

**`events.py`** — Reasoning Event Stream
- `GET /api/agent/events?kind=&session_id=&limit=` — list events
- `POST /api/agent/events` — emit THINK/DECIDE/ACT/MILESTONE/ALERT/OBSERVE/REFLECT
- `POST /api/agent/events/clear` — clear ring buffer
- `GET /api/agent/events/stats` — counts by kind and session
- `GET /api/agent/events/stream` — **SSE** live stream of new events
- Ring buffer (1000 events default), per-session JSONL field log
- Convenience helpers: think/decide/act/milestone/alert/observe/reflect
- 5 routes (incl. SSE)

**`telemetry.py`** — Stuck-Meter & Telemetry
- `GET /api/agent/telemetry?session_id=` — full snapshot
- `POST /api/agent/telemetry/position` — updates stuck-meter (auto-detects stuck at 12+ identical positions)
- `POST /api/agent/telemetry/action` — track success/failure
- `POST /api/agent/telemetry/battle` — won/lost
- `POST /api/agent/telemetry/blackout` — all-fainted counter
- `POST /api/agent/telemetry/health` — party HP
- `POST /api/agent/telemetry/reset` — reset counters
- Auto-emits `ALERT` event when stuck reaches 100
- 7 routes

**`collision.py`** — Collision Map
- `GET /api/spatial/collision?width=&height=&provider=` — full walkability grid
- `GET /api/spatial/grid` — labeled grid (A1..J9)
- `GET /api/spatial/grid/text` — plain text for prompt injection
- `PokemonRedCollisionProvider` registered by default (52 walkable tile IDs)
- Pluggable: subclass `CollisionMapProvider` + `register_collision_provider(...)` to add more games
- 3 routes

**Total: 33 new routes, all smoke-tested, all syntax-clean, no breakage to legacy.**

### Stage 3: Refactor server.py → Blueprints — IN PROGRESS

The new modules are already self-contained and additive. The legacy 6,778-line `server.py` still works and is now augmented with 33 new endpoints. Full blueprint split is the next step.

### Stage 4: Frontend Updates — PENDING
- Field Log panel (reasoning stream)
- Grid map view (SCREEN ⇄ GRID toggle)
- Party belt (all 6 slots compact)
- Sessions manager UI
- Telemetry/stuck-meter widget
- 3-tier objectives panel

### Stage 5: Verify, Docs, Commit — PENDING
- Update README.md, AGENTS.md, API-CONTRACT.md
- Create CHANGELOG.md entry
- Commit in logical chunks




---

## 📡 X / Twitter Research (added 2026-06-21)

Duckets asked me to also search X — "seen some enhancement programs on their as well for emulators on the web."

### Direct Adjacent Projects (from X)

**1. @jhhuang96 — Pokemon Crystal companion (March 2026)**
- PyBoy emulator + browser streaming + Claude-via-OpenClaw AI narrator
- Gives **real-time battle advice and commentary as you play**
- "Vibe coded in a day" — minimal harness
- Tweet: https://x.com/jhhuang96/status/2038153559574032657
- **Insight:** this is exactly the pattern of our project (PyBoy + browser + AI). Our **Field Log / events stream** is the next evolution — a full live broadcast with reasoning stream, not just narration.
- **Apply:** our existing `events.py` SSE stream + future Field Log UI already does this better.

**2. Mechanize GBA Eval (2026)**
- Frontier coding agents (Claude Fable 5 etc.) given 24h to write a complete GBA emulator from scratch
- Claude Fable 5 SOTA at 74.5% ROM compatibility
- Eval benchmark for emulator construction
- **Insight:** validation methodology we could borrow for "does our PyBoy agent still work after refactor?"
- **Apply:** write a small `eval_suite.py` with known PyBoy ROMs + known expected behavior (load → A → A → walk → dialog text appears).

**3. @skirano — Claude-built Game Boy emulator (Nov 2025)**
- GPT/Claude generated a fully functional Game Boy emulator (including SVG rendering of the hardware)
- **Insight:** shows LLM capability for emulator layer generation
- **Apply:** not directly applicable, but supports the case that "AI tooling for emulators" is a hot 2025-26 trend.

**4. @OdinLovis — Photo → playable Game Boy ROM tool (Jan 2026)**
- Open-source Windows tool that uses AI (via fal) to convert any photo → functional .gb/.gbc ROM
- Handles pixel art, animation, scrolling, music, sound effects
- **Insight:** demonstrates "AI gen content for retro emulators" as a category
- **Apply:** could add a "Generate Pokemon ROM" panel as a stretch feature.

### Adjacent / General AI Agent Dashboards (from X)

**5. AgentCommand by @MattPRD (Matt Schlicht)**
- Live dashboard showing 1000+ AI agents spawning, communicating, executing in real time
- Tracks revenue, deploys, code diffs, inter-agent conversations
- **Apply:** pattern reference for "mission control" UI for our multi-agent future

**6. OpenClaw Studio (open-source)**
- "Mission control center" for local AI agents
- Live WebSocket streaming, chat with agents, approval gates, job scheduling
- **Apply:** we're already targeting OpenClaw — this is a competitor/peer reference; we should check what UI patterns they use.

**7. agentcanvas (59★ GitHub)**
- Agent traces (Pydantic + Logfire) → interactive HTML diagrams
- Shows tools, nested agents, costs
- **Apply:** could be a stretch feature for our project — visualize the agent's decision tree from events.jsonl.

**8. LangSmith / Galileo / Langfuse / Arize AI** (top agent observability tools 2026)
- Real-time trace visualization, latency/cost breakdowns, error tracking
- Langfuse is open-source
- **Apply:** rather than build our own, we could optionally add Langfuse integration to the existing event stream for production observability.

### Architectural Insights from X

> "Success is ~90% harness/scaffolding and ~10% raw model capability."
> — common quote across multiple agent gameplay threads on X (e.g. @deforestpeg)

**Implication for our project:** the **agent_features** harness we just built (sessions, memory, events, telemetry, collision) is exactly the right scaffolding. The next 10x improvement comes from making the harness tighter (memory retrieval quality, event timing, stuck detection thresholds) — not from swapping models.

### Common Patterns Mentioned on X
- **SSE or WebSocket** for token-by-token + tool-result delivery (we do this in `events.py` ✅)
- **Progressive commit / preview mode** — show intermediate agent decisions immediately while buffering side-effects
- **Real-time UI elements** — live reasoning traces, slot streams, bundle tables, approval queues, cost tracking
- **Streaming architecture** — emit structured events from agent, stream via SSE/WS, render in React
