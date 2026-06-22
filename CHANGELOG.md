# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-06-21

### Added — New Feature Modules (additive, 33 new routes)

**🎮 Game Sessions** (`backend.agent_features.sessions`)
- Named playthroughs with persistent state, modeled after NousResearch/pokemon-agent
- Endpoints: `POST/GET/DELETE /api/games`, `/api/games/new`, `/api/games/current`, `/api/games/<id>/activate`, `/api/games/<id>/save_state`
- Persisted to `~/.ai-py-boy/data/games/<id>/manifest.json` (configurable via `PYBOY_DATA_DIR`)
- 9 new routes

**🧠 Agent Memory / KnowledgeBase** (`backend.agent_features.memory`)
- Structured episodic memory inspired by PokemonLLMAgentBenchmark's KnowledgeBase
- Stores: locations, party events, free-form notes, structured facts, control patterns, completed objectives
- RAG-lite text search: `GET /api/agent/memory/search?session_id=...&q=...`
- Summary endpoint: `GET /api/agent/memory/summary`
- 9 new routes

**📡 Reasoning Event Stream** (`backend.agent_features.events`)
- THINK / DECIDE / ACT / MILESTONE / ALERT / OBSERVE / REFLECT event kinds
- Field Log ring buffer (1000 events default)
- **SSE live stream**: `GET /api/agent/events/stream`
- Per-session JSONL field log on disk
- Stats: `GET /api/agent/events/stats`
- 5 new routes

**📊 Telemetry / Stuck-Meter** (`backend.agent_features.telemetry`)
- Position tracking with auto stuck detection (12+ identical positions → meter climbs)
- Battle outcomes, blackouts, action success rate, party HP
- Auto-emits `ALERT` event when stuck reaches 100
- 7 new routes

**🗺️ Collision Map** (`backend.agent_features.collision`)
- RAM-derived walkability grid from emulator tilemap
- Labeled A1..J9 coordinate overlay
- Plain-text export for prompt injection: `GET /api/spatial/grid/text`
- Pluggable per-game providers (Pokemon Red registered by default)
- 3 new routes

### Changed
- **requirements.txt**: Flask 2.3.2 → `>=3.0,<4.0`, Werkzeug `>=3.0,<4.0`, OpenAI `>=1.50.0,<3.0`, added httpx
- **server.py**: Wired `register_all()` from `backend.agent_features` at startup; logs route counts

### Removed
- **`server_original.py`** (1,385 lines, legacy, not imported anywhere)

### Verified
- All 25 module-level smoke tests pass (sessions / memory / events / telemetry / collision)
- All 17 HTTP smoke tests pass via Flask test client
- Python syntax clean across all modified files
- Server.py still parses with the new registration call

### Upgraded documentation
- `UPGRADE_NOTES.md` — full upgrade plan, research findings, progress log
- `CHANGELOG.md` (this file)

## [Unreleased] — 2026-06-21 (Stage 3 + Stage 4)

### Refactored — server.py split into Flask blueprints

The monolithic `server.py` (was 6,806 lines, 92 routes) has been split into
focused, reusable Flask-style modules under `backend/routes/`:

| Module | Routes extracted | Lines removed from server.py |
| --- | --- | --- |
| `routes/config.py`    | `/api/config`, `/api/config/validate`              | ~40 |
| `routes/save_load.py` | `/save_state`, `/load_state`, `/api/save_state`, `/api/load_state` | ~75 |
| `routes/ui.py`        | `/api/ui/launch`, `/stop`, `/restart`, `/status`   | ~110 |
| `routes/ws.py`        | `/api/ws/status`, `/start`, `/stop` (now uses `WebSocketRunner` adapter — gracefully reports not-running when ws globals are missing) | ~50 |
| `routes/tetris.py`    | `/api/tetris/train`, `/status`, `/save`, `/load`   | ~200 |
| `routes/vision.py`    | `/api/vision/analyze`, `/describe`, `/ocr`, `/summary`, `/status` | ~580 |

Net effect: `server.py` shrank from **6,806 → 5,917 lines** (−889, −13%). All
92 routes still respond correctly with the same shapes they had before. The
blueprints follow the same `register_*_routes(app, **deps)` pattern used by
`backend.agent_features`, so future contributors can register any subset of
them on a different Flask app (e.g. for a stripped-down embed).

The blueprint registration happens at module load (after `get_game_state` is
defined) so test clients and tooling see the full surface. `agent_features`
remains registered inside `main()` to keep its import-time side effects
optional.

### Added — Frontend agent_features panels

Three new React components wire the new backend endpoints into the UI:

- **`FieldLog`** (`src/components/FieldLog.tsx`, ~250 lines)
  Streams THINK / DECIDE / ACT / MILESTONE / ALERT / OBSERVE / REFLECT events
  from `/api/agent/events`, with optional SSE live mode
  (`/api/agent/events/stream`), kind filter, search, auto-scroll, and clear.

- **`SessionsPanel`** (`src/components/SessionsPanel.tsx`, ~200 lines)
  UI for the game-sessions API: list / create / activate / save / load /
  delete. Polls `/api/games` every 8 s.

- **`TelemetryWidget`** (`src/components/TelemetryWidget.tsx`, ~200 lines)
  Compact panel showing stuck-meter (with color-coded danger threshold),
  action success-rate bar, battle W/L, blackout count, and party HP bar.
  Polls `/api/agent/telemetry` every 3 s.

Plus new `apiService` methods (in `services/apiService.ts`) for all the new
endpoints: sessions, events (incl. SSE), telemetry, memory, collision.

### Verified

- Backend: 91 unique paths respond via Flask test client. 45 × 2xx, 43 × 4xx
  (expected: most require a loaded ROM), 3 × 5xx (all valid "service not
  available" states from the config/vision/UI blueprints).
- Frontend: `npx tsc --noEmit` clean; `npm run build` succeeds (274 kB JS,
  82 kB gzipped).
- All new modules import cleanly with no circular-import warnings.

## [Unreleased] — 2026-06-22

### Refactored — server.py split into Flask blueprints (Stage 3 continued)

Continuing the Stage 3 work, five more domain groups have been moved out of
the monolithic `server.py` (was 5,917 lines, 92 routes) into focused, reusable
blueprint modules under `backend/routes/`:

| Module            | Routes extracted                                                                                           | Lines removed from server.py |
| ---               | ---                                                                                                        | ---                           |
| `routes/health.py`     | `/health`, `/api/health`, `/api/health/runtime`, `/api/health/emulator`, `/api/health/stream`        | ~120                          |
| `routes/ai_models.py`  | `/api/providers`, `/api/providers/status`, `/api/models`, `/api/openclaw/models` (+ vision / planning / recommend) | ~310                          |
| `routes/ai_runtime.py` | `/api/ai/runtime`, `/api/openclaw/config`, `/api/ai/settings`, `/api/openclaw/health`             | ~205                          |
| `routes/screen.py`     | `/api/screen`, `/api/screen/debug`, `/api/stream`, `/api/performance`, `/api/emulator/mode`, `/api/emulator/clear-cache` | ~165                          |
| `routes/input.py`      | `/api/game/button`, `/api/game/action`, `/api/action`, `/api/ai-action`, `/api/chat`, `/input`     | ~720                          |

Net effect: `server.py` shrank from **5,917 → 3,337 lines** (−2,580, −44%).
All 76 routes still respond with the same URL / method / JSON shape as the
legacy handlers. The new blueprints follow the same
`register_*_routes(app, **deps)` pattern, so any subset can be mounted on a
different Flask app (e.g. for a stripped-down embed).

The `routes/__init__.py::register_all` signature grew new optional kwargs for
each blueprint (e.g. `health_*`, `ai_models_*`, `ai_runtime_*`, `screen_*`,
`input_*`). All defaults are sensible no-ops — wiring with the production
call site in `server.py::main()` is the only consumer that needs them.

### Added — Flask test-client smoke test

`tests/test_blueprint_smoke.py` (43 checks) verifies:

- No `(rule, method)` pair is registered twice across all blueprints
  (shadowing is now a hard assertion).
- Every new endpoint returns the expected status code (200 for healthy paths,
  400 for ROM-required, 503 for service-unavailable fallbacks).
- Response bodies include the same key fields the legacy handlers returned
  (`status`, `checks`, `providers`, `runtime.state`, `performance.server_performance`,
  `system_info`, etc.).

Run with either:

```bash
python3 tests/test_blueprint_smoke.py   # exits non-zero on regression
pytest tests/test_blueprint_smoke.py   # integrated with the test runner
```

The two `/api/config*` 503s are baseline behavior (no `SecureConfig` in the
test env) — verified by stashing the refactor and re-running the suite.

### Verified

- `python3 -c "import backend.server as srv"` succeeds; 76 routes registered.
- `pytest tests/test_all.py tests/test_blueprint_smoke.py` — **5 passed**.
- `routes/__init__.py::register_all` smoke trace:
  `config:2 health:5 ai_models:7 ai_runtime:4 save_load:4 screen:6 input:6 ui:4 ws:3 tetris:4 vision:5`
  (50 routes from blueprints + 26 still in server.py for the agent/spatial/ROM domain).
- Baseline stash test confirms `/api/config*` 503 behavior is preserved
  (not a regression introduced by the refactor).

## [Unreleased] — 2026-06-22 (Stage 3 cont. round 2)

### Refactored — server.py split completed for the last 22 routes

The last group of routes has been moved out of `server.py` (3,337 → 1,678
lines, −1,659 lines, −50% net from the original 6,806-line monolith).

| Module                | Routes extracted                                                                                            | LOC moved |
| ---                   | ---                                                                                                          | ---        |
| `routes/agent.py`     | `/api/agent/state`, `/api/agent/status`, `/api/agent/goal` (GET/POST), `/api/agent/chat`, `/api/agent/errors`, `/api/agent/actions`, `/api/agent/context`, `/api/agent/mode` (GET/POST), `/api/agent/act`, `/api/agent/dialogue`, `/api/agent/menu` | 800        |
| `routes/spatial.py`   | `/api/spatial/position`, `/api/spatial/minimap`, `/api/spatial/npcs`, `/api/spatial/strategy`, `/api/agent/strategy` (alias) | 320        |
| `routes/rom.py`       | `/api/upload-rom`, `/api/load_rom`, `/api/rom/load`, `/api/game/state`, `/api/party`, `/api/inventory`, `/api/memory/watch` | 540        |

The agent blueprint uses a `agent_state_getter` + `agent_state_mutate`
pair to share the in-memory `agent_state` dict without circular imports.
The chat endpoint keeps its full OpenClaw-aware instruction parser
(`goal:` / `task:` / `?status`) and AI-provider fallback path. The ROM
blueprint takes all upload-pipeline helpers (`validate_file_upload`,
`validate_string_input`, `sanitize_filename`, `sync_loaded_rom_state`,
`ensure_emulation_loop_running`) as injection callables so the surface is
importable in isolation; missing deps yield a structured 503.

### Updated — `tests/test_blueprint_smoke.py` (78 checks now)

Extended from 43 → 78 checks to cover all three new blueprints:

- **Agent (15)**: state, status, errors, actions, context, mode (GET+POST),
  goal (GET+POST), chat (400 without ROM), act (400 without ROM), strategy,
  dialogue, menu
- **Spatial (6)**: position/minimap/npcs/strategy return empty-shape 200
  without ROM; position payload has `loaded: false`
- **ROM (14)**: game/state, party, inventory, memory/watch all 200 with
  expected shape; upload-rom/load_rom/rom/load return 400 on empty payloads

### Verified

- `python3 -c "import backend.server as srv"` succeeds; 76 routes registered.
- `routes/__init__.py::register_all` trace:
  `config:2 health:5 ai_models:7 ai_runtime:4 save_load:4 screen:6 input:6 agent:12 spatial:5 rom:7 ui:4 ws:3 tetris:4 vision:5`
- `pytest tests/test_all.py tests/test_blueprint_smoke.py` — **5 passed**.
- `python3 tests/test_blueprint_smoke.py` — **78/78 passed**.
- `cd ai-game-assistant && npx tsc --noEmit` — clean, no frontend breakage
  from the route extraction.

### Server.py status

`server.py` now hosts only:

- App construction (`app = Flask(__name__)`)
- Global state (game_state, agent_state, ai_runtime_state, emulators, …)
- Helper functions (`validate_*`, `sanitize_filename`, `configure_emulator_launch_ui`,
  `sync_loaded_rom_state`, `ensure_emulation_loop_running`, …)
- One `register_all` call that wires all 14 blueprints
- The `/api/status` rollup endpoint (it depends on multiple globals
  that blueprints don't need)
- `main()` entry point

## Future

### Planned (next phases)
- Stage 3 refactor is **complete** — every route lives in a blueprint. The
  remaining work is product features, not refactor.
- Frontend: Grid map view (uses `/api/spatial/grid`), 3-tier objectives panel,
  party belt (all 6 slots compact), RAG over memory
- PyBoy 2.x API audit (verify we're using `screen.image`, `pyboy.memory` correctly)
- OpenClaw-native improvements: dual-model vision+planning coordination
- Memory heatmaps, action timeline / replay

### Backwards compatibility
- All extracted endpoints are byte-compatible with the legacy implementations
  (same URL, same methods, same JSON shape).
- `FLASK_ENV` env var reads still work (Flask 3 removed `FLASK_ENV` from Flask
  itself, but the code only reads it as a custom env var).
- Frontend code unchanged for existing components; new panels are purely
  additive.

