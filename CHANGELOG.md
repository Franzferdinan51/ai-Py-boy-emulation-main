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

## Future

### Planned (next phases)
- Refactor `server.py` (6,778 lines, 92 routes) into Flask blueprints by domain
- Frontend: Field Log panel, grid map view, party belt, sessions manager, telemetry widget
- PyBoy 2.x API audit (verify we're using `screen.image`, `pyboy.memory` correctly)
- OpenClaw-native improvements: dual-model vision+planning coordination
- Memory heatmaps, action timeline / replay, RAG over game knowledge

### Backwards compatibility
- All new endpoints are additive — no legacy route was renamed or removed
- `FLASK_ENV` env var reads still work (Flask 3 removed `FLASK_ENV` from Flask itself, but the code only reads it as a custom env var)
- Frontend code unchanged; no breaking changes to existing components

