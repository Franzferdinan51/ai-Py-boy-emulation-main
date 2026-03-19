# AGENTS.md

Agent-first guide for working in `ai-Py-boy-emulation-main`.

## Mission
Build and operate an AI/agent-first Game Boy platform on top of PyBoy, with:
- stable streaming
- reliable save/load
- clean frontend/backend contracts
- MCP tooling for autonomous play
- OpenClaw / LM Studio friendly workflows

## Project layout
- `ai-game-assistant/` — frontend + proxy UI
- `ai-game-server/` — Flask backend, streaming, emulator control, MCP bridge helpers
- `ai-game-server/generic_mcp_server.py` — LM Studio / MCP wrapper
- `skills/` — repo-local AgentSkills
- `docs/` — verification and platform notes
- `saves/` — emulator save artifacts when used

## Canonical runtime
Preferred local stack:
1. backend on `:5002`
2. websocket stream on `:5003`
3. proxy/frontend on `:5173`

## Core invariants
Do not casually break these again:
1. **One authoritative emulation tick owner**
   - Avoid multiple loops ticking PyBoy concurrently.
   - Streaming and emulator stepping must not race.
2. **Frontend/backend contract stability**
   - If the UI expects `party`, do not return `party_pokemon`.
   - Favor stable shapes with empty arrays/defaults instead of missing fields.
3. **Save/load must be real**
   - `/api/save_state` and `/api/load_state` must call actual emulator save/load logic.
   - Do not return fake success placeholders.
4. **Null-safe frontend rendering**
   - Assume strings, arrays, and counters may be null/undefined during startup.
   - Guard `trim`, `length`, `toLocaleString`, and nested property access.
5. **macOS-safe PyBoy operation**
   - Avoid UI-launch assumptions that require Windows-only flags.
   - Prefer documented PyBoy behavior and stable headless/null operation.

## Agent workflow
When changing behavior:
1. identify the exact caller and route/tool contract
2. patch backend shape first
3. patch frontend rendering second
4. verify with a real running ROM
5. if save/load or streaming changed, test those explicitly
6. update docs if the contract changed

## Critical endpoints
Current important endpoints include:
- `POST /api/load_rom`
- `GET /api/game/state`
- `POST /api/game/button`
- `POST /api/action`
- `GET /api/screen`
- `GET /api/stream`
- `POST /api/save_state`
- `POST /api/load_state`
- `GET /api/party`
- `GET /api/inventory`
- `GET /api/memory/watch`
- `GET /api/spatial/position`
- `GET /api/spatial/minimap`
- `GET /api/spatial/npcs`
- `GET /api/spatial/strategy`

## MCP/LM Studio notes
`generic_mcp_server.py` must route tool calls to the real backend endpoints.
Pay special attention to:
- `save_state`
- `load_state`
- `quick_save`
- `quick_load`
- button tools

If web UI behavior works but LM Studio behavior fails, inspect the MCP wrapper first.

## Frontend expectations
The UI now expects agent-first operator panels and stable payloads.
Keep these areas healthy:
- game screen / stream surface
- runtime state tabs
- minimap / npc / strategy panels
- settings and provider controls

## Documentation expectations
When major behavior changes, update at least one of:
- `README.md`
- `TOOLS.md`
- `docs/API-CONTRACT.md` or equivalent contract notes
- repo-local skill files under `skills/`

## Preferred style for fixes
- small, testable, contract-driven
- avoid fake placeholders
- avoid frontend-specific hacks unless backend contract is already correct
- prefer compatibility aliases over breaking route renames

## Before claiming “fixed”
Verify as applicable:
- ROM loads successfully
- stream stays alive for multiple frame events
- UI does not collapse on render
- required compatibility routes return 200
- save/load performs real restore behavior
- MCP path matches web path for core actions
