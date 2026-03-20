---
name: pyboy-platform
description: Operate and extend the ai-Py-boy-emulation-main project as an agent-first Game Boy platform. Use when working on the PyBoy web UI, backend emulator API, streaming, save/load, MCP/LM Studio integration, minimap/NPC/strategy panels, frontend/backend compatibility routes, or stable JSON contracts for Game Boy gameplay automation.
---

# PyBoy Platform

Use this skill when modifying the repo as a platform for AI agents, not just as a game UI.

## Core rule
Treat the platform as three coupled layers:
1. frontend/proxy (`ai-game-assistant/`)
2. backend/emulator API (`ai-game-server/`)
3. MCP wrapper (`ai-game-server/generic_mcp_server.py`)

If one layer changes shape, check the others.

## Start here
Read these files first as needed:
- `AGENTS.md` for project invariants
- `TOOLS.md` for route/port/runtime notes
- `ai-game-server/API-CONTRACT.md` for backend payload shapes

## What to protect
Do not break these without deliberate coordinated changes:
- ROM load flow
- stream continuity
- save/load restore behavior
- frontend null safety
- MCP save/load/button path parity with web UI

## Recommended workflow
1. Identify the exact caller
   - web UI
   - proxy
   - backend route
   - LM Studio / MCP tool
2. Confirm actual response shape
3. Fix backend contract first
4. Fix frontend rendering second
5. If LM Studio is involved, fix `generic_mcp_server.py` too
6. Verify with a loaded ROM

## Stable response design
For UI-facing endpoints:
- always return objects, not bare primitives
- return empty arrays instead of omitting arrays
- use safe defaults for counters and strings
- include `timestamp`
- include `loaded` or equivalent status when useful

## Save/load expectations
Save/load must be real, not placeholder success.
Verify with at least one of:
- restored memory values
- restored screen hash
- restored gameplay-relevant state after a change

## Streaming expectations
Prefer one tick owner.
Avoid:
- multiple background tick loops
- screen capture loops that mutate emulator state unexpectedly
- hidden race conditions between live loop and stream loop

## MCP expectations
When a feature works in the web UI but not LM Studio, inspect:
- `generic_mcp_server.py`
- tool-to-route mapping
- old route aliases still being used

## UI direction
When improving the operator UI, prefer:
- clean minimap/world panels
- party/inventory/memory with scrollable sections
- agent/strategy state that maps to real backend payloads
- null-safe rendering everywhere

## Files most likely to matter
- `ai-game-assistant/App.tsx`
- `ai-game-assistant/src/WebUiApp.tsx`
- `ai-game-assistant/src/components/*.tsx`
- `ai-game-assistant/services/apiService.ts`
- `ai-game-server/src/backend/server.py`
- `ai-game-server/src/backend/emulators/pyboy_emulator.py`
- `ai-game-server/generic_mcp_server.py`

## When done
Update repo-facing docs if the contract changed:
- `README.md`
- `TOOLS.md`
- `AGENTS.md`
- `ai-game-server/API-CONTRACT.md`

## Agent Tools (March 2026)
The platform has dedicated agent-first endpoints for AI gameplay:
- `/api/agent/context` — Full state snapshot for decision making
- `/api/agent/mode` — Game mode detection (exploration, battle, menu, dialogue)
- `/api/agent/act` — Action + observation in single call
- `/api/agent/dialogue` — Text box state
- `/api/agent/menu` — Menu state

MCP tools in `generic_mcp_server.py` map to these routes:
- `get_agent_context`, `get_game_mode`, `act_and_observe`, `get_dialogue_state`, `get_menu_state`

These enable agents to play autonomously without relying on vision alone.
