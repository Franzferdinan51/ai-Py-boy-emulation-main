# AI GameBoy Emulator

An **agent-first Game Boy platform** built on PyBoy, with:
- web UI + proxy frontend
- stable backend emulator API
- MCP / LM Studio integration
- live streaming
- real save/load support
- operator panels for party, inventory, memory, minimap, NPCs, and strategy

This project is no longer just a demo UI. It is intended to be a working foundation for **AI agents that can observe, reason, and act inside Game Boy games**.

## What it includes
- **Frontend**: `ai-game-assistant/`
- **Backend**: `ai-game-server/`
- **MCP wrapper**: `ai-game-server/generic_mcp_server.py`
- **Repo-local skills**: `skills/`
- **Verification/docs**: `docs/`

## Runtime ports
- Frontend/proxy: `http://localhost:5173`
- Backend API: `http://localhost:5002`
- WebSocket stream: `ws://localhost:5003/`

## Quick start
### 1. Start backend
```bash
cd ai-game-server
PYTHONPATH="$PWD/src" python3 -c "from backend.server import app; app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)"
```

### 2. Start frontend/proxy
```bash
cd ai-game-assistant
python3 proxy-server.py
```

### 3. Open UI
```text
http://localhost:5173
```

## Key API routes
### Core emulator
- `POST /api/load_rom`
- `GET /api/game/state`
- `POST /api/game/button`
- `POST /api/game/action`
- `POST /api/action`
- `GET /api/screen`
- `GET /api/stream`

### Save/load
- `POST /api/save_state`
- `POST /api/load_state`

### UI compatibility
- `GET /api/party`
- `GET /api/inventory`
- `GET /api/memory/watch`
- `GET /api/agent/status`
- `POST /api/agent/mode`
- `POST /api/ai/runtime`
- `POST /api/openclaw/config`
- `GET /api/openclaw/health`

### Spatial / AI panels
- `GET /api/spatial/position`
- `GET /api/spatial/minimap`
- `GET /api/spatial/npcs`
- `GET /api/spatial/strategy`

### Agent Tools (AI Gameplay)
These endpoints are designed for AI agents to understand game state and make autonomous decisions:
- `GET /api/agent/context` — Full game state (position, party, inventory, battle, recommendations)
- `GET /api/agent/mode` — Current game mode (exploration, battle, menu, dialogue)
- `POST /api/agent/act` — Execute action and observe result in one call
- `GET /api/agent/dialogue` — Current dialogue/text box state
- `GET /api/agent/menu` — Current menu state

### Sound control
- `GET /api/sound/status`
- `POST /api/sound/enable`
- `POST /api/sound/volume`
- `POST /api/sound/output`
- `GET /api/sound/buffer`

## MCP / LM Studio
LM Studio can connect through:
- `ai-game-server/generic_mcp_server.py`

The wrapper exposes emulator control, save/load, button presses, screen capture, memory access, and game-state helpers.

If web UI behavior and LM Studio behavior diverge, inspect the MCP wrapper first.

## Agent-first files
If you are extending this repo for agents, read:
- `AGENTS.md` — project invariants and agent workflow
- `TOOLS.md` — local runtime/route/tool notes
- `skills/pyboy-platform/SKILL.md` — repo-local agent skill
- `ai-game-server/API-CONTRACT.md` — endpoint response shapes
- `docs/SOUND-SUPPORT.md` — sound configuration and platform caveats

## Important engineering rules
1. **One tick owner**
   - avoid race conditions between stream loops and background emulator loops
2. **Stable payloads**
   - empty arrays/defaults are better than missing fields
3. **Real save/load**
   - never return placeholder success for state restore
4. **Null-safe UI**
   - assume data may be partial during startup
5. **MCP parity**
   - LM Studio and web UI should use the same real backend behaviors

## Verified areas
This repo has had focused repair work around:
- stream stability
- ROM load compatibility
- frontend/backend route compatibility
- save/load API wiring
- LM Studio save/load tool routing
- operator UI null safety

See `docs/` for verification notes and results.

## Repo direction
The intended direction is:
- autonomous gameplay via agents
- clean operator UI for world state and planning
- MCP-aware panels (world, NPCs, strategy, memory)
- reliable backend contracts that support both UI and agents

## Next-step friendly areas
Good extension targets:
- richer minimap/world model
- stronger NPC/interactable detection
- better strategy recommendations
- cleaner backend contract normalization
- stronger gameplay-state save/load proofs
