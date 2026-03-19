# TOOLS.md

Repo-local operator/tooling notes for `ai-Py-boy-emulation-main`.

## Runtime ports
- Frontend/proxy: `http://localhost:5173`
- Backend API: `http://localhost:5002`
- WebSocket stream: `ws://localhost:5003/`

## Useful local commands
### Backend
```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server
PYTHONPATH=/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/src \
python3 -c "from backend.server import app; app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)"
```

### Frontend build
```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-assistant
npm run build
```

### Proxy/frontend
```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-assistant
python3 proxy-server.py
```

## Important backend routes
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

### Spatial / MCP-aware UI
- `GET /api/spatial/position`
- `GET /api/spatial/minimap`
- `GET /api/spatial/npcs`
- `GET /api/spatial/strategy`

## MCP / LM Studio
LM Studio config currently references:
- `ai-game-server/generic_mcp_server.py`

Important MCP save/load tools must hit:
- `/api/save_state`
- `/api/load_state`

Not old placeholder routes.

## Known compatibility patterns
### Good practice
- return stable JSON shapes
- empty arrays instead of omitted arrays
- zero/default strings instead of undefined where UI expects formatting
- include `loaded`/`rom_loaded` style status when helpful

### Common failure modes
- frontend `.trim()` on null
- frontend `.length` on undefined arrays
- `toLocaleString()` on undefined counts
- route alias missing even though canonical route exists
- LM Studio wrapper still pointing to legacy backend routes
- placeholder success route masking real broken behavior

## Save/load verification notes
Important: endpoint success alone is not enough.
Prefer to verify with:
- memory value restoration
- byte count returned by save/load
- screen hash or scene change when practical

## Streaming notes
Prefer simple, stable streaming over clever streaming.
Avoid:
- multiple tick owners
- hidden race conditions between background emulation and SSE/WS capture
- fragile executor logic unless truly needed

## Current repo direction
This project is now an **agent/AI-first platform**, not just a UI wrapper.
Primary use cases:
- OpenClaw-driven gameplay
- LM Studio + MCP gameplay
- operator dashboard for AI state and world state
- autonomous planning via MCP-aware panels
