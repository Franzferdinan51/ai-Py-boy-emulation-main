# OpenClaw Compatibility Guide

**Last Updated:** March 19, 2026

This document captures the OpenClaw-native improvements and compatibility notes for the ai-Py-boy-emulation-main platform.

---

## Current Status: ✅ OpenClaw-Ready

The platform is fully compatible with OpenClaw agents. Key integration points work out of the box.

---

## MCP Server Selection

### Use: `generic_mcp_server.py`

```bash
mcporter add pyboy-emulator --stdio "python3 $REPO/ai-game-server/generic_mcp_server.py"
```

Where `$REPO` = `/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main`

**Why?** This is the fixed, LM Studio-compatible MCP wrapper that was repaired in recent work. It properly:
- Returns image content (not just metadata) for vision workflows
- Routes save/load to real backend endpoints
- Maintains parity with web UI behavior

### Legacy: `mcp_server.py`

The original MCP server at `ai-game-server/mcp_server.py` still exists but has known issues:
- Returns text metadata instead of actual image content for screenshots
- May have route alias mismatches

**Do not use** unless you specifically need legacy behavior.

---

## Runtime Ports

| Service | URL | Purpose |
|---------|-----|---------|
| Backend API | `http://localhost:5002` | Emulator control |
| WebSocket Stream | `ws://localhost:5003/` | Live screen feed |
| Frontend/Proxy | `http://localhost:5173` | Web UI |

---

## Quick Start for OpenClaw Agents

```bash
# 1. Start backend
cd $REPO/ai-game-server
PYTHONPATH="$PWD/src" python3 -c "from backend.server import app; app.run(host='0.0.0.0', port=5002, debug=False, threaded=True, use_reloader=False)"

# 2. Register MCP server
mcporter add pyboy-emulator --stdio "python3 $REPO/ai-game-server/generic_mcp_server.py"

# 3. Verify
mcporter list | grep pyboy

# 4. Load ROM
mcporter call pyboy-emulator.emulator_load_rom rom_path="/path/to/rom.gb"

# 5. Get screen for vision analysis
mcporter call pyboy-emulator.emulator_get_frame include_base64=true
```

---

## Key Backend Routes for Agents

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/load_rom` | POST | Load ROM file |
| `/api/game/state` | GET | Current game state |
| `/api/game/button` | POST | Press single button |
| `/api/action` | POST | Execute action sequence |
| `/api/screen` | GET | Get current screen |
| `/api/stream` | GET | SSE live stream |
| `/api/save_state` | POST | Save emulator state |
| `/api/load_state` | POST | Load emulator state |
| `/api/party` | GET | Pokemon party data |
| `/api/inventory` | GET | Inventory data |
| `/api/spatial/position` | GET | Player position |
| `/api/spatial/minimap` | GET | Minimap image |

---

## Vision Workflow

OpenClaw agents should use the Bailian vision models (free, unlimited):

1. **Get screen:** `emulator_get_frame(include_base64=true)`
2. **Analyze:** Use `bailian/kimi-k2.5` for screen understanding
3. **Act:** Use `emulator_press_sequence(sequence="...")` to control

---

## Known Issues (Resolved)

The following issues were fixed in recent work:

1. **MCP screenshot returning metadata instead of image** → Fixed in `generic_mcp_server.py`
2. **Save/load routing to wrong endpoints** → Now routes to `/api/save_state` and `/api/load_state`
3. **Frontend null-safety crashes** → Added safe defaults and null guards
4. **Route alias mismatches** → Compatibility routes added

---

## OpenClaw-Native Files

These files are designed for OpenClaw agent consumption:

| File | Purpose |
|------|---------|
| `AGENTS.md` | Agent invariants and workflow |
| `TOOLS.md` | Runtime ports, routes, local commands |
| `skills/pyboy-platform/SKILL.md` | Platform thinking guide |
| `ai-game-server/API-CONTRACT.md` | Backend response shapes |

---

## Model Integration

The platform supports multiple AI providers. Current recommendations:

| Model | Use Case | Cost |
|-------|----------|------|
| `bailian/kimi-k2.5` | Vision/screen analysis | FREE |
| `bailian/MiniMax-M2.5` | General tasks | FREE |
| `bailian/qwen3.5-plus` | Complex reasoning | Quota |

---

## Migration Notes

If you were using an older setup:

1. **MCP Server:** Switch from `mcp_server.py` to `generic_mcp_server.py`
2. **Routes:** Backend API is stable at `localhost:5002`
3. **Tools:** MCP tool names are consistent (e.g., `emulator_load_rom`, not `load_rom`)
4. **ROM Path:** Use absolute paths to your actual ROM files

---

## Troubleshooting

### "Emulator not initialized"
- Ensure backend is running: `curl http://localhost:5002/api/game/state`
- Load a ROM first: `emulator_load_rom`

### MCP server not responding
- Verify registration: `mcporter list | grep pyboy`
- Check backend is running on port 5002
- Restart MCP: `mcporter remove pyboy-emulator && mcporter add...`

### Vision not working
- Use `bailian/kimi-k2.5` (free, vision-capable)
- Ensure `include_base64=true` in `emulator_get_frame`

---

## Related Documentation

- `README.md` - Project overview
- `AGENTS.md` - Agent-first invariants
- `TOOLS.md` - Tool and route reference
- `skills/pyboy-platform/SKILL.md` - Platform architecture
- `ai-game-server/API-CONTRACT.md` - API response shapes