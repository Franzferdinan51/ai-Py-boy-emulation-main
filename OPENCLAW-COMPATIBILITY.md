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

## LM Studio Vision Compatibility Guide

### The Image Attachment Problem

**Symptom:** The `get_screen` or `screenshot` MCP tool returns successfully and includes an image, but the agent/model cannot "see" or use the image for reasoning.

**This is NOT a bug in the MCP server.** The `generic_mcp_server.py` correctly returns `ImageContent` alongside `TextContent`. The limitation is in how LM Studio chat interfaces and models handle these images.

### Why Images May Not Work

#### 1. **Non-Vision Model Selected** (Most Common)

If the thinking model is NOT a vision-capable model, it literally cannot process images even if they're attached:

| Model Type | Examples | Can See Images? |
|------------|----------|-----------------|
| Vision-capable | `qwen3-vl-8b`, `qwen3-vl-4b`, `jan-v2-vl-high`, `glm-4.6v-flash` | ✅ Yes |
| Text-only | `qwen3.5-35b-a3b`, `qwen3.5-27b`, `glm-4.7-flash` | ❌ No |

**Fix:** Ensure BOTH thinking AND vision model slots use vision-capable models.

#### 2. **LM Studio Chat Interface Limitations**

Some LM Studio versions or chat configurations do not properly forward image content from tool responses to the model:

- Tool returns `ImageContent(data=base64, mimeType="image/jpeg")` 
- Chat UI shows "image attached" or success message
- But the actual chat message to the model doesn't include the image bytes

**Symptoms:**
- Tool output shows success with image metadata
- Model response ignores the screen content
- Model says "I can't see the image" or provides generic responses

#### 3. **Thinking Mode Issues**

Some reasoning/thinking models have known issues with multimodal input:
- The model may strip image content when in "thinking" mode
- Reasoning tokens may interfere with vision processing

**Fix:** Disable thinking/reasoning for vision tasks, or use a non-thinking vision model.

#### 4. **API Endpoint Misconfiguration**

The LM Studio server MUST be accessed at the chat completions endpoint:
- ✅ Correct: `http://localhost:1234/v1/chat/completions`
- ❌ Wrong: `http://localhost:1234/` (missing `/v1/chat/completions`)

The MCP server handles this internally, but ensure LM Studio server is running with the `/v1` prefix.

#### 5. **Context Length Limits**

Vision models have significant overhead per image:
- A single Game Boy screenshot (160x144) is small, but still adds ~10KB
- If the model is near its context limit, it may silently drop images

**Fix:** Keep conversation history shorter when using vision, or use smaller context models.

### Recommended LM Studio Settings for Vision

#### Option A: High Quality Vision
```
Endpoint: http://localhost:1234/v1
Thinking Model: qwen3-vl-8b  (or qwen/qwen3-vl-8b)
Vision Model: qwen3-vl-8b     (same model handles both)
```
**Best for:** Complex games requiring strategic reasoning + vision

#### Option B: Fast Vision
```
Endpoint: http://localhost:1234/v1
Thinking Model: qwen3-vl-4b  (fast vision-capable)
Vision Model: qwen3-vl-4b
```
**Best for:** Real-time gameplay, quick decisions

#### Option C: Two-Model Setup (Advanced)
```
Endpoint: http://localhost:1234/v1
Thinking Model: qwen3.5-35b-a3b  (best reasoning)
Vision Model: qwen3-vl-8b        (sees the screen)
```
**How it works:**
1. Use vision model to analyze screen via MCP tool
2. Pass text summary to thinking model for decisions
3. This requires custom prompt engineering

### Alternative: Use Bailian for Vision

Instead of LM Studio for vision, use the free Bailian vision models:

```python
# In vision_analysis.py or custom tool
from bailian import VisionClient
client = VisionClient(api_key="your-key")
result = client.analyze(image_base64, prompt="What Pokemon game state is shown?")
```

Bailian models (`bailian/kimi-k2.5`) are:
- ✅ FREE unlimited vision
- ✅ Reliable image handling
- ✅ Better at game state understanding

### Verification: Is Your Vision Working?

Test if vision is actually working:

1. Call `get_screen` or `screenshot` MCP tool
2. Check response contains BOTH:
   - `TextContent` with success message
   - `ImageContent` with actual base64 data
3. Ask the model: "Describe what you see on the game screen"
4. If it describes generic "Game Boy screen" without specifics → vision NOT working
5. If it describes Pokemon, menus, positions → vision IS working

### Code-Side Improvement: Explicit Vision Status

The MCP server now includes a clear message about image attachment. To make this even clearer, you can check tool output:

```python
# In generic_mcp_server.py, get_screen handler:
return [
    TextContent(type="text", text=json.dumps({
        "success": True,
        "has_image": True,  # Explicit flag
        "image_size_kb": len(result.get("image", "")) // 1024,
        "vision_model_note": "Image attached as ImageContent. Ensure your LM Studio model is vision-capable (qwen3-vl-8b, qwen3-vl-4b, jan-v2-vl-*). Text-only models cannot see images."
    })),
    ImageContent(type="image", data=result.get("image"), mimeType="image/jpeg")
]
```

This explicit warning in the text response helps diagnose configuration issues.

---

## Related Documentation

- `README.md` - Project overview
- `AGENTS.md` - Agent-first invariants
- `TOOLS.md` - Tool and route reference
- `skills/pyboy-platform/SKILL.md` - Platform architecture
- `ai-game-server/API-CONTRACT.md` - API response shapes
- `docs/LM-STUDIO-QUICKSTART.md` - LM Studio setup guide