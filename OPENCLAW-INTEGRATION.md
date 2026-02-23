# OpenClaw Integration Guide

**Status:** ✅ Phase 1 Complete (2026-02-23)  
**Repository:** ai-Py-boy-emulation-main  
**Integration Point:** OpenClaw MCP + Skills

---

## What's Been Added

### 1. MCP Server (`ai-game-server/mcp_server.py`)
Exposes emulator controls as MCP tools for OpenClaw agents:
- `emulator_load_rom` - Load ROM files
- `emulator_press_button` - Press controller buttons
- `emulator_get_frame` - Capture screenshots (base64 PNG)
- `emulator_get_state` - Get emulator status
- `emulator_tick` - Advance emulation

### 2. OpenClaw Skill (`skills/game-emulation/SKILL.md`)
Documentation and usage guide for OpenClaw agents to discover and use the emulator.

### 3. Vision Bridge (`ai-game-server/vision_bridge.py`)
Utility for capturing frames and integrating with vision models for gameplay analysis.

### 4. Gamer Agent Spawner (`tools/spawn-gamer-agent.sh`)
Helper script to spawn sub-agents for automated gameplay.

---

## Architecture

```
[ OpenClaw Main Agent ]
         |
         |-- [ Sub-Agent: Gamer Strategy ] (Gemini 3 Flash)
         |         |
         |         |-- [MCP Tools] ←→ [ PyBoy Emulator ]
         |         |-- [Vision Analysis] ←→ [ Screenshots ]
         |
         |-- [ Browser Automation ] (Fallback)
                   |
                   |-- [ React Web UI ] ←→ [ Flask API ]
```

---

## Quick Start

### 1. Install Dependencies
```bash
cd /home/duckets/ai-Py-boy-emulation-main/ai-game-server
pip install pyboy mcp pillow
```

### 2. Register MCP Server
```bash
mcporter add pyboy-emulator --stdio "python3 mcp_server.py"
mcporter list  # Verify 5 tools available
```

### 3. Test Tools
```bash
# Load ROM
mcporter call pyboy-emulator.emulator_load_rom rom_path="/path/to/pokemon.gb"

# Press button
mcporter call pyboy-emulator.emulator_press_button button="START"

# Get screenshot
mcporter call pyboy-emulator.emulator_get_frame
```

### 4. Spawn Gamer Agent
```bash
/home/duckets/.openclaw/workspace/tools/spawn-gamer-agent.sh \
    "/path/to/pokemon.gb" \
    "Start new game and walk to first route"
```

---

## Vision-Based Gameplay Loop

The key innovation is **vision-first gameplay**:

```python
# Agent workflow
while not objective_complete:
    # 1. Capture current frame
    frame = emulator_get_frame()
    
    # 2. Analyze with vision model
    decision = vision_analyze(
        image=frame,
        prompt="What button should I press to progress in this Pokemon game?"
    )
    
    # 3. Execute action
    emulator_press_button(decision.button)
    
    # 4. Save screenshot for debugging
    save_screenshot(frame, f"emulator-{timestamp}.png")
```

---

## Integration with OpenClaw Features

### Screenshot Pipeline
All emulator frames integrate with OpenClaw's vision-first automation:
- Screenshots saved to: `/home/duckets/.openclaw/workspace/screenshots/`
- Naming: `emulator-{game}-{step}-{timestamp}.png`
- Vision models: Gemini 3 Flash (unlimited), Qwen-VL

### Sub-Agent Strategy
- **Default model:** Gemini 3 Flash (unlimited on Pro plan)
- **Task breakdown:** Complex objectives split into sub-tasks
- **Progress reporting:** Agent posts updates on completion

### Browser Automation Fallback
If MCP is unavailable, use browser automation:
```bash
# Start Flask server
python3 start_server.py

# Use OpenClaw browser tool to control React UI
browser navigate http://localhost:5173
# ... automate via UI clicks
```

---

## Next Steps (Phase 2)

- [ ] Test with actual ROM files
- [ ] Create game-specific strategy guides (Pokemon, Tetris, Mario)
- [ ] Add REST API endpoints for non-MCP access
- [ ] Implement RAM reading for game state extraction
- [ ] Add save/load state functionality
- [ ] Create pre-built agent strategies for popular games

---

## Files Modified/Created

| File | Type | Purpose |
|------|------|---------|
| `ai-game-server/mcp_server.py` | Created | MCP server (5 tools) |
| `ai-game-server/vision_bridge.py` | Created | Frame capture utility |
| `skills/game-emulation/SKILL.md` | Created | OpenClaw skill docs |
| `tools/spawn-gamer-agent.sh` | Created | Sub-agent spawner |
| `OPENCLAW-INTEGRATION.md` | Created | This guide |

---

## Troubleshooting

### PyBoy Not Found
```bash
pip install pyboy
```

### MCP Server Won't Start
```bash
# Test manually
python3 ai-game-server/mcp_server.py
# Check for import errors
```

### Tools Not Showing in mcporter
```bash
mcporter remove pyboy-emulator
mcporter add pyboy-emulator --stdio "python3 /path/to/mcp_server.py"
mcporter list
```

---

**Author:** DuckBot  
**Date:** 2026-02-23  
**License:** MIT
