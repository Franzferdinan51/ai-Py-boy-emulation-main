# Game Emulation Skill

**Status:** ✅ Ready for OpenClaw agents  
**Repo:** ai-Py-boy-emulation-main  
**MCP Server:** ai-game-server/mcp_server.py

## What This Skill Provides

AI agents can control Game Boy emulators (PyBoy) to play games autonomously using vision-based decision making.

## Quick Start

### 1. Register MCP Server (One-time)

```bash
cd /Users/duckets/.openclaw/workspace/ai-pyboy-emulation/ai-game-server
mcporter add pyboy-emulator --stdio "python3 mcp_server.py"
```

Verify tools are available:
```bash
mcporter list | grep pyboy-emulator
```

### 2. Use Tools Directly

```bash
# Load a ROM
mcporter call pyboy-emulator.emulator_load_rom rom_path="/path/to/pokemon.gb"

# Press buttons
mcporter call pyboy-emulator.emulator_press_button button="START"
mcporter call pyboy-emulator.emulator_press_button button="A"

# Get screenshot for vision analysis
mcporter call pyboy-emulator.emulator_get_frame

# Check emulator state
mcporter call pyboy-emulator.emulator_get_state
```

## MCP Tools Available

| Tool | Description | Example |
|------|-------------|---------|
| `emulator_load_rom` | Load Game Boy ROM | `rom_path="/path/to/game.gb"` |
| `emulator_press_button` | Press button (A/B/UP/DOWN/LEFT/RIGHT/START/SELECT) | `button="A"` |
| `emulator_get_frame` | Get screenshot as base64 PNG | (no args) |
| `emulator_get_state` | Get emulator status | (no args) |
| `emulator_tick` | Advance N frames | `frames=10` |

## Vision-Based Gameplay

For AI agents with vision capabilities:

```python
# 1. Get current screen
frame = emulator_get_frame()  # Returns base64 PNG

# 2. Analyze with vision model (use kimi-k2.5 for FREE unlimited)
# Send base64 image to vision model with prompt like:
# "What button should I press to progress in Pokemon?"

# 3. Execute the decision
emulator_press_button(button="A")  # or whatever the AI decides
```

## Recommended Models

For OpenClaw agents, use Bailian models (free unlimited):

| Model | Vision | Cost | Best For |
|-------|--------|------|----------|
| `bailian/kimi-k2.5` | ✅ Yes | FREE | Primary vision agent |
| `bailian/MiniMax-M2.5` | ❌ No | FREE | Planning/reasoning |
| `bailian/qwen3.5-plus` | ✅ Yes | 18K/mo | Complex reasoning |

**Example spawn for vision-based gameplay:**
```bash
sessions_spawn \
  --task "Play Pokemon using pyboy-emulator MCP tools" \
  --model bailian/kimi-k2.5
```

## Python API Usage

```python
from ai_game_server.openclaw_agent import OpenClawGameAgent

config = {
    'ROM_PATH': '/path/to/pokemon.gb',
    'MODEL_DEFAULTS': {'MODEL': 'bailian/kimi-k2.5'},
    'CURRENT_GOAL': 'Get to the first Pokemon'
}

agent = OpenClawGameAgent(config)
agent.run_auto(max_turns=10, use_vision=True)
```

## Files & Locations

```
ai-pyboy-emulation/
├── ai-game-server/
│   ├── mcp_server.py          # MCP tools server
│   ├── openclaw_agent.py      # Python agent with vision
│   ├── vision_bridge.py       # Screenshot utilities
│   └── requirements.txt       # Dependencies
├── skills/game-emulation/
│   └── SKILL.md              # This file
└── run_openclaw_agent.sh     # Shell launcher
```

## Dependencies

```bash
pip install pyboy pillow openai mcp
```

## Troubleshooting

**PyBoy not found:**
```bash
pip install pyboy
```

**MCP tools not showing:**
```bash
mcporter remove pyboy-emulator
mcporter add pyboy-emulator --stdio "python3 /path/to/mcp_server.py"
```

**Vision not working:**
- Use `bailian/kimi-k2.5` (FREE unlimited vision)
- Or use local LM Studio with qwen3-vl-8b

---

**Author:** DuckBot  
**Date:** 2026-03-19