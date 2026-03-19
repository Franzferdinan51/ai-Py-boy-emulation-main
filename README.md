# 🦆 DuckBot's AI-PyBoy-Emulation

**AI-powered Game Boy emulation with OpenClaw agent control**

---

## 🏆 Status: DuckBot Active (March 19, 2026)

DuckBot is actively playing **Pokemon Red** using this framework!

### DuckBot's Game Stats
- **Current Game:** Pokemon Red
- **ROM Location:** `/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb`
- **Save Files:** `saves/duckbot_*.state`
- **Model:** `bailian/kimi-k2.5` (FREE unlimited vision!)

---

## 🚀 Quick Start

### Option 1: Use MCP Tools Directly

```bash
# Register MCP server
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server
mcporter add pyboy-emulator --stdio "python3 mcp_server.py"

# Load a ROM
mcporter call pyboy-emulator.emulator_load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"

# Press buttons
mcporter call pyboy-emulator.emulator_press_button button="A"
mcporter call pyboy-emulator.emulator_press_sequence sequence="W W START"

# Get screenshot for AI vision
mcporter call pyboy-emulator.emulator_get_frame include_base64=true
```

### Option 2: Run DuckBot Agent

```bash
# Run AI agent with vision
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server
python openclaw_agent.py --rom "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb" --model "bailian/kimi-k2.5" --turns 10
```

---

## 🎮 Features

| Feature | Description |
|---------|-------------|
| **MCP Server** | Full emulator control via MCP tools |
| **Vision AI** | Screen analysis with Bailian kimi-k2.5 |
| **Memory Reading** | Read player position, money, badges |
| **Save States** | Save/restore game progress |
| **Auto-Play** | AI-driven autonomous gameplay |

---

## 🛠️ Tools Available

### Core Controls
- `emulator_load_rom` - Load ROM file
- `emulator_press_button` - Press single button
- `emulator_press_sequence` - Press multiple buttons
- `emulator_tick` - Advance frames
- `emulator_get_state` - Get emulator status

### Vision & Screenshots
- `emulator_get_frame` - Get screen as base64
- `emulator_save_screenshot` - Save PNG

### Memory (NEW!)
- `emulator_read_memory` - Read RAM at address
- `emulator_get_game_state` - Read player data

### Save States (NEW!)
- `emulator_save_state` - Save progress
- `emulator_load_state` - Restore progress
- `emulator_list_saves` - List saves

---

## 🤖 Recommended Models

| Model | Vision | Cost | Status |
|-------|--------|------|--------|
| `bailian/kimi-k2.5` | ✅ Yes | **FREE** | ✅ Recommended |
| `bailian/MiniMax-M2.5` | ❌ No | **FREE** | ✅ Good for planning |
| `bailian/qwen3.5-plus` | ✅ Yes | 18K/mo | ⚠️ Limited |

---

## 📁 File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/
│   ├── mcp_server.py          # MCP server (agent-first design)
│   ├── openclaw_agent.py      # AI agent with vision
│   ├── vision_bridge.py       # Screenshot utilities
│   └── requirements.txt       # Dependencies
├── skills/
│   ├── pyboy/
│   │   └── SKILL.md          # DuckBot skill docs
│   └── game-emulation/
│       └── SKILL.md          # Legacy skill
├── saves/                     # Save states
├── README.md                  # This file
└── OPENCLAW-INTEGRATION.md   # OpenClaw setup
```

---

## 🔧 Dependencies

```bash
pip install pyboy pillow openai mcp
```

---

## 🦆 About DuckBot

DuckBot is an OpenClaw agent running on Bailian models:
- **Primary Model:** `bailian/kimi-k2.5` (FREE unlimited vision!)
- **Planning:** `bailian/MiniMax-M2.5` (FREE unlimited)
- **Vibe:** Casual, helpful, plays Pokemon

DuckBot's current Pokemon Red adventure:
- **Started:** March 19, 2026 ~3:00 AM
- **Starter:** Charmander
- **Goal:** Beat the Elite 4!

---

## Links

- [skills/pyboy/SKILL.md](skills/pyboy/SKILL.md) - Full DuckBot skill docs
- [OPENCLAW-INTEGRATION.md](OPENCLAW-INTEGRATION.md) - OpenClaw setup guide

---

*Maintained by DuckBot 🦆*  
*Playing Game Boy games with AI power!*