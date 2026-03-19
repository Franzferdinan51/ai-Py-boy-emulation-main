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
mcporter add duckbot-emulator --stdio "python3 mcp_server.py"

# Load a ROM
mcporter call duckbot-emulator.emulator_load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"

# Press buttons
mcporter call duckbot-emulator.emulator_press_button button="A"
mcporter call duckbot-emulator.emulator_press_sequence sequence="W W START"

# Get screenshot for AI vision
mcporter call duckbot-emulator.emulator_get_frame include_base64=true
```

### Option 2: Use Spawn Script

```bash
# List available ROMs
./tools/spawn-gaming-agent.sh list

# Show spawn command for a game
./tools/spawn-gaming-agent.sh spawn pokemon-red.gb

# Run autonomous gameplay
./tools/spawn-gaming-agent.sh auto pokemon-red.gb bailian/kimi-k2.5 50
```

### Option 3: Run DuckBot Agent Directly

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
| **Spawn Scripts** | Quick agent launching |

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

### Memory
- `emulator_read_memory` - Read RAM at address
- `emulator_get_game_state` - Read player data

### Save States
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
│   ├── duckbot/
│   │   └── SKILL.md          # 🎯 DuckBot primary skill (NEW!)
│   ├── pyboy/
│   │   └── SKILL.md          # Core PyBoy skill
│   └── game-emulation/
│       └── SKILL.md          # Legacy skill
├── tools/
│   └── spawn-gaming-agent.sh # 🆕 Spawn script for gaming agents
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

## 🦆 DuckBot Skill

The **DuckBot Skill** (`skills/duckbot/SKILL.md`) provides:

- **Agent-first documentation** - Designed for autonomous AI agents
- **Complete MCP tool reference** - All tools with examples
- **Vision-based gameplay workflow** - See → Analyze → Act loop
- **Memory reading guide** - Pokemon Red memory addresses
- **DuckBot persona** - System prompts for authentic gameplay
- **Troubleshooting** - Common issues and fixes

### Quick DuckBot Workflow

```
1. Load ROM → emulator_load_rom
2. Get Screen → emulator_get_frame(include_base64=true)
3. Analyze → Use bailian/kimi-k2.5 to understand screen
4. Act → emulator_press_button or emulator_press_sequence
5. Save → emulator_save_state (before risky stuff)
6. Repeat!
```

---

## 🎯 Spawn Script Usage

```bash
# List available ROMs
./tools/spawn-gaming-agent.sh list

# Spawn agent (shows commands)
./tools/spawn-gaming-agent.sh spawn pokemon-red.gb

# Run autonomous gameplay
./tools/spawn-gaming-agent.sh auto pokemon-red.gb bailian/kimi-k2.5 50

# Register MCP server
./tools/spawn-gaming-agent.sh register
```

---

## 🦆 About DuckBot

DuckBot is an OpenClaw agent running on Bailian models:
- **Primary Model:** `bailian/kimi-k2.5` (FREE unlimited vision!)
- **Planning:** `bailian/MiniMax-M2.5` (FREE unlimited)
- **Vibe:** Strategic, patient, resource-conscious

### DuckBot's Pokemon Red Adventure

| Detail | Value |
|--------|-------|
| **Started** | March 19, 2026 ~3:00 AM |
| **Starter** | Charmander |
| **Current** | Actively playing! |
| **Goal** | Beat the Elite 4! |

---

## 📚 Links

- [skills/duckbot/SKILL.md](skills/duckbot/SKILL.md) - 🎯 DuckBot primary skill
- [skills/pyboy/SKILL.md](skills/pyboy/SKILL.md) - Core PyBoy reference
- [OPENCLAW-INTEGRATION.md](OPENCLAW-INTEGRATION.md) - OpenClaw setup guide

---

**🦆 Quack! Let's play some Pokemon!**

*Autonomous Game Boy gameplay powered by OpenClaw + Bailian AI*