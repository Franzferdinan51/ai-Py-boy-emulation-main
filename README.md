# 🎮 AI GameBoy Emulator - Agent-First

**OpenClaw Agent-powered emulation for Game Boy, Game Boy Color, and Game Boy Advance games.**

---

## What This Is

An MCP server + web interface that lets AI agents control Game Boy emulation autonomously. Agents can:
- 🤖 **Read game memory** (position, inventory, HP, etc.)
- 🎮 **Press buttons** to control games
- 👁️ **Analyze screens** via vision AI
- 🧠 **Make autonomous decisions** to play games

---

## Supported Systems

| System | Format | Status |
|--------|--------|--------|
| Game Boy | .gb | ✅ Full Support |
| Game Boy Color | .gbc | ✅ Full Support |
| Game Boy Advance | .gba | ✅ Full Support |

**Works with ANY ROM:** Pokemon, Zelda, Mario, Tetris, and more!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI GAMEBOY ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────────┐
                         │   OPENCLAW AGENT    │
                         │   (bailian/kimi)    │
                         └──────────┬──────────┘
                                    │
                                    │ 1. Spawns MCP calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER                                    │
│                   ai-game-server/mcp_server.py                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Button       │  │ Memory       │  │ Save/Load    │  │ Auto-Play   │ │
│  │ Controls     │  │ Reading      │  │ States       │  │ Modes       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │   PYBOY           │           │   PYGBA           │
        │   EMULATOR        │           │   EMULATOR        │
        │   (.gb/.gbc)      │           │   (.gba)          │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      GAME SCREEN OUTPUT       │
                    │   (160x144 / 240x160)         │
                    └───────────────────────────────┘
```

### Component Flow

```
┌─────────┐    MCP     ┌─────────────┐   PyBoy    ┌─────────┐
│ Agent   │ ─────────► │ MCP Server  │ ─────────► │ Emulator│
│         │  JSON/RPC │             │  calls     │         │
│         │ ◄──────── │             │ ◄────────  │         │
└─────────┘  Response └─────────────┘            └─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Memory/RAM     │
                        │  Reading        │
                        └─────────────────┘
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Clone and enter
git clone https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
cd ai-Py-boy-emulation-main

# Install Python dependencies
cd ai-game-server
pip install -r requirements.txt
pip install pyboy pygba mcp

# Install Node (for web UI)
cd ../ai-game-assistant
npm install
```

### Step 2: Start Backend

```bash
cd ai-game-server
python3 mcp_server.py
```

### Step 3: Register MCP

```bash
mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"
```

### Step 4: Spawn Agent

```bash
openclaw sessions spawn --task "Play Pokemon Red"
```

---

## 🎯 Agent Workflows

### Basic Gameplay Loop

```
┌──────────────────────────────────────────────────────────────┐
│                      AGENT DECISION LOOP                     │
└──────────────────────────────────────────────────────────────┘

 1. GET STATE    → get_player_position, get_party_info, get_money
 2. GET VISION   → get_screen_base64 → analyze with vision AI
 3. DECIDE       → Based on game state and goals
 4. ACT          → emulator_press_sequence
 5. SAVE         → save_game_state (before risky actions)
```

### Example: Play Pokemon Red

```python
# 1. Load ROM
emulator_load_rom(rom_path="pokemon-red.gb")

# 2. Start game
emulator_press_sequence("START A W60 A")

# 3. Get game state
position = get_player_position()
party = get_party_info()

# 4. Get screen for vision
screen = get_screen_base64()

# 5. Make decisions and act
emulator_press_sequence("RIGHT RIGHT RIGHT A")

# 6. Save progress
save_game_state("checkpoint-1")
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | Full agent guide with all tools |
| [guides/DECISION_TREE.md](guides/DECISION_TREE.md) | How agents make decisions |
| [guides/API_REFERENCE.md](guides/API_REFERENCE.md) | Complete MCP API reference |
| [EXAMPLES.md](EXAMPLES.md) | Example prompts and game sessions |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and fixes |

---

## 🛠️ MCP Tools Reference

### Core Controls

| Tool | Description |
|------|-------------|
| `emulator_load_rom` | Load a ROM file |
| `emulator_press_button` | Press single button |
| `emulator_press_sequence` | Press button sequence |
| `emulator_tick` | Advance N frames |
| `emulator_get_state` | Get emulator status |

### Vision

| Tool | Description |
|------|-------------|
| `get_screen_base64` | Get screen for AI vision |
| `emulator_save_screenshot` | Save screen to file |

### Memory Reading

| Tool | Description |
|------|-------------|
| `get_player_position` | Player X,Y coordinates |
| `get_party_info` | Pokemon party data |
| `get_inventory` | Items in bag |
| `get_map_location` | Current map ID |
| `get_money` | Player money |

### Save States

| Tool | Description |
|------|-------------|
| `save_game_state` | Save emulator state |
| `load_game_state` | Load saved state |
| `emulator_list_saves` | List all saves |

### Auto-Play

| Tool | Description |
|------|-------------|
| `auto_battle` | AI battles Pokemon |
| `auto_explore` | AI explores world |
| `auto_grind` | Grind for XP/money |

### Session

| Tool | Description |
|------|-------------|
| `session_start` | Start agent session |
| `session_get` | Get session data |
| `session_set` | Store session data |

---

## 💾 Memory Addresses (Pokemon Red)

| Address | Description |
|---------|-------------|
| 0xD062 | Player X |
| 0xD063 | Player Y |
| 0xD35E | Current Map ID |
| 0xD6F5 | Money (BCD) |
| 0xD163 | Party Count |
| 0xD16B-0xD17F | Party Pokemon 1 |

---

## 🖥️ Web UI (Optional)

```bash
# Start web UI
cd ai-game-assistant
npm run dev
```

Access at **http://localhost:5173**

Features:
- Manual game control
- Agent status monitoring
- Screen viewing
- Settings configuration

---

## 📁 File Structure

```
ai-Py-boy-emulation-main/
├── README.md                 # This file
├── AGENTS.md                 # Agent-first guide
├── CLAUDE.md                 # Claude Code guidance
├── ENHANCEMENTS.md           # Recent updates
├── ai-game-server/           # Python backend + MCP
│   ├── mcp_server.py         # MCP server (main entry)
│   ├── src/
│   │   └── backend/
│   │       └── server.py     # Flask API
│   └── requirements.txt
├── ai-game-assistant/        # React web UI
├── skills/                   # OpenClaw skills
│   ├── pyboy/
│   └── game-agent/
├── tools/                    # Agent utilities
├── guides/                   # Documentation
│   ├── DECISION_TREE.md
│   ├── API_REFERENCE.md
│   └── (more docs)
├── saves/                    # Save states
└── roms/                     # Game ROMs
```

---

## 🔧 Troubleshooting

### MCP Not Working

```bash
# Check registration
mcporter list | grep gameboy

# Re-register
mcporter remove gameboy
mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"
```

### Emulator Won't Start

```bash
# Check PyBoy installed
python3 -c "from pyboy import PyBoy; print('OK')"

# Check port available
lsof -i :5002
```

### ROM Won't Load

- Verify ROM file exists
- Check file format (.gb, .gbc, .gba)
- Ensure file is readable

**More troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🧠 Model Configuration

| Model | Use Case | Provider |
|-------|----------|----------|
| `[SELECT_VISION_MODEL]` | Vision/screen analysis | Bailian (FREE) |
| `MiniMax-M2.7` | Agent decisions | Bailian (FREE) |
| `[SELECT_REASONING_MODEL]` | Complex reasoning | Bailian |

---

## Links

- **GitHub:** https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
- **Documentation:** [AGENTS.md](AGENTS.md)
- **AI GameBoy Skill:** [skills/duckbot/SKILL.md](skills/duckbot/SKILL.md)
- **PyBoy Skill:** [skills/pyboy/SKILL.md](skills/pyboy/SKILL.md)

---

## License

MIT

---

**🎮 AI GameBoy** - *Quack! Let's play!*

*Autonomous Game Boy gameplay powered by OpenClaw + Bailian AI*