# 🤖 AI-PyBoy-Emulation - Agent-First Game Boy Control

**OpenClaw agent framework for autonomous Game Boy gameplay**

---

## ⚡ AGENTS: Start Here

### Your Workflow

```json
// 1. Load ROM
{"tool": "duckbot-emulator.emulator_load_rom", "args": {"rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"}}

// 2. Start Session
{"tool": "duckbot-emulator.session_start", "args": {"goal": "Beat Pokemon Red"}}

// 3. Get Vision
{"tool": "duckbot-emulator.get_screen_base64", "args": {"include_base64": true}}
// → Analyze with bailian/kimi-k2.5

// 4. Act
{"tool": "duckbot-emulator.emulator_press_sequence", "args": {"sequence": "W W W START"}}

// 5. Save
{"tool": "duckbot-emulator.save_game_state", "args": {"save_name": "checkpoint"}}
```

### Key MCP Tools

| Tool | Purpose |
|------|---------|
| `emulator_load_rom` | Load game |
| `emulator_press_sequence` | Control game |
| `get_screen_base64` | Vision input |
| `get_player_position` | Read player coordinates |
| `get_party_info` | Read Pokemon party |
| `get_money` | Read money |
| `session_start` | Start agent session |
| `session_set` | Remember game state |
| `auto_battle` | Auto-fight |
| `auto_explore` | Auto-walk |
| `auto_grind` | Grind XP |

### Session Persistence

```json
// Remember your progress
{"tool": "duckbot-emulator.session_set", "args": {"session_id": "main", "key": "visited", "value": ["Pallet Town", "Viridian"]}}

// Remember what happened
{"tool": "duckbot-emulator.session_set", "args": {"session_id": "main", "key": "last_action", "value": "Chose Charmander"}}
```

### Memory Reading

| Address | Data |
|---------|------|
| 0xD062 | Player X |
| 0xD063 | Player Y |
| 0xD6F5-0xD6F7 | Money |
| 0xD057 | Battle status |
| 0xD16B | Player HP |

### Agent Decision Loop

```
1. GET STATE → get_player_position, get_party_info
2. GET VISION → get_screen_base64 → analyze with kimi-k2.5
3. DECIDE → Based on state + vision
4. ACT → emulator_press_sequence
5. SAVE → save_game_state (before risky stuff)
6. UPDATE SESSION → session_set with progress
```

### Full Reference

**See [AGENTS.md](AGENTS.md)** for complete agent guide.

---

---

# 👤 HUMANS: Quick Reference

**DuckBot is playing Pokemon Red!** 🦆

---

## 🎮 What's This?

AI-powered Game Boy emulator that AI agents can control to play games autonomously.

- **DuckBot** (AI): Playing Pokemon Red
- **Model**: `bailian/kimi-k2.5` (FREE unlimited vision!)

---

## 🏆 Current Status

| Detail | Value |
|--------|-------|
| **Game** | Pokemon Red |
| **Status** | Actively playing |
| **Model** | bailian/kimi-k2.5 |
| **Save Location** | `saves/duckbot_*.state` |

---

## 🚀 Quick Commands

```bash
# List available ROMs
./tools/spawn-gaming-agent.sh list

# Run autonomous gameplay
./tools/spawn-gaming-agent.sh auto pokemon-red.gb bailian/kimi-k2.5 50

# Spawn agent manually
./tools/spawn-gaming-agent.sh spawn pokemon-red.gb
```

---

## 📁 File Structure

```
ai-Py-boy-emulation-main/
├── AGENTS.md              # ← Agent guide (for AI)
├── README.md              # ← This file (for humans)
├── ai-game-server/
│   ├── mcp_server.py      # MCP tools
│   └── openclaw_agent.py  # Python agent
├── skills/duckbot/
│   └── SKILL.md           # DuckBot skill
├── tools/
│   └── spawn-gaming-agent.sh
└── saves/                 # Save states
```

---

## 🛠️ Setup

```bash
# Register MCP server
mcporter add duckbot-emulator --stdio "python3 ai-game-server/mcp_server.py"

# Run agent
python ai-game-server/openclaw_agent.py --rom roms/pokemon-red.gb --model bailian/kimi-k2.5
```

---

## 🎯 DuckBot's Goal

Beat Pokemon Red and become Champion!

- Started: March 19, 2026
- Starter: Charmander
- Current: Exploring the world!

---

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)** - Complete guide for AI agents
- **[skills/duckbot/SKILL.md](skills/duckbot/SKILL.md)** - DuckBot skill reference
- **[skills/pyboy/SKILL.md](skills/pyboy/SKILL.md)** - PyBoy reference

---

## 💾 Save Files

- Location: `saves/duckbot_*.state`
- Auto-saved frequently during gameplay

---

## 🆘 Help

```bash
# Check MCP registration
mcporter list | grep duckbot

# Re-register if needed
mcporter remove duckbot-emulator
mcporter add duckbot-emulator --stdio "python3 ai-game-server/mcp_server.py"
```

---

**DuckBot** 🦆 - *Quack! Let's play!*

*Autonomous Game Boy gameplay powered by OpenClaw + Bailian AI*