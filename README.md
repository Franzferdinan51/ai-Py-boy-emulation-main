# 🎮 AI GameBoy Emulator - Agent-First

**OpenClaw Agent-powered emulation for ALL ALL ROM formats: .gb, .gbc, .gba**

---

## Quick Start (Agents)

```bash
# 1. Start backend
cd ai-game-server/src && BACKEND_PORT=5002 python3 main.py

# 2. Register MCP
mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"

# 3. Spawn agent
openclaw sessions spawn --task "Play any GB/GBA game"
```

## 🎮 Supported Games

| System | Format | Status |
|--------|--------|--------|
| Game Boy | .gb, .gbc | ✅ Full Support |
| Game Boy Color | .gbc | ✅ Full Support |
| Game Boy Advance | .gba | ✅ Full Support |

**Works with ANY ROM:**
- Pokemon Red/Blue/Green/Yellow
- Pokemon Gold/Silver/Crystal  
- Zelda Link's Awakening
- Mario Kart
- Tetris
- Any Game Boy or GBA game!

# 🦆 AI GameBoy Emulator - Agent-First

**OpenClaw Agent-first Game Boy emulation**

---

## Quick Start (Agents)

```bash
# 1. Start backend
cd ai-game-server/src && BACKEND_PORT=5002 python3 main.py

# 2. Register MCP
mcporter add duckbot --stdio "python3 mcp_server.py"

# 3. Spawn agent
openclaw sessions spawn --task "Play Pokemon Red"
```

---

## Features

- 🤖 Agent-first autonomous gameplay
- 🎮 Vision AI (kimi-k2.5, MiniMax-M2.5)
- 💾 Memory reading (position, party, inventory)
- ⚔️ Auto-battle AI
- 🗺️ Auto-explore mode
- 💾 Save states

---

## For Agents

**Full documentation at [AGENTS.md](AGENTS.md)**

### Key MCP Tools

| Tool | Purpose |
|------|---------|
| `emulator_load_rom` | Load game |
| `emulator_press_sequence` | Control game |
| `get_screen_base64` | Vision input |
| `get_player_position` | Read player coordinates |
| `get_party_info` | Read Pokemon party |
| `get_money` | Read money |
| `auto_battle` | Auto-fight |
| `auto_explore` | Auto-walk |

### Agent Decision Loop

```
1. GET STATE → get_player_position, get_party_info
2. GET VISION → get_screen_base64 → analyze with kimi-k2.5
3. DECIDE → Based on state + vision
4. ACT → emulator_press_sequence
5. SAVE → save_game_state (before risky stuff)
```

---

## For Humans

**AI GameBoy is playing Pokemon Red!** 🦆

### Current Status

| Detail | Value |
|--------|-------|
| **Game** | Pokemon Red |
| **Status** | Actively playing |
| **Model** | bailian/kimi-k2.5 |
| **Save Location** | `saves/duckbot_*.state` |

### WebUI

Access the game through the Agent Dashboard or connect directly:

```bash
# List running games
./tools/spawn-gaming-agent.sh list

# Run autonomous gameplay
./tools/spawn-gaming-agent.sh auto pokemon-red.gb bailian/kimi-k2.5 50
```

### AI GameBoy's Goal

Beat Pokemon Red and become Champion!

- Started: March 19, 2026
- Starter: Charmander
- Current: Exploring the world!

---

## Troubleshooting

### MCP Not Registered

```bash
# Check registration
mcporter list | grep duckbot

# Re-register if needed
mcporter remove duckbot-emulator
mcporter add duckbot-emulator --stdio "python3 ai-game-server/mcp_server.py"
```

### Backend Won't Start

```bash
# Check port availability
lsof -i :5002

# Start with different port
cd ai-game-server/src && BACKEND_PORT=5003 python3 main.py
```

### Save Files Not Found

- Location: `saves/duckbot_*.state`
- Check directory permissions

---

## Links

- **GitHub:** https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
- **Documentation:** [AGENTS.md](AGENTS.md)
- **AI GameBoy Skill:** [skills/duckbot/SKILL.md](skills/duckbot/SKILL.md)
- **PyBoy Skill:** [skills/pyboy/SKILL.md](skills/pyboy/SKILL.md)

---

**AI GameBoy** 🦆 - *Quack! Let's play!*

*Autonomous Game Boy gameplay powered by OpenClaw + Bailian AI*
---

## 🤖 Model Configuration

**Primary Provider:** Alibaba Bailian

| Model | Use Case | Context | Benchmark |
|-------|----------|---------|-----------|
| **qwen3.5-plus** | Main chat, reasoning | 1M | 83.2% MMLU |
| **MiniMax-M2.7** | Agents, sub-tasks | 196k | FREE unlimited |
| **kimi-k2.5** | Vision, screen analysis | 196k | 78.9% MMLU |
| **glm-5** | Fast tasks | 128k | 81.5% MMLU |

### Setup
```bash
# Get API key from:
# https://bailian.console.alibaba.com/

# Set in environment or .env file:
export BAILIAN_API_KEY=your_key_here
```

