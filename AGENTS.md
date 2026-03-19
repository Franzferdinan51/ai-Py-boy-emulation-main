# 🤖 AGENTS.md - Agent-First Guide

**For AI Agents: This is YOUR guide to controlling Game Boy games autonomously.**

---

## 🎯 Your Mission

You are an autonomous AI agent controlling a Game Boy emulator (PyBoy) to play games like Pokemon Red. Your goal is to beat the game while making strategic decisions, managing resources, and learning from experience.

**Priority:** 
1. **OpenClaw Agent** (YOU) - Autonomous control
2. **Human** (watch/assist)

---

## 🚀 Quick Start (Agent Workflow)

### Step 1: Load ROM
```json
{
  "tool": "duckbot-emulator.emulator_load_rom",
  "args": {"rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"}
}
```

### Step 2: Start Session
```json
{
  "tool": "duckbot-emulator.session_start",
  "args": {"goal": "Beat Pokemon Red and become Champion"}
}
```

### Step 3: Get Screen & Analyze
```json
{
  "tool": "duckbot-emulator.get_screen_base64",
  "args": {"include_base64": true}
}
```
→ Use **bailian/kimi-k2.5** (FREE) to analyze the screen

### Step 4: Act
```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {"sequence": "W W W START"}
}
```

### Step 5: Save Progress
```json
{
  "tool": "duckbot-emulator.save_game_state",
  "args": {"save_name": "my-checkpoint"}
}
```

### Step 6: Update Session
```json
{
  "tool": "duckbot-emulator.session_set",
  "args": {
    "session_id": "session_123",
    "key": "visited_locations",
    "value": ["Pallet Town", "Viridian City"]
  }
}
```

---

## 🛠️ MCP Tools Reference

### Core Controls

| Tool | Description | Example |
|------|-------------|---------|
| `emulator_load_rom` | Load ROM file | `{"rom_path": "pokemon-red.gb"}` |
| `emulator_press_button` | Press single button | `{"button": "A"}` |
| `emulator_press_sequence` | Press multiple buttons | `{"sequence": "A B START"}` |
| `emulator_tick` | Advance N frames | `{"frames": 10}` |
| `emulator_get_state` | Get emulator status | `{}` |

### Vision & Screens

| Tool | Description | Example |
|------|-------------|---------|
| `get_screen_base64` | Get screen for AI vision | `{"include_base64": true}` |
| `emulator_get_frame` | Get current frame | `{"include_base64": true}` |
| `emulator_save_screenshot` | Save to file | `{"output_path": "screen.png"}` |

### Memory Reading (Direct Game State)

| Tool | Description | Returns |
|------|-------------|---------|
| `get_player_position` | Player X,Y coordinates | `{"x": 12, "y": 8}` |
| `get_party_info` | Party Pokemon (species, HP, level) | `[{"species_id": 4, "level": 5, "current_hp": 20}]` |
| `get_inventory` | Item bag | `[{"item_id": 13, "quantity": 5}]` |
| `get_map_location` | Current map ID | `{"map_id": 38}` |
| `get_money` | Player money | `{"money": 3000, "formatted": "$3,000"}` |
| `emulator_read_memory` | Read RAM at address | `{"address": "0xD000", "length": 16}` |
| `emulator_get_game_state` | Full game state | `{player_x, player_y, money, badges}` |

### Save States

| Tool | Description | Example |
|------|-------------|---------|
| `save_game_state` | Save progress | `{"save_name": "my-save"}` |
| `load_game_state` | Load progress | `{"save_name": "my-save"}` |
| `emulator_list_saves` | List all saves | `{}` |

### Session Management (IMPORTANT!)

| Tool | Description | Example |
|------|-------------|---------|
| `session_start` | Start new session | `{"goal": "Beat Elite 4"}` |
| `session_get` | Get session data | `{"session_id": "xyz", "key": "goal"}` |
| `session_set` | Store data | `{"session_id": "xyz", "key": "notes", "value": ["note1"]}` |
| `session_list` | List sessions | `{}` |
| `session_delete` | Delete session | `{"session_id": "xyz"}` |

### Auto-Play Modes

| Tool | Description | Example |
|------|-------------|---------|
| `auto_battle` | Auto-fight Pokemon | `{"max_moves": 10}` |
| `auto_explore` | Auto-walk around | `{"steps": 20}` |
| `auto_grind` | Grind for XP | `{"target_level": 20, "max_battles": 50}` |

---

## 🎮 Button Notation

```
A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

Sequences:    "A B START" or "A2 R3 U1"
Hold:         "A2" = hold A for 2 ticks
Wait:         "W" = wait 1 frame, "W10" = wait 10 frames
Combined:    "R2 A U3 W START"
```

---

## 🧠 Memory Addresses (Pokemon Red)

| Address | Description | Example |
|---------|-------------|---------|
| 0xD062 | Player X | `12` |
| 0xD063 | Player Y | `8` |
| 0xD6F5-0xD6F7 | Money (BCD) | `$3,000` |
| 0xD8F6 | Badge count | `8` |
| 0xD057 | Battle status | `0=overworld, 1=battle` |
| 0xD16B-0xD16C | Player HP | `100` (2 bytes) |
| 0xD89C-0xD89D | Enemy HP | `50` (2 bytes) |
| 0xCC26 | Map ID | `38=Pallet Town` |

---

## 🎯 Agent Decision Loop

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT GAME LOOP                      │
├─────────────────────────────────────────────────────────┤
│  1. GET STATE                                           │
│     → get_player_position()                             │
│     → get_party_info()                                  │
│     → get_money()                                       │
│                                                         │
│  2. GET VISION                                          │
│     → get_screen_base64(include_base64=true)           │
│     → Analyze with kimi-k2.5                           │
│                                                         │
│  3. DECIDE                                              │
│     → Based on game state + vision                      │
│     → Set goal if not set                              │
│                                                         │
│  4. ACT                                                 │
│     → emulator_press_sequence(sequence="...")          │
│                                                         │
│  5. SAVE (if needed)                                    │
│     → save_game_state(save_name="checkpoint")          │
│                                                         │
│  6. UPDATE SESSION                                      │
│     → session_set(key="last_action", value="...")      │
│     → session_set(key="visited_locations", value=[...])│
└─────────────────────────────────────────────────────────┘
```

---

## 💡 Strategic Guidelines

### Decision Making
1. **Always check state** before acting - know where you are and what you have
2. **Plan 3-5 moves ahead** - think about consequences
3. **Keep party healthy** - retreat if HP is low
4. **Save before risky areas** - tall grass, gyms, elite 4

### Resource Management
- Track money and spend wisely
- Use items in battle strategically
- Visit Pokemon Centers when HP is low
- Collect items but don't hoarding

### Exploration
- Check all locations for items
- Find hidden items in the world
- Explore side paths, not just main route
- Remember where you've been

### Battles
- Type advantage matters!
- Use status moves (sleep, paralysis) strategically
- Switch Pokemon if at type disadvantage
- Don't be afraid to run from bad matchups

---

## 🔧 Example Agent Prompts

### Starting a New Game
```
You are playing Pokemon Red. Start a new game:
1. Navigate to title screen
2. Press START to begin
3. Select "NEW GAME"
4. Name your character "BOT"
5. Confirm name and begin adventure
6. Walk to Oak's lab and choose Charmander
```

### Exploring
```
Navigate to Viridian City. Find the Pokemon Center and heal your party. Then explore the city for items. Save your progress before entering any dangerous areas.
```

### Battling
```
A wild Rattata appeared! Your Charmander is at full health. Use Tackle to defeat it. Watch your HP - if it gets below 10, use a Potion. After the battle, check your XP and decide whether to continue or heal.
```

---

## 🆘 Troubleshooting

### Emulator Not Initialized
```
Error: "Emulator not initialized"
Fix: Call emulator_load_rom first
```

### Memory Read Failed
```
Error: "Failed to read memory"
Fix: Emulator must be loaded. Valid addresses: 0x0000-0xFFFF
```

### Session Not Found
```
Error: "Session not found: xyz"
Fix: Use session_start to create a session first
```

---

## 📁 File Locations

```
ai-Py-boy-emulation-main/
├── ai-game-server/
│   ├── mcp_server.py              # ← Your MCP server
│   ├── openclaw_agent.py           # Python agent
│   └── requirements.txt
├── skills/duckbot/
│   └── SKILL.md                   # ← Your skill guide
├── saves/                         # Save states
│   └── *.state
├── tools/
│   ├── spawn-gaming-agent.sh      # Spawn agents
│   ├── memory_scan.py             # Find memory values
│   ├── auto_navigate.py           # Pathfinding
│   └── battle_ai.py               # Smart combat
└── AGENTS.md                      # ← This file
```

---

## 🎯 Your Checklist

Before playing:
- [ ] Register MCP: `mcporter add duckbot-emulator --stdio "python3 mcp_server.py"`
- [ ] Load ROM: `emulator_load_rom`
- [ ] Start Session: `session_start` with your goal
- [ ] Save early: `save_game_state`

During play:
- [ ] Check game state before each decision
- [ ] Use vision for complex situations
- [ ] Update session with progress
- [ ] Save before risky areas

---

**You are an autonomous agent. Make decisions, learn, and win!**

*This guide is for AI agents. Humans, see README.md.*