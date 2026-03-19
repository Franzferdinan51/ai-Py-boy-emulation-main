# 🤖 AGENTS.md - Agent-First Guide for OpenClaw

**For AI Agents: This is YOUR guide to controlling Game Boy games autonomously via OpenClaw.**

---

## 🎯 Your Mission

You are an autonomous AI agent controlling a Game Boy emulator (PyBoy) via OpenClaw to play games like Pokemon Red. Your goal is to beat the game while making strategic decisions, managing resources, and learning from experience.

**Priority:** 
1. **OpenClaw Agent** (YOU) - Autonomous control
2. **Human** (watch/assist)

---

## 🚀 Quick Start (OpenClaw Agent Workflow)

### Step 1: Register MCP Server (If Not Already)

```bash
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

### Step 2: Load ROM

```json
{
  "tool": "duckbot-emulator.emulator_load_rom",
  "args": {"rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"}
}
```

### Step 3: Start Session (Optional - Use Sub-Agents!)

```bash
# Spawn a sub-agent for a specific gaming task
sessions_spawn --task "Beat the Elite 4 in Pokemon Red" --model bailian/MiniMax-M2.5
```

### Step 4: Get Screen & Analyze

```json
{
  "tool": "duckbot-emulator.get_screen_base64",
  "args": {"include_base64": true}
}
```
→ Use **bailian/kimi-k2.5** (FREE vision) to analyze the screen

### Step 5: Act

```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {"sequence": "W W W START"}
}
```

### Step 6: Save Progress

```json
{
  "tool": "duckbot-emulator.save_game_state",
  "args": {"save_name": "my-checkpoint"}
}
```

---

## 🎮 OpenClaw Agent Patterns

### Sub-Agent Spawning (sessions_spawn)

The power of OpenClaw is spawning specialized sub-agents:

```bash
# Spawn vision-focused gaming agent
sessions_spawn \
  --task "Explore Viridian City in Pokemon Red, find the Pokemart and buy Potions" \
  --model bailian/kimi-k2.5 \
  --label "exploration-agent" \
  --runTimeoutSeconds 300

# Spawn battle specialist
sessions_spawn \
  --task "Battle Brock at Pewter City Gym. Use type advantage (Water/Grass) to win." \
  --model bailian/qwen3.5-plus \
  --label "battle-agent"

# Spawn with cleanup (delete session after)
sessions_spawn \
  --task "Quick: grind Route 1 for 10 minutes" \
  --model bailian/MiniMax-M2.5 \
  --cleanup delete
```

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `task` | Task description | "Battle Brock" |
| `model` | Override default model | `bailian/kimi-k2.5` |
| `label` | Session label | "gym-battle" |
| `runTimeoutSeconds` | Max runtime (0=unlimited) | 300 |
| `cleanup` | Delete/keep after | `delete` or `keep` |

### Session Tools

```bash
# List active gaming sessions
sessions_list

# Get session details with messages
sessions_list activeMinutes=30 messageLimit=5

# Get transcript history
sessions_history sessionKey="agent:xxx:subagent:yyy"

# Send message to gaming session
sessions_send sessionKey="agent:xxx:subagent:yyy" message="Continue grinding!"
```

### Multi-Agent Routing

Route gaming tasks to specialized models:

| Task Type | Model | Why |
|-----------|-------|-----|
| **Screen Analysis** | `bailian/kimi-k2.5` | FREE vision + image understanding |
| **Battle Strategy** | `bailian/qwen3.5-plus` | Best reasoning (83.2% MMLU) |
| **Exploration** | `bailian/MiniMax-M2.5` | FREE unlimited, fast |
| **Complex Puzzles** | `bailian/qwen3.5-plus` | Complex reasoning |
| **Quick Tasks** | `bailian/MiniMax-M2.5` | Fastest, FREE |

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

## 🧠 Memory Addresses (any Game Boy game)

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
| 0xD18C | Player Pokemon level | `5` |
| 0xD163 | Player Pokemon species | `4=Charmander` |
| 0xD883 | Enemy Pokemon species | `19=Rattata` |

---

## 👁️ Vision Workflows

### Standard Vision Loop (OpenClaw-Native)

```
┌─────────────────────────────────────────────────────┐
│           OPENCLAW VISION GAMEPLAY                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. SPAWN AGENT (optional)                          │
│     sessions_spawn --task "Explore..." --model     │
│                                                     │
│  2. GET FRAME                                       │
│     └─ emulator_get_frame(include_base64=true)     │
│                                                     │
│  3. ANALYZE WITH VISION                            │
│     └─ Use bailian/kimi-k2.5 (FREE)                │
│                                                     │
│  4. DECIDE                                          │
│     └─ Choose button sequence                       │
│                                                     │
│  5. ACT                                             │
│     └─ emulator_press_sequence(sequence="...")      │
│                                                     │
│  6. SAVE (if needed)                                │
│     └─ emulator_save_state(save_name="...")         │
│                                                     │
│  7. ANNOUNCE (auto via sessions_spawn)            │
│     └─ Results delivered to main chat              │
│                                                     │
│  8. REPEAT                                          │
└─────────────────────────────────────────────────────┘
```

### Vision Prompt Templates

#### Screen Analysis Prompt

```
Analyze this Pokemon game screenshot and tell me:
1. What screen am I on? (title, menu, battle, overworld)
2. What's happening in this screen?
3. What options are available?
4. What is my likely goal right now?
5. What button(s) should I press to progress?

Provide specific button inputs.
```

#### Battle Analysis Prompt

```
This is a Pokemon battle. Tell me:
- My Pokemon: [species], Level [X], HP: [Y/Z]
- Enemy Pokemon: [species], Level [X], HP: [Y/Z]
- My moves: [list available moves with types]

Analyze the type matchup and tell me:
1. Do I have type advantage?
2. Which move is best?
3. Should I fight, use item, switch, or run?
4. What's my exact button sequence?
```

#### Exploration Prompt

```
I'm exploring in any Game Boy game.
- Current location: [from memory]
- My goal: [reach next city / find items / explore]

Looking at this screen:
1. What do I see? (buildings, paths, items, NPCs)
2. Where can I go?
3. What's worth investigating?
4. What buttons get me there?
```

---

## 🌲 Decision Trees

### Master Game Loop Decision Tree

```
START
  │
  ├─► GET SCREEN
  │     emulator_get_frame(include_base64=true)
  │
  ├─► ANALYZE WITH VISION
  │     "What should I do?"
  │
  ├─► DETERMINE CONTEXT
  │     │
  │     ├─► TITLE SCREEN?
  │     │     └─► Press START → Select NEW GAME
  │     │
  │     ├─► BATTLE?
  │     │     └─► Analyze matchup → Fight/Item/Switch/Run
  │     │
  │     ├─► MENU?
  │     │     └─► Navigate to goal
  │     │
  │     ├─► DIALOGUE?
  │     │     └─► Press A to continue
  │     │
  │     └─► OVERWORLD?
  │           └─► Navigate to goal
  │
  ├─► DECIDE ACTION
  │     Choose button sequence
  │
  ├─► EXECUTE
  │     emulator_press_sequence(sequence="...")
  │
  ├─► SAVE (if needed)
  │     emulator_save_state(save_name="...")
  │
  └─► ANNOUNCE (if spawned)
        └─► Results sent to main chat
```

---

## 💡 Strategic Guidelines

### Decision Making (OpenClaw Agent)
1. **Always check state** before acting - know where you are and what you have
2. **Plan 3-5 moves ahead** - think about consequences
3. **Keep party healthy** - retreat if HP is low
4. **Save before risky areas** - tall grass, gyms, elite 4
5. **Use sub-agents** for complex tasks - let specialized agents handle them

### Using Sub-Agents Effectively
- Spawn for distinct tasks (exploration, grinding, battles)
- Label sessions for debugging
- Set appropriate timeouts
- Use vision model for screen-heavy tasks
- Use reasoning model for strategy-heavy tasks

### Resource Management
- Track money and spend wisely
- Use items in battle strategically
- Visit Pokemon Centers when HP is low
- Collect items but don't hoard

---

## 🔧 Example Agent Prompts (OpenClaw Format)

### Starting a New Game via Sub-Agent

```
Spawn a sub-agent:
sessions_spawn --task "Start a new Pokemon Red game: 1) Navigate to title screen 2) Press START 3) Select NEW GAME 4) Name character BOT 5) Choose Charmander 6) Walk to Oak's lab" --model bailian/kimi-k2.5 --label "new-game"
```

### Exploring via Sub-Agent

```
sessions_spawn --task "Navigate to Viridian City via Route 1: 1) Walk DOWN out of Pallet Town 2) Battle wild Pokemon to gain XP 3) Navigate north to Viridian City 4) Save progress before entering city" --model bailian/MiniMax-M2.5 --label "exploration"
```

### Battle via Sub-Agent

```
sessions_spawn --task "Battle Brock in Pewter City Gym: 1) His Geodude is Rock/Ground type 2) Use Water or Grass Pokemon 3) If only Charmander, use Ember 4) Keep an eye on HP 5) Save before battle!" --model bailian/qwen3.5-plus --label "gym-battle"
```

---

## 🆘 Troubleshooting

### MCP Server Not Registered

```bash
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
mcporter list | grep duckbot
```

### Sub-Agent Not Spawning

```bash
# Check session tools enabled
sessions_list

# Verify model is valid
# Valid: bailian/kimi-k2.5, bailian/qwen3.5-plus, bailian/MiniMax-M2.5
```

### Vision Not Working

```bash
# Ensure include_base64=true
emulator_get_frame(include_base64=true)

# Use correct model
# bailian/kimi-k2.5 is recommended for vision
```

### Emulator Not Initialized

```bash
# Load ROM first
emulator_load_rom(rom_path="/path/to/rom.gb")
```

---

## 📁 File Locations

```
ai-Py-boy-emulation-main/
├── skills/
│   ├── openclaw/
│   │   └── SKILL.md              # ← OpenClaw integration patterns
│   ├── duckbot/
│   │   └── SKILL.md             # ← DuckBot persona
│   └── pyboy/
│       └── SKILL.md             # ← PyBoy reference
├── ai-game-server/
│   ├── mcp_server.py            # ← Your MCP server
│   └── requirements.txt
├── tools/
│   └── AGENT_QUICKSTART.md
└── AGENTS.md                    # ← This file
```

---

## 🎯 Your Checklist (OpenClaw)

Before playing:
- [ ] Register MCP: `mcporter add duckbot-emulator --stdio "python3 mcp_server.py"`
- [ ] Verify registration: `mcporter list | grep duckbot`
- [ ] Load ROM: `emulator_load_rom`
- [ ] Use sub-agents for complex tasks

During play:
- [ ] Check game state before each decision
- [ ] Use vision model (bailian/kimi-k2.5) for complex situations
- [ ] Spawn sub-agents for specialized tasks
- [ ] Save before risky areas
- [ ] Use sessions_spawn for complex multi-step tasks

---

## 🔗 Related Documentation

- [OpenClaw Skills](https://docs.openclaw.ai/tools/skills) - Skill format
- [Session Tools](https://docs.openclaw.ai/concepts/session-tool) - sessions_spawn
- [Skills/openclaw/SKILL.md](skills/openclaw/SKILL.md) - OpenClaw integration skill
- [Skills/duckbot/SKILL.md](skills/duckbot/SKILL.md) - DuckBot persona

---

## 🏆 Best Practices Summary

| Practice | Why |
|----------|-----|
| Use `sessions_spawn` for complex tasks | Isolated session, auto-announce results |
| Use vision model for screen analysis | bailian/kimi-k2.5 is FREE |
| Label your sub-agents | Easier debugging |
| Set appropriate timeouts | Prevents runaway agents |
| Save before risky content | Game state protection |
| Use right model for task | Vision vs reasoning vs fast |

---

**You are an autonomous OpenClaw agent. Spawn sub-agents, make decisions, and win!**

*This guide follows OpenClaw patterns. See skills/openclaw/SKILL.md for detailed integration patterns.*

*This guide is for AI agents. Humans, see README.md.*