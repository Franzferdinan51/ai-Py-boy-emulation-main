# 🦆 PyBoy Game Emulation Skill

**Status:** ✅ Ready for DuckBot  
**Maintainer:** DuckBot  
**Repo:** `ai-Py-boy-emulation-main`  
**Last Updated:** March 19, 2026

---

## What This Skill Provides

DuckBot can control Game Boy emulators (via PyBoy) to play games autonomously using vision-based AI decision making.

---

## 🚀 Quick Start

### 1. Register MCP Server (One-time)

```bash
cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server
mcporter add pyboy-emulator --stdio "python3 mcp_server.py"
```

### 2. Use Tools in Conversation

```bash
# Load a ROM
mcporter call pyboy-emulator.emulator_load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"

# Press buttons to start game
mcporter call pyboy-emulator.emulator_press_button button="START"
mcporter call pyboy-emulator.emulator_press_sequence sequence="W W W A W A2 START"

# Get screenshot for vision analysis
mcporter call pyboy-emulator.emulator_get_frame include_base64=true
```

---

## 🎮 MCP Tools Reference

### Core Emulator Controls

| Tool | Description | Parameters |
|------|-------------|-------------|
| `emulator_load_rom` | Load Game Boy ROM | `rom_path: string` |
| `emulator_press_button` | Press single button | `button: string` |
| `emulator_press_sequence` | Press multiple buttons | `sequence: string, delay?: number` |
| `emulator_tick` | Advance N frames | `frames?: number` |
| `emulator_get_state` | Get emulator status | (none) |

### Vision & Screenshots

| Tool | Description | Parameters |
|------|-------------|-------------|
| `emulator_get_frame` | Get screen as base64 PNG | `include_base64?: boolean` |
| `emulator_save_screenshot` | Save screenshot to file | `output_path?: string` |

### Memory Reading

| Tool | Description | Parameters |
|------|-------------|-------------|
| `emulator_read_memory` | Read bytes from RAM | `address: number\|hex, length?: number` |
| `emulator_get_game_state` | Read player position/money/badges | (none) |

### Save States

| Tool | Description | Parameters |
|------|-------------|-------------|
| `emulator_save_state` | Save current state | `save_name?: string` |
| `emulator_load_state` | Load saved state | `save_name: string` |
| `emulator_list_saves` | List all saves | (none) |

---

## 🎮 Button Notation

### Basic Buttons
```
A, B, UP, DOWN, LEFT, RIGHT, START, SELECT
```

### Sequences
```
"A B START"           # Press A, then B, then START
"A2"                  # Press A twice (shorthand)
"R3"                  # Hold RIGHT for 3 frames
"U1 D1 L1 R1"         # Up, down, left, right once each
"W"                   # Wait 1 frame
"W10"                 # Wait 10 frames
```

### Combined Example
```
"A W W START W A W W"  # Press A, wait 2 frames, press START, wait 1 frame, press A, etc.
```

---

## 👁️ Vision-Based Gameplay Workflow

DuckBot uses **[SELECT_VISION_MODEL]** (Bailian) for vision - it's FREE and unlimited!

### Standard Loop

```
1. GET SCREEN
   → emulator_get_frame(include_base64=true)

2. ANALYZE (with bailian/[SELECT_VISION_MODEL])
   → "What should I do? I'm at [location]. Goals: [objectives]"

3. DECIDE
   → Choose button sequence based on analysis

4. ACT
   → emulator_press_sequence(sequence="...")

5. SAVE (as needed)
   → emulator_save_state(save_name="checkpoint")

6. REPEAT
```

---

## 🧠 Decision Prompts for Autonomous Play

### Starting a New Game

```
You are DuckBot playing Pokemon Red. Analyze the screen and tell me:
1. What screen am I on? (title, menu, overworld, battle)
2. What is the highlighted/selected option?
3. What button(s) should I press to start a new game?

Respond with specific button sequence needed.
```

### Exploration Mode

```
DuckBot Exploration:
- Current screen shows [describe what you see]
- My goal: [reach destination / find item / explore]
- Available paths: [list visible directions]

What should I do? Give me a button sequence to navigate.
```

### Battle Mode

```
DuckBot Battle Analysis:
- My Pokemon: [species], Level [X], HP: [Y/Z]
- Enemy Pokemon: [species], Level [X], HP: [Y/Z]
- My moves: [list available moves]

Analyze the type matchup and recommend:
1. Which move to use (consider type advantage!)
2. Or should I use an item / switch / run?
```

### Menu Navigation

```
I need to [open bag / check party / use item / save game].
Current screen: [describe what's visible]
Menu structure: [if visible]

What buttons do I press to navigate there?
```

---

## 👁️ Vision Analysis Prompts

### Screen Analysis Prompt

```
Analyze this Pokemon Red screenshot and provide:
1. **Location**: Where am I? (town, route, building, battle)
2. **Context**: What's happening? (walking, menu, dialogue)
3. **Options**: What choices are available?
4. **Goal**: What's my objective right now?
5. **Action**: What button(s) should I press?

Be specific - give exact button inputs.
```

### Battle Analysis Prompt

```
This is a Pokemon battle screen. Tell me:
1. My Pokemon species and current HP
2. Enemy Pokemon species and current HP  
3. Type matchup (who has advantage?)
4. What move should I select?
5. Button sequence to execute it
```

### Menu Analysis Prompt

```
Analyze this menu screen:
1. What menu is open? (PokeGear, Bag, Party)
2. What items/options are visible?
3. Which is currently selected?
4. What's my likely goal here?
5. Button sequence to accomplish it
```

### Overworld Analysis Prompt

```
Analyze this overworld screen:
1. Exact location name
2. Visible NPCs, buildings, signs
3. Accessible paths (which directions can I walk)
4. Any items visible on screen
5. Suggested exploration goal
```

---

## 💾 Memory Address Reference (Pokemon Red)

### Player State

| Address | Description | Example |
|---------|-------------|---------|
| 0xD062 | Player X position (tile) | `12` |
| 0xD063 | Player Y position (tile) | `8` |
| 0xD057 | Game mode | `0=overworld, 3=battle` |
| 0xCC26 | Current map ID | `38=Pallet Town` |
| 0xD6F5-0xD6F7 | Money (BCD format) | `0x30 0x00 = $3,000` |
| 0xD8F6 | Badge count | `0-8` |
| 0xD11C | Play time (seconds) | `3600 = 1 hour` |

### Battle State

| Address | Description | Example |
|---------|-------------|---------|
| 0xD057 | Battle status | `0=overworld, 1=battle` |
| 0xD6B5-0xD6B6 | Player current HP | `50` (low, high byte) |
| 0xD6BE-0xD6BF | Player max HP | `100` (low, high byte) |
| 0xD6C9 | Player level | `5` |
| 0xD163 | Player Pokemon species ID | `4=Charmander` |
| 0xD89C-0xD89D | Enemy current HP | `20` (low, high byte) |
| 0xD8A0-0xD8A1 | Enemy max HP | `50` (low, high byte) |
| 0xD8C6 | Enemy level | `3` |
| 0xD883 | Enemy Pokemon species ID | `19=Rattata` |

### Inventory

| Address | Description | Example |
|---------|-------------|---------|
| 0xD31D | Number of items | `12` |
| 0xD31E-0xD347 | Item IDs (50 bytes) | `13=Potion` |
| 0xD348-0xD371 | Item quantities | `5` |

### Map IDs (Common)

| ID | Location | ID | Location |
|----|----------|----|----------|
| 0x26 | Pallet Town | 0x2C | Viridian City |
| 0x2D | Viridian Forest | 0x31 | Pewter City |
| 0x36 | Route 22 | 0x37 | Route 2 |
| 0x3D | Cerulean City | 0x47 | Route 24 |
| 0x4D | Vermilion City | 0x5A | Celadon City |
| 0x68 | Fuchsia City | 0x73 | Saffron City |
| 0x86 | Cinnabar Island | 0x8D | Indigo Plateau |

### Reading Memory

```python
# Read single address
address = 0xD062
value = emulator.memory[address]

# Read multiple bytes
address = 0xD6F5
length = 3
values = [emulator.memory[address + i] for i in range(length)]
# values = [0x30, 0x00, 0x00] = $3,000

# Get game state
emulator_get_game_state()
# Returns: {player_x, player_y, money, badges, map_id}
```

---

## 📋 Comprehensive Examples

### Example 1: Complete Game Start

```python
# Step 1: Load ROM
emulator_load_rom(rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb")

# Step 2: Start game (wait for animation, press START)
emulator_press_sequence(sequence="W W W W W START")

# Step 3: Select "NEW GAME"
emulator_press_button(button="A")

# Step 4: Name entry - press A to confirm defaults
emulator_press_sequence(sequence="A A A A A START")

# Step 5: Walk out of house (DOWN 3 times)
emulator_press_sequence(sequence="D D D")

# Step 6: Go to Oak's lab (RIGHT 6 times)
emulator_press_sequence(sequence="R R R R R R")

# Step 7: Get screenshot and analyze
emulator_get_frame(include_base64=true)
# → Analyze with [SELECT_VISION_MODEL]
```

### Example 2: Battle Flow

```python
# Detect battle
game_state = emulator_get_game_state()
# {game_mode: 3} means in battle

# Get battle info
# (Use vision or memory reading)

# Decide action based on type matchup
# Fire vs Grass → Ember is super effective!
# Fire vs Water → Bad matchup, consider switching!

# Execute attack
emulator_press_sequence(sequence="A W A")  # Select FIGHT, then move

# Use item if needed
if player_hp < 20:
    emulator_press_sequence(sequence="B W A")  # Open bag, select potion
```

### Example 3: Exploration & Item Collection

```python
# Get current position
game_state = emulator_get_game_state()
# {player_x: 12, player_y: 8, map_id: 38}

# Use vision to find visible items
emulator_get_frame(include_base64=true)
# → Vision sees: "There's a Potion on the ground at X, Y"

# Navigate to item
# Calculate path, then execute
emulator_press_sequence(sequence="R R D D A")  # Walk and pick up

# Save progress
emulator_save_state(save_name="duckbot-route-1-items")
```

### Example 4: Healing & Pokemon Center

```python
# Navigate to Pokemon Center
# (from memory or vision analysis)

# Enter building (UP to enter)
emulator_press_button(button="U")

# Talk to nurse (A)
emulator_press_button(button="A")
# Wait for healing animation
emulator_press_sequence(sequence="W W W W W W W W W W")

# Say thanks (A)
emulator_press_button(button="A")

# Exit (DOWN)
emulator_press_button(button="D")

# Save state
emulator_save_state(save_name="duckbot-healed")
```

---

## 🔧 Troubleshooting

### Common Issues

#### PyBoy not found
```bash
pip install pyboy pillow mcp
```

#### MCP server not registered
```bash
# Add the server
mcporter add pyboy-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"

# Verify
mcporter list | grep pyboy

# If issues, remove and re-add
mcporter remove pyboy-emulator
mcporter add pyboy-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

#### Emulator not initialized
```
Error: "Emulator not initialized"
Cause: Must load ROM first
Fix: Call emulator_load_rom with valid ROM path
```

#### Memory read errors
```
Error: "Failed to read memory"
Cause: Emulator not initialized or invalid address
Fix:
- Ensure ROM is loaded first
- Use valid address range: 0x0000-0xFFFF
- Game-specific addresses vary by ROM
```

#### Save state errors
```
Error: "Cannot save state"
Cause: Directory not writable or path issue
Fix:
- Ensure saves/ directory exists and is writable
- File extension added automatically
- Check path permissions
```

#### Vision not working
```
Problem: Can't get base64 image
Fix:
- Use bailian/[SELECT_VISION_MODEL] (FREE, recommended)
- Ensure include_base64=true in emulator_get_frame
- Check that ROM is loaded
```

#### Buttons not responding
```
Problem: Button presses don't seem to work
Fix:
- Add wait frames (W) between inputs
- Some games need timing: "W A W A W A"
- Check game state is ready for input
```

#### Stuck in menu
```
Problem: Can't exit menu / navigating incorrectly
Fix:
- Press B to go back (often)
- Press START to close
- Use vision to understand menu structure
```

---

## 💡 Best Practices

### Before Playing
- [ ] Register MCP server (one-time)
- [ ] Test ROM loads correctly
- [ ] Save initial state ("start")

### During Gameplay
- [ ] Check game state before decisions
- [ ] Use vision for complex situations
- [ ] Save before risky areas (gyms, elite 4)
- [ ] Track HP and resources

### Decision Making
- [ ] Plan 3-5 moves ahead
- [ ] Consider type advantages in battle
- [ ] Keep party healthy
- [ ] Don't hoard items - use them!

### Save Strategy
- [ ] Save after reaching milestones
- [ ] Save before entering dangerous areas
- [ ] Save after winning difficult battles
- [ ] Keep multiple save slots for recovery

---

## 📁 File Locations

```
ai-Py-boy-emulation-main/
├── ai-game-server/
│   ├── mcp_server.py          # MCP tools server
│   ├── openclaw_agent.py      # Python agent
│   ├── vision_bridge.py       # Screenshot utilities
│   └── requirements.txt       # Dependencies
├── skills/pyboy/
│   └── SKILL.md              # This file
├── tools/
│   ├── memory_scan.py         # Memory scanning tool
│   ├── battle_ai.py           # Battle AI tool
│   ├── auto_navigate.py       # Auto pathfinding
│   └── spawn-gaming-agent.sh  # Agent spawner
├── saves/                     # Save states
│   └── duckbot_*.state        # DuckBot saves
└── README.md                 # Quick start guide
```

---

## 🦆 DuckBot's Save Files

**Location:** `saves/duckbot_*.state`

DuckBot automatically saves progress during gameplay using the naming convention:
- `duckbot-[location]-[description].state`
- Example: `duckbot-viridian-gym-ready.state`

---

## 🎯 Recommended Models

For DuckBot, use **Bailian models** (free unlimited):

| Model | Vision | Cost | Best For |
|-------|--------|------|----------|
| `bailian/[SELECT_VISION_MODEL]` | ✅ Yes | **FREE** | Primary vision agent |
| `bailian/[SELECT_TEXT_MODEL]` | ❌ No | **FREE** | Planning/reasoning |
| `bailian/[SELECT_REASONING_MODEL]` | ✅ Yes | 18K/mo quota | Complex reasoning |

**Recommended:** `bailian/[SELECT_VISION_MODEL]` for all gameplay - it's FREE and vision-capable!

---

## Dependencies

```bash
pip install pyboy pillow openai mcp
```

---

**DuckBot** 🦆 - Playing Game Boy games with AI power!