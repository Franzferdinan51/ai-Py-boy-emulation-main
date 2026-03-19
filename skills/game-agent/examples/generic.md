# Generic Game Guide - Game Boy Emulation

**Status:** ✅ Complete  
**Version:** 1.0  
**Last Updated:** March 19, 2026

---

## Overview

This guide covers generic Game Boy emulation workflows, memory scanner usage, and agent automation patterns that work across different games. Use this as a reference for any Game Boy game you want to automate.

---

## Core Concepts

### Game Boy Memory Model

The original Game Boy has 64KB of RAM (0x0000-0xFFFF), split into:

| Range | Description |
|-------|-------------|
| 0x0000-0x3FFF | ROM Bank 0 (fixed) |
| 0x4000-0x7FFF | Switchable ROM Bank |
| 0x8000-0x9FFF | VRAM (tile data) |
| 0xA000-0xBFFF | External RAM (cartridge) |
| 0xC000-0xCFFF | Working RAM (fixed) |
| 0xD000-0xDFFF | Working RAM (switchable, 7 banks) |
| 0xE00-0xEFFF | Sprite RAM (OAM) |
| 0xFF00-0xFF7F | I/O Registers |
| 0xFF80-0xFFFF | High RAM |

### Common Memory Addresses (Most Games)

| Address | Description |
|---------|-------------|
| 0xD011 | Game state (0=overworld, 3=battle, etc.) |
| 0xD016 | Current player X tile |
| 0xD017 | Current player Y tile |
| 0xD35E | Current map ID (Pokemon-specific) |
| 0xD347-0xD349 | Money (BCD format) |

---

## Memory Scanner Usage

### Reading Memory

```bash
# Read 1 byte at address
mcporter call pyboy.get_memory address=0xD000 length=1

# Read multiple bytes
mcporter call pyboy.get_memory address=0xD000 length=16

# Read as hex
mcporter call pyboy.get_memory address=0xFF00 length=8
```

### Finding Values

**Process for discovering game-specific addresses:**

1. **Initial Scan**
   - Note current value
   - Make in-game change
   - Scan for changed values

2. **Narrow Down**
   - Use same/different comparison
   - Rule out false positives

3. **Verify**
   - Test in different contexts
   - Confirm reliability

### Memory Watching

```bash
# Watch a specific address (manual polling)
while true; do
  mcporter call pyboy.get_memory address=0xD011 length=1
  sleep 1
done
```

---

## Agent Workflow Patterns

### Basic Workflow Loop

```
┌─────────────────────────────────────────────┐
│              AGENT WORKFLOW                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  GET    │───▶│ ANALYZE  │───▶│  ACT   │ │
│  │ SCREEN  │    │ w/ Vision│    │ Button │ │
│  └──────────┘    └──────────┘    └────────┘ │
│       │                              │       │
│       │                              │       │
│       ▼                              ▼       │
│  ┌──────────┐               ┌──────────────┐ │
│  │   GET    │◀──────────────│   WAIT       │ │
│  │  STATE   │               │   (frames)   │ │
│  └──────────┘               └──────────────┘ │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │        SAVE (as needed)               │   │
│  └──────────────────────────────────────┘   │
│                                              │
└─────────────────────────────────────────────┘
```

### Code Pattern

```python
import subprocess
import json
import time

def call_tool(tool_name, args=None):
    """Call MCP tool via mcporter"""
    cmd = ["mcporter", "call", f"pyboy.{tool_name}"]
    if args:
        for k, v in args.items():
            cmd.append(f"{k}={v}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def game_loop(objective, max_iterations=100):
    """Standard game agent loop"""
    for i in range(max_iterations):
        # 1. Get screen
        screen = call_tool("get_screen")
        
        # 2. Analyze with vision
        decision = vision_model.analyze(
            image=screen,
            objective=objective,
            game_state=call_tool("get_session_info")
        )
        
        # 3. Execute action
        if decision["action"] == "press":
            call_tool("press_button", {
                "button": decision["button"],
                "hold_duration": decision.get("duration", 1)
            })
        
        # 4. Wait for animation
        call_tool("wait_frames", {"frames": decision.get("wait", 30)})
        
        # 5. Check state
        state = call_tool("get_session_info")
        
        # 6. Save checkpoint if needed
        if i % 50 == 0:
            call_tool("save_game_state", {"save_path": f"checkpoint-{i}.state"})
        
        # 7. Check if objective complete
        if decision.get("complete"):
            print(f"Objective complete at iteration {i}")
            break
```

---

## Vision Integration

### Screen Capture

```bash
# Get screen as base64
mcporter call pyboy.get_screen

# Response includes:
# {
#   "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAQ...",
#   "width": 160,
#   "height": 144
# }
```

### Vision Model Prompts

```python
VISION_PROMPTS = {
    "default": """You are a Game Boy game assistant.
Current screen: {screenshot}
Game state: {state}

What button should I press next?
Options: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

Reply with just the button name.""",

    "exploration": """You are exploring a Game Boy game.
Current screen shows {location_type}
Visible elements: {elements}

What should I do to explore this area?
Reply with button sequence.""",

    "battle": """BATTLE ANALYSIS:
- My Pokemon: {my_pokemon}
- Enemy Pokemon: {enemy_pokemon}
- My HP: {my_hp}%
- Enemy HP: {enemy_hp}%

Type advantage: {type_matchup}

What move should I use? Reply with button.""",
}
```

---

## Button Input Patterns

### Input Notation

| Notation | Meaning | Example |
|----------|---------|---------|
| `A` | Press A once | `A` = tap A |
| `A5` | Press A five times | `A A A A A` |
| `R3` | Hold RIGHT for 3 frames | Hold right 50ms |
| `W10` | Wait 10 frames | ~166ms |
| `U D L R` | Move in square | Up, down, left, right |

### Common Sequences

```python
BUTTON_SEQUENCES = {
    # Navigation
    "move_up_3": "U W W W",
    "move_right_5": "R W W W W W", 
    "enter_door": "RIGHT W W W W",
    
    # Menu
    "confirm": "A",
    "cancel": "B",
    "open_menu": "START",
    "close_menu": "B",
    
    # Battle
    "attack": "A",
    "run": "DOWN DOWN A",  # Usually down twice then A
    "bag": "B A",
    "pokemon": "B B A",
}
```

---

## Save States

### Save/Load Operations

```bash
# Save current state
mcporter call pyboy.save_game_state save_path="/path/to/save.state"

# Load previous state
mcporter call pyboy.load_game_state save_path="/path/to/save.state"

# List available saves (via session info)
mcporter call pyboy.get_session_info
```

### Save Strategy

```
Checkpoint Strategy:
├── Save before: risky situations
│   ├── Entering unknown areas
│   ├── Difficult battles
│   └── Puzzle sections
│
├── Save after: milestones
│   ├── Completing objectives
│   ├── Winning battles
│   └── Reaching new areas
│
└── Keep recent saves
    ├── 3-4 recent checkpoints
    ├── Label by progress
    └── Test load occasionally
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "ROM not loaded" | No ROM in memory | Call `load_rom` first |
| "Invalid button" | Wrong button name | Use A/B/UP/DOWN/LEFT/RIGHT/START/SELECT |
| "Save not found" | Wrong path | Check file exists |
| "Memory read error" | Invalid address | Use 0x0000-0xFFFF |

### Recovery Patterns

```python
def safe_execute(tool, args=None, max_retries=3):
    """Execute MCP tool with retry logic"""
    for attempt in range(max_retries):
        try:
            result = call_tool(tool, args)
            if result.get("success"):
                return result
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    # Recovery: try to reload ROM
    print("Recovery: reloading ROM")
    call_tool("load_rom", {"rom_path": ROM_PATH})
    return call_tool(tool, args)
```

---

## Game-Specific Patterns

### RPG Pattern

```python
# RPG games (Pokemon, Final Fantasy, etc.)
rpg_loop = {
    "steps": [
        ("explore", "Find next objective"),
        ("navigate", "Move toward goal"),
        ("battle", "Win combat if encountered"),
        ("collect", "Get items along the way"),
        ("save", "Checkpoint after milestones"),
    ],
    "memory_tracking": [
        "player_position",
        "money", 
        "hp",
        "inventory"
    ]
}
```

### Platformer Pattern

```python
# Platformers (Mario, Metroid, etc.)
platformer_loop = {
    "steps": [
        ("scan", "Check screen for hazards"),
        ("move", "Navigate obstacles"),
        ("jump", "Time jumps carefully"),
        ("collect", "Grab items/powerups"),
        ("avoid", "Stay away from enemies"),
    ],
    "focus": "precise_timing"
}
```

### Puzzle Pattern

```python
# Puzzle games (Tetris, breakout, etc.)
puzzle_loop = {
    "steps": [
        ("analyze", "Understand current state"),
        ("plan", "Determine optimal moves"),
        ("execute", "Execute move sequence"),
        ("evaluate", "Check result"),
        ("adapt", "Adjust strategy if needed"),
    ],
    "focus": "pattern_recognition"
}
```

---

## Advanced Techniques

### Multi-Step Sequences

```python
def execute_sequence(sequence, delay=100):
    """Execute complex button sequence"""
    for button in sequence.split():
        call_tool("press_button", {"button": button})
        call_tool("wait_frames", {"frames": delay // 16})
```

### State Machine

```python
class GameState:
    OVERWORLD = 0
    MENU = 1
    BATTLE = 2
    DIALOG = 3
    CUTSCENE = 4

def handle_state(state):
    handlers = {
        GameState.OVERWORLD: handle_overworld,
        GameState.BATTLE: handle_battle,
        GameState.MENU: handle_menu,
        GameState.DIALOG: handle_dialog,
    }
    return handlers.get(state, handle_default)(state)
```

### Memory Watchdog

```python
def watch_memory(address, callback, interval=1):
    """Watch memory address and trigger callback on change"""
    last_value = None
    while True:
        result = call_tool("get_memory", {"address": address, "length": 1})
        current = result["data"][0]
        
        if last_value is not None and current != last_value:
            callback(last_value, current)
        
        last_value = current
        time.sleep(interval)
```

---

## Tool Reference

### Complete Tool List

| Tool | Purpose | Key Parameters |
|------|---------|-----------------|
| `load_rom` | Load game | `rom_path` |
| `get_screen` | Screenshot | - |
| `press_button` | Input | `button`, `hold_duration` |
| `wait_frames` | Delay | `frames` |
| `get_memory` | Read RAM | `address`, `length` |
| `get_player_info` | Game data | - |
| `get_inventory` | Items | - |
| `get_party_pokemon` | Party | - |
| `save_game_state` | Save | `save_path` |
| `load_game_state` | Load | `save_path` |
| `get_session_info` | Status | - |
| `get_server_info` | Version | - |
| `health_check` | Test | - |

---

## File Locations

```
/Users/duckets/.openclaw/workspace/
├── mcp-pyboy/
│   ├── src/mcp_server/
│   │   └── server.py       # MCP server
│   ├── saves/              # Save states
│   └── roms/               # Game ROMs
│
└── ai-Py-boy-emulation-main/
    ├── skills/game-agent/
    │   ├── SKILL.md        # Main docs
    │   ├── openclaw/       # OpenClaw setup
    │   └── examples/
    │       ├── pokemon_red.md  # Pokemon guide
    │       └── generic.md      # This file
    └── ai-game-server/
        └── mcp_server.py   # Alternative MCP
```

---

## See Also

- [game-agent/SKILL.md](../SKILL.md) - Main skill documentation  
- [openclaw/SKILL.md](../openclaw/SKILL.md) - OpenClaw integration
- [pokemon_red.md](./pokemon_red.md) - Pokemon-specific guide
- [AGENTS.md](../../AGENTS.md) - Agent-first patterns
- [OPENCLAW-INTEGRATION.md](../../OPENCLAW-INTEGRATION.md) - Setup guide

---

**Author:** Game Agent 🦆  
**Purpose:** Generic Game Boy automation guide