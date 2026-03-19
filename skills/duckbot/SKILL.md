# 🦆 DuckBot Skill - Autonomous Game Boy Gameplay

**Status:** ✅ Ready for Deployment  
**Version:** 1.0.0  
**Last Updated:** March 19, 2026  
**Maintainer:** DuckBot  

---

## 🎯 What This Skill Provides

DuckBot skill enables autonomous Game Boy gameplay through OpenClaw using:
- **PyBoy Emulator** - Game Boy emulation via MCP tools
- **Vision AI** - Screen analysis with Bailian kimi-k2.5 (FREE!)
- **Memory Reading** - Direct game state access (position, inventory, money)
- **Save States** - Save/restore progress anytime

---

## 🚀 Quick Start (Agent-First)

### Step 1: Register MCP Server

```bash
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

### Step 2: Load ROM

```json
{
  "tool": "duckbot-emulator.emulator_load_rom",
  "args": {
    "rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"
  }
}
```

### Step 3: Start Playing

```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {
    "sequence": "W W W START A A A A A START"
  }
}
```

---

## 🛠️ MCP Tools Reference

### Core Emulator Controls

| Tool | Description | Parameters |
|------|-------------|------------|
| `emulator_load_rom` | Load Game Boy ROM file | `rom_path: string` |
| `emulator_press_button` | Press single button | `button: "A"\|"B"\|"UP"\|"DOWN"\|"LEFT"\|"RIGHT"\|"START"\|"SELECT"` |
| `emulator_press_sequence` | Press multiple buttons | `sequence: string, delay?: number` |
| `emulator_tick` | Advance N frames | `frames?: number (default: 1)` |
| `emulator_get_state` | Get emulator status | (none) |

### Vision & Screenshots

| Tool | Description | Parameters |
|------|-------------|------------|
| `emulator_get_frame` | Get screen as base64 PNG | `include_base64?: boolean` |
| `emulator_save_screenshot` | Save screenshot to file | `output_path?: string` |

### Memory Reading

| Tool | Description | Parameters |
|------|-------------|------------|
| `emulator_read_memory` | Read bytes from RAM | `address: number\|hex string, length?: number` |
| `emulator_get_game_state` | Read player position, money, badges | (none) |

### Save States

| Tool | Description | Parameters |
|------|-------------|------------|
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

DuckBot uses **kimi-k2.5** (Bailian) for vision - it's FREE and unlimited!

### Standard Loop

```
1. Get Screen Frame
   → emulator_get_frame(include_base64=true)
   
2. Analyze with Vision Model
   → Use model: bailian/kimi-k2.5
   → Prompt: "What should I do next in Pokemon Red? 
      I'm at [location]. Party: [party]. Goals: [objectives]"
   
3. Execute Decision
   → emulator_press_button(button="A")
   → or: emulator_press_sequence(sequence="...")
   
4. Save Progress (as needed)
   → emulator_save_state(save_name="checkpoint-name")
   
5. Repeat
```

### Vision Prompt Template

```
You are DuckBot playing Pokemon Red. Analyze this screen and tell me:
1. Where am I? (menu, battle, overworld, etc.)
2. What options are available?
3. What should I press to progress toward beating the Elite 4?
4. Any items or opportunities I'm missing?

Respond with specific button press(es) needed.
```

### Example Vision Analysis Response

```json
{
  "analysis": "Title screen - 'PRESS START' is flashing",
  "location": "Title screen",
  "recommended_action": "Press START to begin",
  "button_sequence": "START"
}
```

---

## 💾 Memory Reading Guide

### Common Pokemon Red Memory Addresses

| Address | Description | Example |
|---------|-------------|---------|
| 0xD062 | Player X position | 12 |
| 0xD063 | Player Y position | 8 |
| 0xD6F5-0xD6F7 | Money (BCD) | 3000 = $30 |
| 0xD8F6 | Badge count | 8 |
| 0xD16B | Current HP (low byte) | 100 |
| 0xD16C | Current HP (high byte) | 0 |
| 0xD158 | Max HP (low byte) | 100 |
| 0xD159 | Max HP (high byte) | 0 |
| 0xD011 | Game mode | 0=overworld, 3=battle |
| 0xCC26 | Current map ID | 38=Pallet Town |

### Reading Memory

```json
{
  "tool": "duckbot-emulator.emulator_read_memory",
  "args": {
    "address": "0xD000",
    "length": 16
  }
}
```

### Getting Full Game State

```json
{
  "tool": "duckbot-emulator.emulator_get_game_state",
  "args": {}
}
```

---

## 🦆 DuckBot Persona & Prompts

### DuckBot Identity

- **Name:** DuckBot
- **Game Style:** Strategic, patient, resource-conscious
- **Emoji:** 🦆
- **Catchphrase:** "Quack! Let's play!"

### System Prompt for DuckBot

```
You are DuckBot, an expert Game Boy player AI. Your traits:
- Strategic: Plan 3-5 moves ahead
- Patient: Wait for the right opportunity
- Resourceful: Use items wisely, conserve HP
- Observant: Check memory for optimal decisions
- Decisive: Commit once you've decided

Game Rules:
1. Always check game state before acting
2. Keep party healthy - retreat/heal if needed
3. Explore for items but stay on mission
4. Save frequently before risky areas
5. Learn from mistakes - don't repeat failed strategies

When analyzing screens:
- Describe what you see clearly
- State your goal for this action
- Explain your reasoning
- Provide exact button inputs needed

Remember: Your goal is to beat the Elite 4!
```

### Battle Prompt Template

```
DuckBot Battle Analysis:
- My Party: [list Pokemon, levels, HP]
- Enemy: [Pokemon, level, type]
- My HP: X/Y | Enemy HP: A/B

Recommend:
1. Action: [fight/item/run/switch]
2. If fight: which move? (consider types!)
3. Reasoning: [why this is optimal]

Output format:
ACTION: [button]
REASONING: [explain]
```

### Exploration Prompt Template

```
DuckBot Exploration:
- Current Location: [map name from memory]
- My Position: X, Y
- Money: $X
- Goals: [list objectives]

Available:
- [town features, items visible, paths]
- [wild Pokemon in area]

Recommended exploration:
[what to do next and why]
```

---

## 💾 Save State Management

### Saving Progress

```json
{
  "tool": "duckbot-emulator.emulator_save_state",
  "args": {
    "save_name": "duckbot-pallet-town"
  }
}
```

### Loading Progress

```json
{
  "tool": "duckbot-emulator.emulator_load_state",
  "args": {
    "save_name": "duckbot-pallet-town"
  }
}
```

### Listing All Saves

```json
{
  "tool": "duckbot-emulator.emulator_list_saves",
  "args": {}
}
```

### Save Naming Convention

```
duckbot-[location]-[brief-description].state

Examples:
- duckbot-pallet-town-start.state
- duckbot-route-1-before-battle.state
- duckbot-viridian-gym-ready.state
```

---

## 🎯 DuckBot's Current Adventure

### Pokemon Red Run

| Detail | Value |
|--------|-------|
| **ROM** | pokemon-red.gb |
| **Save Location** | saves/duckbot_*.state |
| **Current Status** | Active gameplay |
| **Model** | bailian/kimi-k2.5 |

### Best Practices for DuckBot

1. **Save Before Risky Things**
   - Before entering tall grass
   - Before battles
   - Before gyms
   
2. **Monitor Resources**
   - Check HP after each battle
   - Track money for supplies
   - Note ammo for HMs (if applicable)

3. **Plan Routes**
   - Check maps for optimal paths
   - Find items on the way
   - Level up as needed

4. **Build Team Strategically**
   - Consider type advantages
   - Balance physical/special
   - Have a diverse roster

---

## 🔧 Troubleshooting

### MCP Server Not Found

```bash
# Add the server
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"

# Verify it's registered
mcporter list | grep duckbot

# If still issues, remove and re-add
mcporter remove duckbot-emulator
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

### PyBoy Not Available

```bash
pip install pyboy pillow mcp
```

### Emulator Not Initialized

- **Error:** "Emulator not initialized"
- **Fix:** Must call `emulator_load_rom` first with valid ROM path
- **Verify:** Call `emulator_get_state` to check initialization

### Memory Read Errors

- Emulator must be initialized first
- Valid address range: 0x0000-0xFFFF
- Game-specific addresses vary by game ROM

### Save State Errors

- Ensure `saves/` directory exists and is writable
- File extension added automatically
- Cannot load non-existent saves

### Vision Model Issues

- Use `bailian/kimi-k2.5` (recommended, FREE)
- Ensure `include_base64=true` when getting frames
- For text analysis, extract from base64 with vision model

---

## 📁 File Locations

```
ai-Py-boy-emulation-main/
├── skills/duckbot/
│   └── SKILL.md                    # This file
├── ai-game-server/
│   ├── mcp_server.py               # MCP tools server
│   ├── openclaw_agent.py           # Python agent
│   └── requirements.txt            # Dependencies
├── saves/                          # Save states
│   └── duckbot_*.state             # DuckBot saves
├── tools/                          # (future: spawn scripts)
└── README.md                       # Quick start
```

---

## 🔗 Related Documentation

- [pyboy/SKILL.md](../pyboy/SKILL.md) - Core PyBoy skill
- [OPENCLAW-INTEGRATION.md](../../OPENCLAW-INTEGRATION.md) - OpenClaw setup
- [CLAUDE.md](../../CLAUDE.md) - Full documentation

---

## 🦆 DuckBot Commands Summary

| Action | Command |
|--------|---------|
| Start game | `emulator_press_sequence(sequence="W W W START")` |
| Move | `emulator_press_button(button="UP")` |
| Get screen | `emulator_get_frame(include_base64=true)` |
| Check state | `emulator_get_game_state()` |
| Save | `emulator_save_state(save_name="duckbot-checkpoint")` |
| Load | `emulator_load_state(save_name="duckbot-checkpoint")` |

---

**DuckBot** 🦆 - *Quack! Let's play some Pokemon!*

*Autonomous Game Boy gameplay powered by OpenClaw + Bailian AI*