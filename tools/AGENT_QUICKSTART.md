# 🚀 Agent Quick Start Guide

**5-minute setup for autonomous Game Boy gameplay**

*For AI agents controlling PyBoy emulator*

---

## ⏱️ Quick Setup (5 Minutes)

### Step 1: Verify Environment (30 seconds)

```bash
# Check Python dependencies
pip show pyboy pillow mcp

# If missing, install:
pip install pyboy pillow mcp
```

### Step 2: Register MCP Server (1 minute)

```bash
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"

# Verify
mcporter list | grep duckbot
```

### Step 3: Load ROM (30 seconds)

```json
{
  "tool": "duckbot-emulator.emulator_load_rom",
  "args": {
    "rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"
  }
}
```

### Step 4: Start Game (1 minute)

```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {
    "sequence": "W W W W W START"
  }
}
```

### Step 5: Start Autonomous Play! 🦆

```json
{
  "tool": "duckbot-emulator.emulator_get_frame",
  "args": {
    "include_base64": true
  }
}
```

→ Analyze with `bailian/[SELECT_VISION_MODEL]` → Press buttons → Repeat!

---

## 📋 Minimal Configuration

### Required Tools

| Tool | Purpose |
|------|---------|
| `emulator_load_rom` | Load Game Boy ROM |
| `emulator_press_sequence` | Control game |
| `emulator_get_frame` | Get screenshot for vision |
| `emulator_get_game_state` | Read player position/money |
| `emulator_save_state` | Save progress |

### Button Reference

```
A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

Sequences: "A B START", "R R R D D"
Wait: "W" (wait 1 frame), "W10" (wait 10)
Hold: "R3" (hold RIGHT for 3 frames)
```

---

## 🎮 First Autonomous Play Session

### The Complete Loop

```python
# 1. GET SCREEN
response = {
  "tool": "duckbot-emulator.emulator_get_frame",
  "args": {"include_base64": true}
}
# response.image_base64 contains the screen

# 2. ANALYZE WITH VISION
# Use model: bailian/[SELECT_VISION_MODEL] (FREE!)
# Prompt: "What button should I press to play Pokemon Red?"

# 3. EXECUTE
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {"sequence": "A W W START"}
}

# 4. CHECK STATE (optional)
{
  "tool": "duckbot-emulator.emulator_get_game_state",
  "args": {}
}
# Returns: {x: 12, y: 8, money: 3000, badges: 0}

# 5. SAVE (before risky stuff)
{
  "tool": "duckbot-emulator.emulator_save_state",
  "args": {"save_name": "checkpoint-1"}
}
```

### Example: Start a New Game

```json
{
  "tool": "duckbot-emulator.emulator_load_rom",
  "args": {"rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"}
}
```

```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {"sequence": "W W W W W START"}
}
```

```json
{
  "tool": "duckbot-emulator.emulator_press_button",
  "args": {"button": "A"}
}
```

```json
{
  "tool": "duckbot-emulator.emulator_press_sequence",
  "args": {"sequence": "A A A A A START"}
}
```

→ **Now you're playing!** Use the vision loop to continue.

---

## 🧠 Simple Decision Tree

```
GET SCREEN → ANALYZE → DECIDE → ACT → SAVE → REPEAT

Where am I?
├── Title Screen → Press START → Select NEW GAME
├── Battle → Analyze matchup → Fight/Item/Run
├── Menu → Navigate to goal → Execute
└── Overworld → Decide direction → Move

What should I do?
1. Check game state (position, HP, money)
2. Analyze screen with vision
3. Choose action based on goals
4. Execute button sequence
5. Save if needed
6. Repeat!
```

---

## 📍 Your First Hour

### 0-5 min: Setup (Done! ✅)

- [x] MCP registered
- [x] ROM loaded
- [x] Game started

### 5-15 min: Early Game

- [ ] Name character (press A through name)
- [ ] Walk out of house
- [ ] Visit Professor Oak
- [ ] Choose starter (Charmander recommended 🔥)
- [ ] Explore Pallet Town
- [ ] Save state

### 15-30 min: Route 1

- [ ] Head to Viridian City
- [ ] Battle wild Pokemon
- [ ] Catch Pidgey/Rattata
- [ ] Save progress

### 30-60 min: Viridian City

- [ ] Visit Pokemart (buy Potions!)
- [ ] Explore city
- [ ] Head toward Route 2
- [ ] Enter Viridian Forest
- [ ] Navigate to Pewter City

---

## 🔧 Troubleshooting

### "Emulator not initialized"

**Cause:** No ROM loaded  
**Fix:** Call `emulator_load_rom` first

### "Failed to read memory"

**Cause:** Emulator not ready or invalid address  
**Fix:** Load ROM, then read memory

### "MCP tool not found"

**Cause:** Server not registered  
**Fix:**
```bash
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

### Vision not working

**Cause:** Missing base64  
**Fix:** Use `include_base64: true`

---

## 🎯 Pro Tips

1. **Add "W" between inputs** - Games need timing!
2. **Save before risky stuff** - Use `emulator_save_state`
3. **Check game state** - Know where you are before deciding
4. **Use vision for complex situations** - Menus, battles, exploration
5. **Track your resources** - Money, HP, items matter!

---

## 📁 Quick Reference

```
ai-Py-boy-emulation-main/
├── ai-game-server/mcp_server.py    # MCP endpoint
├── saves/                          # Save states
└── skills/                         # Skill docs
    ├── pyboy/SKILL.md
    └── duckbot/SKILL.md
```

---

## 🦆 You're Ready!

```
🦆 QUACK! Let's play! 🦆

The world of Kanto awaits!
Your goal: Become the Champion!

Start with:
1. Get screen
2. Analyze with [SELECT_VISION_MODEL]
3. Press buttons
4. Repeat!

Go forth and conquer!
```

*Autonomous gameplay powered by OpenClaw + Bailian AI*