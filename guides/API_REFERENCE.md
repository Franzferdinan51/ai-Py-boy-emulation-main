# 🎮 API_REFERENCE.md - MCP Server API Reference

**Complete API documentation for the AI GameBoy MCP Server**

---

## Overview

The MCP server exposes Game Boy emulator controls as MCP (Model Context Protocol) tools. Agents can load ROMs, press buttons, read game memory, and use autonomous play modes.

**Server Version:** 3.0.0  
**Location:** `ai-game-server/mcp_server.py`

---

## Base Configuration

### Starting the MCP Server

```bash
# Standard startup
cd ai-game-server
python3 mcp_server.py

# With custom port
python3 mcp_server.py --port 5003

# With logging
python3 mcp_server.py --log-level DEBUG
```

### Registering with mcporter

```bash
# Register the MCP server
mcporter add gameboy --stdio "python3 ai-game-server/mcp_server.py"

# Verify registration
mcporter list | grep gameboy

# Test connection
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | python3 ai-game-server/mcp_server.py
```

---

## MCP Tools

### Core Emulator Controls

#### `emulator_load_rom`

Load a ROM file into the emulator.

| Property | Value |
|----------|-------|
| **Description** | Load a Game Boy ROM file |
| **Parameters** | `rom_path` (string, required) - Path to ROM file |

**Example Request:**
```json
{
  "tool": "emulator_load_rom",
  "args": {"rom_path": "/path/to/pokemon-red.gb"}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "rom_loaded": "pokemon-red.gb",
    "rom_size": 1048576,
    "game_title": "POKEMON RED",
    "cgb_support": true
  },
  "timing_ms": 150
}
```

---

#### `emulator_press_button`

Press a single emulator button.

| Property | Value |
|----------|-------|
| **Description** | Press one Game Boy button |
| **Parameters** | `button` (string, required) - Button to press |

**Valid Buttons:** `A`, `B`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `START`, `SELECT`

**Example Request:**
```json
{
  "tool": "emulator_press_button",
  "args": {"button": "A"}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {"button": "A", "action": "pressed"},
  "timing_ms": 5
}
```

---

#### `emulator_press_sequence`

Press multiple buttons in sequence with timing.

| Property | Value |
|----------|-------|
| **Description** | Execute a sequence of button presses |
| **Parameters** | `sequence` (string, required) - Button sequence |

**Sequence Syntax:**
- Single press: `A`, `B`, `START`
- Hold: `A2` (hold A for 2 ticks)
- Wait: `W` (wait 1 frame), `W10` (wait 10 frames)
- Direction combos: `UP`, `DOWN`, `LEFT`, `RIGHT`
- Combined: `R2 A U3 W START`

**Example Request:**
```json
{
  "tool": "emulator_press_sequence",
  "args": {"sequence": "DOWN DOWN START"}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "sequence": "DOWN DOWN START",
    "buttons_pressed": 3,
    "total_wait": 0
  },
  "timing_ms": 25
}
```

---

#### `emulator_tick`

Advance the emulator by a specified number of frames.

| Property | Value |
|----------|-------|
| **Description** | Advance emulation by N frames |
| **Parameters** | `frames` (integer, optional, default: 1) - Number of frames |

**Example Request:**
```json
{
  "tool": "emulator_tick",
  "args": {"frames": 10}
}
```

---

#### `emulator_get_state`

Get current emulator state.

| Property | Value |
|----------|-------|
| **Description** | Get emulator status and information |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "rom_loaded": "pokemon-red.gb",
    "emulator_running": true,
    "frame_count": 12345,
    "speed": 1.0
  }
}
```

---

### Vision & Screen Capture

#### `get_screen_base64`

Get the current game screen as a base64-encoded image.

| Property | Value |
|----------|-------|
| **Description** | Get screen for AI vision analysis |
| **Parameters** | `include_base64` (boolean, optional, default: true) |

**Example Request:**
```json
{
  "tool": "get_screen_base64",
  "args": {"include_base64": true}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "screen": "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkp...",
    "width": 160,
    "height": 144,
    "format": "png",
    "size_bytes": 4523
  },
  "timing_ms": 45
}
```

**Usage with Vision AI:**
```bash
# Get screen and send to [SELECT_VISION_MODEL] for analysis
screen_data=$(get_screen_base64)
echo "$screen_data" | base64 -d > /tmp/screen.png
# Send to vision model for analysis
```

---

#### `emulator_get_frame`

Get current frame (alias for get_screen_base64).

| Property | Value |
|----------|-------|
| **Description** | Get current game frame |
| **Parameters** | `include_base64` (boolean, optional) |

---

#### `emulator_save_screenshot`

Save screen to a file.

| Property | Value |
|----------|-------|
| **Description** | Save current screen to file |
| **Parameters** | `output_path` (string, required) - File path to save |

**Example Request:**
```json
{
  "tool": "emulator_save_screenshot",
  "args": {"output_path": "/tmp/screenshot.png"}
}
```

---

### Memory Reading

#### `get_player_position`

Read player X,Y coordinates from game memory.

| Property | Value |
|----------|-------|
| **Description** | Get player position in game world |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "x": 12,
    "y": 8,
    "map_id": 38,
    "map_name": "Pallet Town"
  }
}
```

---

#### `get_party_info`

Read party Pokemon/monsters from memory.

| Property | Value |
|----------|-------|
| **Description** | Get all Pokemon in party |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "party": [
      {
        "slot": 0,
        "species_id": 4,
        "species_name": "Charmander",
        "level": 5,
        "current_hp": 20,
        "max_hp": 20,
        "attack": 12,
        "defense": 9,
        "speed": 10,
        "moves": ["Scratch", "Growl", "Tail Whip", "Ember"]
      }
    ],
    "party_count": 1
  }
}
```

---

#### `get_inventory`

Read inventory/bag items from memory.

| Property | Value |
|----------|-------|
| **Description** | Get items in player's bag |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "items": [
      {"item_id": 13, "name": "Potion", "quantity": 5},
      {"item_id": 5, "name": "Poke Ball", "quantity": 10}
    ],
    "item_count": 2
  }
}
```

---

#### `get_map_location`

Read current map/location ID from memory.

| Property | Value |
|----------|-------|
| **Description** | Get current location |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "map_id": 38,
    "map_name": "Pallet Town",
    "area_type": "town"
  }
}
```

---

#### `get_money`

Read player money/currency from memory.

| Property | Value |
|----------|-------|
| **Description** | Get player money |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "money": 3000,
    "formatted": "$3,000"
  }
}
```

---

#### `emulator_read_memory`

Read raw RAM at a specific address.

| Property | Value |
|----------|-------|
| **Description** | Read raw memory at address |
| **Parameters** | `address` (string, required) - Hex address (e.g., "0xD000") |
| | `length` (integer, optional) - Number of bytes |

**Example Request:**
```json
{
  "tool": "emulator_read_memory",
  "args": {"address": "0xD000", "length": 16}
}
```

---

#### `emulator_get_game_state`

Get complete game state snapshot.

| Property | Value |
|----------|-------|
| **Description** | Get full game state |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "player": {
      "x": 12,
      "y": 8,
      "money": 3000,
      "badges": []
    },
    "party": [...],
    "inventory": [...],
    "map_id": 38,
    "game_time": 1234
  }
}
```

---

### Save States

#### `save_game_state`

Save current emulator state to file.

| Property | Value |
|----------|-------|
| **Description** | Save game progress |
| **Parameters** | `save_name` (string, optional) - Save identifier |

**Example Request:**
```json
{
  "tool": "save_game_state",
  "args": {"save_name": "my-checkpoint"}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "save_name": "my-checkpoint",
    "filepath": "saves/duckbot_my-checkpoint.state",
    "size_bytes": 2048
  }
}
```

---

#### `load_game_state`

Load a previously saved state.

| Property | Value |
|----------|-------|
| **Description** | Load saved game |
| **Parameters** | `save_name` (string, required) - Save identifier |

**Example Request:**
```json
{
  "tool": "load_game_state",
  "args": {"save_name": "my-checkpoint"}
}
```

---

#### `emulator_list_saves`

List all available save files.

| Property | Value |
|----------|-------|
| **Description** | List all saves |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "saves": [
      {"name": "my-checkpoint", "created": "2026-03-19T10:00:00"},
      {"name": "before-battle", "created": "2026-03-19T10:15:00"}
    ]
  }
}
```

---

### Auto-Play Modes

#### `auto_battle`

Let AI automatically fight Pokemon.

| Property | Value |
|----------|-------|
| **Description** | AI-controlled battle |
| **Parameters** | `max_moves` (integer, optional) - Max battle turns |

**Example Request:**
```json
{
  "tool": "auto_battle",
  "args": {"max_moves": 10}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "moves_executed": 5,
    "battle_result": "won",
    "exp_gained": 25,
    "actions": [
      {"move": "Ember", "damage": 10, "result": "hit"},
      {"move": "Scratch", "damage": 6, "result": "hit"}
    ]
  }
}
```

---

#### `auto_explore`

Autonomous world exploration.

| Property | Value |
|----------|-------|
| **Description** | AI explores game world |
| **Parameters** | `steps` (integer, optional) - Steps to take |

**Example Request:**
```json
{
  "tool": "auto_explore",
  "args": {"steps": 20}
}
```

---

#### `auto_grind`

Grind for XP or money.

| Property | Value |
|----------|-------|
| **Description** | Farm XP or money |
| **Parameters** | `target_level` (integer, optional) - Level to reach |
| | `max_battles` (integer, optional) - Max battles |

**Example Request:**
```json
{
  "tool": "auto_grind",
  "args": {"target_level": 20, "max_battles": 50}
}
```

---

### Session Management

#### `session_start`

Start a new agent session.

| Property | Value |
|----------|-------|
| **Description** | Create new session for tracking progress |
| **Parameters** | `goal` (string, required) - Session goal |

**Example Request:**
```json
{
  "tool": "session_start",
  "args": {"goal": "Beat Elite 4 and become Champion"}
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "session_abc123",
    "goal": "Beat Elite 4 and become Champion",
    "created_at": "2026-03-19T10:00:00"
  }
}
```

---

#### `session_get`

Get session data.

| Property | Value |
|----------|-------|
| **Description** | Retrieve session value |
| **Parameters** | `session_id` (string, required) |
| | `key` (string, required) - Data key |

**Example Request:**
```json
{
  "tool": "session_get",
  "args": {"session_id": "session_abc123", "key": "goal"}
}
```

---

#### `session_set`

Store data in session.

| Property | Value |
|----------|-------|
| **Description** | Save session data |
| **Parameters** | `session_id` (string, required) |
| | `key` (string, required) - Data key |
| | `value` (any, required) - Data value |

**Example Request:**
```json
{
  "tool": "session_set",
  "args": {
    "session_id": "session_abc123",
    "key": "visited_locations",
    "value": ["Pallet Town", "Viridian City", "Pewter City"]
  }
}
```

---

#### `session_list`

List all sessions.

| Property | Value |
|----------|-------|
| **Description** | List all sessions |
| **Parameters** | None |

---

#### `session_delete`

Delete a session.

| Property | Value |
|----------|-------|
| **Description** | Delete session |
| **Parameters** | `session_id` (string, required) |

---

## HTTP API Endpoints

The server also exposes HTTP endpoints for non-MCP access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/api/state` | GET | Get emulator state |
| `/api/screen` | GET | Get screen image |
| `/api/input` | POST | Send input |
| `/api/save` | POST | Save state |
| `/api/load` | POST | Load state |

---

## Error Codes

| Code | Meaning | Resolution |
|------|---------|-------------|
| `ROM_NOT_FOUND` | ROM file doesn't exist | Check file path |
| `INVALID_BUTTON` | Invalid button pressed | Use valid button |
| `NO_EMULATOR` | Emulator not running | Load ROM first |
| `SAVE_NOT_FOUND` | Save file doesn't exist | Check save name |
| `MEMORY_ERROR` | Memory read failed | Try different address |
| `SESSION_NOT_FOUND` | Session doesn't exist | Create session first |

---

## Response Format

All responses follow this format:

```json
{
  "success": true,
  "timestamp": "2026-03-19T10:00:00.000Z",
  "server": {
    "version": "3.0.0",
    "started": "2026-03-19T09:00:00.000Z"
  },
  "data": { ... },
  "timing_ms": 15
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "ROM_NOT_FOUND",
    "message": "ROM file not found",
    "suggestion": "Check the file path is correct"
  },
  "timing_ms": 5
}
```

---

## Related Documentation

- [DECISION_TREE.md](DECISION_TREE.md) - How agents make decisions
- [EXAMPLES.md](EXAMPLES.md) - Example prompts and sessions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [README.md](../README.md) - Main documentation