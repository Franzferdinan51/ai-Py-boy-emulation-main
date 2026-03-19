# 🎮 API_REFERENCE.md - MCP Server API Reference

**Complete API documentation for the AI GameBoy MCP Server**

---

## Overview

The MCP server exposes Game Boy emulator controls as MCP (Model Context Protocol) tools. Agents can load ROMs, press buttons, read game memory, and use autonomous play modes.

**Server Version:** 5.0.0  
**Location:** `ai-game-server/mcp_server.py`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [MCP Tools](#mcp-tools)
   - [Core Emulator Controls](#core-emulator-controls)
   - [Vision & Screen Capture](#vision--screen-capture)
   - [Memory Reading](#memory-reading)
   - [Save States](#save-states)
   - [Auto-Play Modes](#auto-play-modes)
   - [Session Management](#session-management)
3. [HTTP API Endpoints](#http-api-endpoints)
4. [Error Codes](#error-codes)
5. [Response Format](#response-format)

---

## Quick Start

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
    "rom": "/path/to/pokemon-red.gb",
    "frame": 0
  },
  "tool": "emulator_load_rom"
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
  "data": {"button": "A", "frame": 1},
  "tool": "emulator_press_button"
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
  "initialized": true,
  "rom_path": "/path/to/pokemon-red.gb",
  "frame_count": 12345,
  "pyboy_available": true
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
  "frame": 100,
  "dimensions": {"width": 160, "height": 144},
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "timing_ms": 45
}
```

---

### Memory Reading

#### `get_memory_address`

Read a specific memory address.

| Property | Value |
|----------|-------|
| **Description** | Read memory at address with description |
| **Parameters** | `address` (integer or string, required) - Memory address (e.g., 54370 or "0xD062") |

**Example Request:**
```json
{
  "tool": "get_memory_address",
  "args": {"address": "0xD062"}
}
```

**Example Response:**
```json
{
  "success": true,
  "tool": "get_memory_address",
  "data": {
    "address": "0xd062",
    "value": 12,
    "value_hex": "0xc",
    "value_binary": "0b1100",
    "description": "Player X position = 12"
  },
  "frame": 100
}
```

---

#### `set_memory_address`

Write to a specific memory address (⚠️ can corrupt game state!).

| Property | Value |
|----------|-------|
| **Description** | Write byte to memory address |
| **Parameters** | `address` (integer or string, required), `value` (integer 0-255, required) |

**Example Request:**
```json
{
  "tool": "set_memory_address",
  "args": {"address": "0xD062", "value": 20}
}
```

---

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
  "tool": "get_player_position",
  "data": {"x": 12, "y": 8}
}
```

---

#### `get_party_pokemon`

Get detailed party Pokemon information.

| Property | Value |
|----------|-------|
| **Description** | Get all Pokemon in party with stats |
| **Parameters** | None |

**Example Response:**
```json
{
  "success": true,
  "tool": "get_party_pokemon",
  "data": {
    "party_count": 1,
    "party": [
      {
        "slot": 1,
        "species_id": 4,
        "species_name": "Charmander",
        "level": 5,
        "current_hp": 20,
        "max_hp": 20,
        "hp_percent": 100.0,
        "status": "healthy"
      }
    ],
    "summary": {
      "total_hp": "20/20",
      "health_percent": 100.0,
      "average_level": 5.0,
      "status": "ready"
    }
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
  "tool": "get_inventory",
  "data": {
    "unique_items": 2,
    "total_items": 15,
    "items": [
      {"slot": 1, "item_id": 13, "item_name": "Potion", "quantity": 5, "category": "healing"},
      {"slot": 2, "item_id": 25, "item_name": "Poke Ball", "quantity": 10, "category": "poke_balls"}
    ],
    "summary": {
      "poke_balls": 10,
      "healing_items": 5,
      "other_items": 0
    }
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
  "tool": "get_map_location",
  "data": {"map_id": 38, "map_hex": "0x26"}
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
  "tool": "get_money",
  "data": {"money": 3000, "formatted": "$3,000"}
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
  "path": "saves/my-checkpoint.state",
  "frame": 12345
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

### Auto-Play Modes

#### `auto_explore_mode`

Autonomous world exploration.

| Property | Value |
|----------|-------|
| **Description** | AI explores game world |
| **Parameters** | `steps` (integer, optional, default: 20) - Steps to take |
| | `avoid_battles` (boolean, optional, default: true) - Avoid wild battles |
| | `session_id` (string, optional) - Session for tracking |

**Example Request:**
```json
{
  "tool": "auto_explore_mode",
  "args": {"steps": 20, "avoid_battles": true}
}
```

---

#### `auto_battle_mode`

AI-powered battle mode.

| Property | Value |
|----------|-------|
| **Description** | AI fights in battles |
| **Parameters** | `max_moves` (integer, optional, default: 15) - Max battle turns |
| | `strategy` (string, optional, default: "aggressive") - "aggressive", "defensive", or "balanced" |

**Example Request:**
```json
{
  "tool": "auto_battle_mode",
  "args": {"max_moves": 10, "strategy": "aggressive"}
}
```

---

### Session Management

#### `session_start`

Start a new agent session.

| Property | Value |
|----------|-------|
| **Description** | Create new session for tracking progress |
| **Parameters** | `session_id` (string, optional) - Custom session ID |
| | `goal` (string, optional) - Session goal |
| | `ttl_seconds` (integer, optional, default: 3600) - Session TTL |

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
  "tool": "session_start",
  "data": {
    "session_id": "session_1710844800000",
    "message": "New session created (persisted to disk)",
    "goal": "Beat Elite 4 and become Champion",
    "ttl_seconds": 3600
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
| | `key` (string, optional) - Specific data key |

**Example Request:**
```json
{
  "tool": "session_get",
  "args": {"session_id": "session_1710844800000", "key": "goal"}
}
```

---

#### `session_set`

Store data in session.

| Property | Value |
|----------|-------|
| **Description** | Save session data (persisted to disk) |
| **Parameters** | `session_id` (string, required) |
| | `key` (string, required) - Data key |
| | `value` (any, required) - Data value |

**Example Request:**
```json
{
  "tool": "session_set",
  "args": {
    "session_id": "session_1710844800000",
    "key": "visited_locations",
    "value": ["Pallet Town", "Viridian City"]
  }
}
```

---

#### `session_list`

List all sessions.

| Property | Value |
|----------|-------|
| **Description** | List all active sessions |
| **Parameters** | None |

---

#### `session_delete`

Delete a session.

| Property | Value |
|----------|-------|
| **Description** | Delete session and remove from disk |
| **Parameters** | `session_id` (string, required) |

---

## HTTP API Endpoints

The server also exposes HTTP endpoints for non-MCP access:

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/api/status` | GET | Get server status |
| `/api/config` | GET | Get configuration |
| `/api/config/validate` | GET | Validate configuration |

### Emulator Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/load_rom` | POST | Load ROM file (body: `{"rom_path": "..."}`) |
| `/api/upload-rom` | POST | Upload ROM file (multipart form) |
| `/api/action` | POST | Send button input (body: `{"action": "A"}`) |
| `/api/game/button` | POST | Press button (body: `{"button": "A"}`) |
| `/api/ai-action` | POST | Get AI-recommended action |
| `/api/screen` | GET | Get screen as JSON with base64 image |

### Game State

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/game/state` | GET | Get game state (running, rom_name, etc.) |
| `/api/agent/status` | GET | Get agent status (mode, current_action, etc.) |
| `/api/agent/mode` | GET/POST | Get or set agent mode |

### Memory & State

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/<address>` | GET | Read memory address (query: `size`, `format`) |
| `/api/memory/<address>` | POST | Write to memory (body: `{"value": 255}`) |
| `/api/memory/watch` | GET | Get watched memory addresses |
| `/api/party` | GET | Get party Pokemon info |
| `/api/inventory` | GET | Get inventory items |

### Save States

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/save_state` | POST | Save game state |
| `/api/load_state` | POST | Load game state |
| `/save_state` | POST | Alternative save endpoint |
| `/load_state` | POST | Alternative load endpoint |

### AI & Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat with AI |
| `/api/providers/status` | GET | AI provider status |
| `/api/models` | GET | Available AI models |

### UI Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ui/launch` | POST | Launch UI |
| `/api/ui/stop` | POST | Stop UI |
| `/api/ui/restart` | POST | Restart UI |
| `/api/ui/status` | GET | UI status |

### Performance

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/performance` | GET | Performance metrics |
| `/api/emulator/mode` | GET | Emulator mode |
| `/api/emulator/clear-cache` | POST | Clear emulator cache |

---

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `EMULATOR_NOT_INITIALIZED` | Emulator not running | Load ROM first |
| `ROM_NOT_FOUND` | ROM file doesn't exist | Check file path |
| `INVALID_ROM` | Invalid ROM format | Use valid .gb/.gbc/.gba |
| `BUTTON_INVALID` | Invalid button pressed | Use valid button |
| `MEMORY_READ_ERROR` | Memory read failed | Try different address |
| `MEMORY_WRITE_ERROR` | Memory write failed | Address may be read-only |
| `SAVE_NOT_FOUND` | Save file doesn't exist | Check save name |
| `SESSION_NOT_FOUND` | Session doesn't exist | Create session first |
| `SESSION_EXPIRED` | Session TTL exceeded | Create new session |
| `INVALID_PARAMETER` | Invalid parameter | Check parameter values |
| `OPERATION_FAILED` | Operation failed | Check logs for details |
| `BATTLE_NOT_ACTIVE` | Not in a battle | Enter battle first |
| `INVALID_ADDRESS` | Memory address out of range | Use 0x0000-0xFFFF |
| `STREAM_ERROR` | Screen streaming failed | Try non-streaming method |

---

## Response Format

All MCP tool responses follow this format:

```json
{
  "success": true,
  "tool": "tool_name",
  "data": { ... },
  "frame": 12345,
  "timing_ms": 15.5
}
```

Error responses:

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "suggestions": ["Suggestion 1", "Suggestion 2"],
  "recoverable": true
}
```

---

## Related Documentation

- [DECISION_TREE.md](DECISION_TREE.md) - How agents make decisions
- [VISION_GUIDE.md](VISION_GUIDE.md) - Vision AI integration
- [EXAMPLES.md](../EXAMPLES.md) - Example prompts and sessions
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Common issues
- [README.md](../README.md) - Main documentation
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide for new users