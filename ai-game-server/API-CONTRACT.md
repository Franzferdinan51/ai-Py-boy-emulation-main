# API Contract Documentation

**Version:** 1.0.0  
**Last Updated:** 2026-03-19  
**Status:** Stable

This document defines the stable JSON response shapes for the PyBoy backend API endpoints.

---

## Core Principles

1. **Stable shapes** - Response structure never changes between versions
2. **Safe defaults** - All endpoints return valid JSON even when no ROM is loaded
3. **No placeholder lies** - If data isn't available, fields are empty/null, not fake data
4. **Consistent timestamps** - All responses include ISO 8601 timestamp
5. **Loaded indicator** - `loaded` boolean indicates if ROM is active

---

## Response Shapes

### Party Endpoint

**GET `/api/party`**

Returns party Pokemon information.

```json
{
  "party_count": 3,
  "party": [
    {
      "slot": 1,
      "species_id": 25,
      "species_name": "Pikachu",
      "level": 12,
      "hp": 35,
      "max_hp": 40,
      "status": 0,
      "status_text": "OK",
      "type1": "Electric",
      "type2": null,
      "moves": [],
      "hp_percent": 87.5
    }
  ],
  "timestamp": "2026-03-19T20:00:00.000000"
}
```

**Empty/Loading Response:**
```json
{
  "party_count": 0,
  "party": [],
  "timestamp": "2026-03-19T20:00:00.000000"
}
```

---

### Inventory Endpoint

**GET `/api/inventory`**

Returns player money and items.

```json
{
  "money": 12345,
  "money_formatted": "¥12,345",
  "item_count": 5,
  "items": [
    {
      "slot": 1,
      "id": 4,
      "name": "Poké Ball",
      "quantity": 10
    },
    {
      "slot": 2,
      "id": 19,
      "name": "Potion",
      "quantity": 3
    }
  ],
  "timestamp": "2026-03-19T20:00:00.000000"
}
```

**Empty/Loading Response:**
```json
{
  "money": 0,
  "money_formatted": "¥0",
  "item_count": 0,
  "items": [],
  "timestamp": "2026-03-19T20:00:00.000000"
}
```

---

### Memory Watch Endpoint

**GET `/api/memory/watch`**

Returns watched memory addresses and their current values.

```json
{
  "addresses": [],
  "values": [],
  "timestamp": "2026-03-19T20:00:00.000000",
  "frame_count": 12345
}
```

**Note:** This endpoint is a stub for future implementation. Currently returns empty arrays.

---

### Spatial Position Endpoint

**GET `/api/spatial/position`**

Returns player position and map information.

```json
{
  "x": 12,
  "y": 8,
  "map_id": 0,
  "map_name": "Pallet Town",
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": true
}
```

**Empty/Loading Response:**
```json
{
  "x": 0,
  "y": 0,
  "map_id": 0,
  "map_name": "none",
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": false
}
```

---

### Spatial Minimap Endpoint

**GET `/api/spatial/minimap`**

Returns minimap tile data (sparse).

```json
{
  "width": 20,
  "height": 18,
  "tiles": [],
  "player": {
    "x": 12,
    "y": 8
  },
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": true,
  "note": "Sparse minimap - tile data not implemented"
}
```

**Empty/Loading Response:**
```json
{
  "width": 0,
  "height": 0,
  "tiles": [],
  "player": {"x": 0, "y": 0},
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": false
}
```

---

### Spatial NPCs Endpoint

**GET `/api/spatial/npcs`**

Returns nearby NPCs (including enemy Pokemon in battle).

```json
{
  "npcs": [
    {
      "id": 25,
      "name": "Pikachu",
      "x": -1,
      "y": -1,
      "type": "enemy_pokemon",
      "level": 5,
      "hp_percent": 45.0
    }
  ],
  "count": 1,
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": true,
  "note": "Battle NPCs only - sprite memory reading not implemented"
}
```

**Empty/Loading Response:**
```json
{
  "npcs": [],
  "count": 0,
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": false
}
```

---

### Spatial Strategy Endpoint

**GET `/api/spatial/strategy`**

Returns strategic analysis and recommendations.

```json
{
  "status": "Exploring with 3 Pokemon",
  "health": {
    "party_healthy": true,
    "lowest_hp_percent": 75.0,
    "needs_healing": false
  },
  "battle": {
    "in_battle": false,
    "recommendation": "none"
  },
  "recommendations": [
    "Money: ¥12,345"
  ],
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": true
}
```

**Battle Example:**
```json
{
  "status": "In battle vs Rattata",
  "health": {
    "party_healthy": true,
    "lowest_hp_percent": 85.0,
    "needs_healing": false
  },
  "battle": {
    "in_battle": true,
    "recommendation": "attack",
    "enemy": {
      "species_id": 19,
      "species_name": "Rattata",
      "level": 3,
      "hp": 12,
      "max_hp": 15,
      "hp_percent": 80.0
    }
  },
  "recommendations": [
    "Money: ¥500",
    "Battle: attack"
  ],
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": true
}
```

**Empty/Loading Response:**
```json
{
  "status": "no_rom",
  "health": {
    "party_healthy": true,
    "lowest_hp_percent": 100,
    "needs_healing": false
  },
  "battle": {
    "in_battle": false,
    "recommendation": "none"
  },
  "recommendations": [],
  "timestamp": "2026-03-19T20:00:00.000000",
  "loaded": false
}
```

---

## Game State Endpoints

### Game State

**GET `/api/game/state`**

Returns current game state.

```json
{
  "rom_loaded": true,
  "active_emulator": "pyboy",
  "rom_path": "/path/to/rom.gb",
  "rom_name": "Pokemon Red",
  "frame_count": 12345,
  "fps": 60,
  "speed_multiplier": 1.0,
  "ai_providers": {...}
}
```

### Agent Status

**GET `/api/agent/status`**

Returns AI agent status.

```json
{
  "mode": "manual",
  "enabled": false,
  "last_decision": null,
  "last_action": null
}
```

### Screen

**GET `/api/screen`**

Returns base64-encoded screen image.

```json
{
  "image": "base64_encoded_jpeg...",
  "shape": [144, 160, 3],
  "timestamp": 1234567890.123,
  "pyboy_frame": 12345,
  "performance": {
    "total_time_ms": 5.23,
    "conversion_time_ms": 2.1,
    "current_fps": 60.0,
    "adaptive_fps_target": 60
  }
}
```

---

## Action Endpoints

### Button/Action

**POST `/api/game/button`**  
**POST `/api/game/action`**  
**POST `/api/action`**

Execute a button press or action.

**Request:**
```json
{
  "button": "A",
  "frames": 1
}
```

**Response:**
```json
{
  "message": "Action executed successfully",
  "action": "A",
  "frames": 1,
  "history_length": 42
}
```

**Valid Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `START`, `SELECT`, `NOOP`

---

## Error Responses

All endpoints return safe JSON on error:

```json
{
  "error": "Error description",
  "timestamp": "2026-03-19T20:00:00.000000"
}
```

Or for endpoints with `loaded` field:

```json
{
  ...default_fields...,
  "loaded": false,
  "error": "Error description"
}
```

---

## Map ID Reference (Pokemon Red/Blue)

| ID | Name |
|----|------|
| 0x00 | Pallet Town |
| 0x01 | Viridian City |
| 0x02 | Pewter City |
| 0x03 | Cerulean City |
| 0x04 | Lavender Town |
| 0x05 | Vermilion City |
| 0x06 | Celadon City |
| 0x07 | Fuchsia City |
| 0x08 | Cinnabar Island |
| 0x09 | Indigo Plateau |
| 0x0A | Saffron City |
| 0x38 | Route 1 |
| 0x39 | Route 2 |
| 0x3A | Route 3 |

---

## Testing

Test endpoints locally with curl:

```bash
# Health check
curl http://localhost:5002/health

# Party info
curl http://localhost:5002/api/party

# Inventory
curl http://localhost:5002/api/inventory

# Position
curl http://localhost:5002/api/spatial/position

# Strategy
curl http://localhost:5002/api/spatial/strategy

# Press A button
curl -X POST http://localhost:5002/api/game/button \
  -H "Content-Type: application/json" \
  -d '{"button": "A", "frames": 1}'
```

---

## Changelog

### v1.0.0 (2026-03-19)
- Initial stable API contract
- Added `/api/party` with Pokemon party reading
- Added `/api/inventory` with money and items
- Added `/api/memory/watch` stub
- Added `/api/spatial/position` with player coordinates
- Added `/api/spatial/minimap` (sparse implementation)
- Added `/api/spatial/npcs` (battle NPCs only)
- Added `/api/spatial/strategy` with recommendations
- All endpoints return stable JSON shapes
- Safe empty/loading responses for all endpoints