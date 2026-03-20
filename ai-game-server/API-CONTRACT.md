# API Contract - Model Discovery & Settings

**Last Updated:** March 19, 2026

This document describes the backend routes that power settings/model selection in the AI-Py-Boy platform.

## Overview

All model/provider endpoints now return consistent metadata for frontend consumption:

- `id`: Unique identifier (used in API calls)
- `name`: Short display name
- `label`: Full display name for dropdowns (includes category suffix)
- `provider`: Provider family
- `category`: `vision`, `reasoning`, or `general`
- `capabilities`: Array of capabilities (`vision`, `reasoning`, `text`)
- `is_vision_capable`: Boolean for quick filtering
- `is_free`: Boolean indicating free/unlimited usage
- `manual_allowed`: Boolean - can user enter custom model ID?
- `is_default`: Boolean - is this the default for this provider?
- `context_window`: Estimated context window size
- `description`: Human-readable description

---

## Endpoints

### GET `/api/providers`

**Purpose:** Get all available AI providers with their models. Primary endpoint for settings UI.

**Query Params:** None

**Response:**

```json
{
  "providers": [
    {
      "id": "openclaw",
      "name": "OpenClaw Gateway",
      "status": "available",
      "available": true,
      "manual_allowed": true,
      "priority": 1,
      "error": null,
      "models": [
        {
          "id": "bailian/kimi-k2.5",
          "name": "Kimi K2.5",
          "label": "Kimi K2.5 (Vision)",
          "provider": "bailian",
          "category": "vision",
          "capabilities": ["vision", "reasoning", "text"],
          "is_vision_capable": true,
          "is_free": true,
          "manual_allowed": true,
          "is_default": true,
          "context_window": 196608,
          "description": "Best for game screen analysis (FREE)"
        }
      ],
      "default_model": "bailian/kimi-k2.5"
    },
    {
      "id": "lmstudio",
      "name": "LM Studio (Local)",
      "status": "available",
      "available": true,
      "manual_allowed": true,
      "priority": 2,
      "models": [...]
    }
  ],
  "default_provider": "openclaw",
  "manual_allowed": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET `/api/models`

**Purpose:** Get models for a specific provider (when `?provider=X`) or all providers (when no query param).

**Query Params:**
- `provider` (optional): Provider name to filter by

**Response (specific provider):**

```json
{
  "provider": "lmstudio",
  "name": "LM Studio (Local)",
  "status": "available",
  "available": true,
  "manual_allowed": true,
  "models": [
    {
      "id": "qwen3-vl-8b",
      "name": "Qwen3 VL 8B",
      "label": "Qwen3 VL 8B",
      "provider": "lmstudio",
      "category": "vision",
      "capabilities": ["vision", "reasoning", "text"],
      "is_vision_capable": true,
      "is_free": true,
      "manual_allowed": true,
      "is_default": false,
      "context_window": 8192,
      "description": "Vision model for screen analysis"
    }
  ],
  "default_model": "qwen3-vl-8b",
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET `/api/openclaw/models`

**Purpose:** Get all models available through OpenClaw Gateway.

**Query Params:**
- `refresh` (optional): Force cache refresh (`true`/`false`, default: `false`)

**Response:**

```json
{
  "provider": "openclaw",
  "name": "OpenClaw Gateway",
  "status": "available",
  "available": true,
  "manual_allowed": true,
  "models": [
    {
      "id": "bailian/kimi-k2.5",
      "name": "Kimi K2.5",
      "label": "Kimi K2.5 (Vision)",
      "provider": "bailian",
      "category": "vision",
      "capabilities": ["vision", "reasoning", "text"],
      "is_vision_capable": true,
      "is_free": true,
      "manual_allowed": true,
      "is_default": true,
      "context_window": 196608,
      "priority": 100,
      "description": "Best for game screen analysis (FREE)"
    }
  ],
  "default_model": "bailian/kimi-k2.5",
  "timestamp": "2026-03-19T20:00:00Z",
  "cached": true
}
```

---

### GET `/api/openclaw/models/vision`

**Purpose:** Get only vision-capable models from OpenClaw.

**Response:** Same shape as `/api/openclaw/models` but filtered to `is_vision_capable: true`.

```json
{
  "provider": "openclaw",
  "name": "OpenClaw Vision Models",
  "category": "vision",
  "models": [...],
  "default_model": "bailian/kimi-k2.5",
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET `/api/openclaw/models/planning`

**Purpose:** Get models suitable for planning/decision making.

**Response:** Same shape as `/api/openclaw/models` but filtered to reasoning models.

```json
{
  "provider": "openclaw",
  "name": "OpenClaw Planning Models",
  "category": "planning",
  "models": [...],
  "default_model": "bailian/glm-5",
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET `/api/openclaw/models/recommend`

**Purpose:** Get model recommendation for a specific use case.

**Query Params:**
- `use_case`: One of `vision`, `planning`, `fast`, `quality`, `free` (default: `planning`)

**Response:**

```json
{
  "recommended": {
    "id": "bailian/kimi-k2.5",
    "name": "Kimi K2.5",
    "label": "Kimi K2.5 (Vision)",
    "provider": "bailian",
    "category": "vision",
    "is_vision_capable": true,
    "is_free": true,
    "description": "Best for game screen analysis (FREE)",
    "is_default": true
  },
  "use_case": "vision",
  "reason": "Best vision model available (Kimi K2.5)",
  "alternatives": [
    {
      "id": "bailian/qwen3.5-plus",
      "name": "Qwen 3.5 Plus",
      "provider": "bailian",
      "is_free": false,
      "is_vision_capable": true
    }
  ],
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET/POST `/api/ai/runtime`

**Purpose:** Get or set the current AI runtime configuration.

**GET Response:**

```json
{
  "state": {
    "provider": "openclaw",
    "model": "bailian/kimi-k2.5",
    "api_endpoint": "http://localhost:18789"
  },
  "available_providers": ["openclaw", "lmstudio", "gemini"],
  "provider_status": {
    "openclaw": {"status": "available", "available": true},
    "lmstudio": {"status": "available", "available": true}
  },
  "default_provider": "openclaw",
  "manual_allowed": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**POST Body:**

```json
{
  "provider": "lmstudio",
  "model": "qwen3-vl-8b",
  "api_endpoint": "http://localhost:1234/v1"
}
```

**POST Response:**

```json
{
  "ok": true,
  "state": {
    "provider": "lmstudio",
    "model": "qwen3-vl-8b",
    "api_endpoint": "http://localhost:1234/v1"
  },
  "message": "Runtime updated: provider=lmstudio, model=qwen3-vl-8b",
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

### GET/POST `/api/openclaw/config`

**Purpose:** Get or set OpenClaw-specific configuration.

**GET Response:**

```json
{
  "endpoint": "http://localhost:18789",
  "dual_model": {
    "enabled": true,
    "vision_model": "bailian/kimi-k2.5",
    "planning_model": "bailian/glm-5"
  },
  "status": "available",
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**POST Body:**

```json
{
  "endpoint": "http://localhost:18789",
  "vision_model": "bailian/kimi-k2.5",
  "planning_model": "bailian/glm-5",
  "use_dual_model": true
}
```

---

### GET `/api/ai/settings`

**Purpose:** Comprehensive settings endpoint combining all AI configuration. Use for initial settings page load.

**Response:**

```json
{
  "runtime": {
    "provider": "openclaw",
    "model": "bailian/kimi-k2.5",
    "api_endpoint": "http://localhost:18789"
  },
  "providers": [
    {
      "id": "openclaw",
      "name": "OpenClaw Gateway",
      "status": "available",
      "available": true,
      "manual_allowed": true,
      "models": [...],
      "default_model": "bailian/kimi-k2.5"
    }
  ],
  "default_provider": "openclaw",
  "openclaw": {
    "endpoint": "http://localhost:18789",
    "status": "available",
    "models_count": 5
  },
  "dual_model": {
    "enabled": true,
    "vision_model": "bailian/kimi-k2.5",
    "planning_model": "bailian/glm-5",
    "available": true
  },
  "manual_allowed": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

---

## Provider Categories

| Category | Description | Example Models |
|----------|-------------|----------------|
| `vision` | Vision/image analysis | kimi-k2.5, qwen3-vl-8b, llava |
| `reasoning` | Text reasoning/planning | glm-5, qwen3.5-plus, MiniMax-M2.5 |
| `general` | General purpose | Default fallback |

---

## Manual Model Entry

All providers support `manual_allowed: true`, meaning users can enter custom model IDs not in the discovered list.

**Frontend Implementation:**

1. Show dropdown with discovered models
2. Add "Custom model..." option
3. When selected, show text input for manual model ID
4. Submit the manual ID as `model` in POST requests

---

## Backward Compatibility

These changes are **backward compatible**:

- Existing endpoints still work
- New fields are additive (don't break existing code)
- `/api/models?provider=X` returns enhanced response
- `/api/providers/status` unchanged for compatibility

---

## Frontend Usage Examples

### Populate Settings Dropdown

```typescript
// Fetch all providers with models
const response = await fetch('/api/providers');
const data = await response.json();

// Build dropdown options
data.providers.forEach(provider => {
  if (provider.available) {
    provider.models.forEach(model => {
      console.log(`${model.label} (${model.id})`);
    });
  }
});
```

### Set Runtime Model

```typescript
await fetch('/api/ai/runtime', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    provider: 'lmstudio',
    model: 'qwen3-vl-8b'
  })
});
```

### Get Vision Models Only

```typescript
const response = await fetch('/api/openclaw/models/vision');
const data = await response.json();

// data.models contains only vision-capable models
```

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "error": "Provider 'unknown' not found",
  "available_providers": ["openclaw", "lmstudio", "gemini"]
}
```

---

---

## Agent Tools API (AI Gameplay)

These endpoints are designed for AI agents to understand game state and make decisions. They provide structured game information beyond raw screen data.

### GET `/api/agent/context`

**Purpose:** Get comprehensive agent context for AI decision making. Returns a complete snapshot of game state.

**Response:**

```json
{
  "loaded": true,
  "rom_name": "Pokemon Red",
  "frame": 12345,
  "game_mode": "exploration",
  "position": {
    "x": 10,
    "y": 20,
    "map_id": 1,
    "map_name": "Pallet Town"
  },
  "party": {
    "count": 1,
    "pokemon": [
      {"species": "Charmander", "level": 5, "hp": 35, "max_hp": 35, "hp_percent": 100}
    ]
  },
  "inventory": {
    "money": 500,
    "items": [
      {"name": "Potion", "count": 5},
      {"name": "Poke Ball", "count": 10}
    ]
  },
  "battle": {
    "in_battle": false,
    "enemy": null
  },
  "health_summary": {
    "party_healthy": true,
    "lowest_hp_percent": 100,
    "needs_healing": false
  },
  "recommendations": [],
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**Why this helps agents:**
- Provides complete game state in one call (no need for multiple API calls)
- Includes actionable recommendations ("Heal party at Pokemon Center")
- Returns safe defaults when data is unavailable
- Agents can make decisions without needing vision

---

### GET `/api/agent/mode`

**Purpose:** Get current game mode/state for quick decision making.

**Response:**

```json
{
  "mode": "exploration",
  "in_battle": false,
  "in_menu": false,
  "in_dialogue": false,
  "details": {
    "battle_type": "none",
    "menu_type": "none",
    "dialogue_active": false
  },
  "loaded": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**Game Modes:**
- `exploration` - Moving around the world
- `battle` - In a Pokemon battle
- `menu` - In a menu (bag, Pokemon, etc.)
- `dialogue` - Reading text
- `title` - Title screen
- `none` - No ROM loaded

**Why this helps agents:**
- Quick check for state machine transitions
- Determines what actions are valid (can't move in menu)
- Battle detection triggers different decision logic

---

### POST `/api/agent/act`

**Purpose:** Execute an action and observe the result in one call. Combines button press + observation.

**Request Body:**

```json
{
  "action": "A",
  "frames": 1
}
```

**Valid Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `START`, `SELECT`

**Response:**

```json
{
  "success": true,
  "action": "A",
  "frames": 1,
  "result": {
    "game_mode": "exploration",
    "position_changed": false,
    "text_appeared": false,
    "battle_started": false,
    "menu_opened": false
  },
  "context": {
    "position": {"x": 10, "y": 20},
    "in_battle": false
  },
  "frame": 12346,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**Why this helps agents:**
- Single round-trip for action + feedback
- Detects state changes (battle started, text appeared, etc.)
- Agents can react immediately to game responses

---

### GET `/api/agent/dialogue`

**Purpose:** Get current dialogue/text box state.

**Response:**

```json
{
  "active": false,
  "text": null,
  "has_options": false,
  "options": [],
  "selected_option": 0,
  "can_advance": true,
  "loaded": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**Why this helps agents:**
- Knows when to wait for text vs. take actions
- Detects choice menus in dialogue
- Can auto-advance through text boxes

---

### GET `/api/agent/menu`

**Purpose:** Get current menu state.

**Response:**

```json
{
  "active": false,
  "type": "none",
  "selection": 0,
  "options": [],
  "can_close": true,
  "loaded": true,
  "timestamp": "2026-03-19T20:00:00Z"
}
```

**Menu Types:**
- `main` - Start menu
- `pokemon` - Pokemon selection
- `bag` - Item bag
- `battle` - Battle menu (FIGHT/PKMN/ITEM/RUN)
- `save` - Save/load menu
- `none` - No menu active

**Why this helps agents:**
- Knows what menu options are available
- Cursor position for selection
- Can close menus appropriately

---

## Agent Tool Usage with LM Studio / MCP

### MCP Server Connection

The `generic_mcp_server.py` wraps these backend routes as MCP tools for LM Studio:

```bash
# Start the MCP server (connects to backend at localhost:5002)
cd ai-game-server
python generic_mcp_server.py
```

### Example: Agent Decision Loop

```python
# Pseudocode for agent loop using these tools
import requests

API = "http://localhost:5002"

def agent_loop():
    while True:
        # 1. Get current context
        context = requests.get(f"{API}/api/agent/context").json()
        
        if not context["loaded"]:
            print("No ROM loaded")
            break
        
        # 2. Check game mode
        mode = requests.get(f"{API}/api/agent/mode").json()
        
        if mode["in_battle"]:
            # Battle logic
            attack()
        elif mode["mode"] == "exploration":
            # Exploration logic
            explore()
        
        # 3. Act and observe result
        result = requests.post(
            f"{API}/api/agent/act",
            json={"action": "A", "frames": 1}
        ).json()
        
        if result["result"]["text_appeared"]:
            # Wait for dialogue
            dialogue = requests.get(f"{API}/api/agent/dialogue").json()
            advance_dialogue()
```

### MCP Tool Mapping

When using LM Studio with the MCP server, these tools map to:

| MCP Tool | Backend Route | Purpose |
|----------|--------------|---------|
| `get_agent_context` | `/api/agent/context` | Full state snapshot |
| `get_game_mode` | `/api/agent/mode` | Current mode detection |
| `act_and_observe` | `/api/agent/act` | Action + observation |
| `get_dialogue_state` | `/api/agent/dialogue` | Text box state |
| `get_menu_state` | `/api/agent/menu` | Menu state |

---

## Cache Behavior

- OpenClaw model discovery caches results for 5 minutes
- Use `?refresh=true` to force refresh
- Check `cached` field in response to know if data is from cache