# API Contract - OpenClaw-Style Agent/Health/Status Endpoints

**Last Updated:** March 19, 2026
**Version:** 3.0.0

This document describes the OpenClaw-compatible agent, health, and status endpoints for the AI Game Boy Server.

---

## Canonical HTTP / MCP Contract Matrix

The canonical routes below are the preferred contract for new web and MCP
callers. Legacy aliases remain available for compatibility, but they should not
be treated as the primary surface.

| Canonical HTTP route | Supported aliases | High-level payload fields | No-ROM behavior | Matching generic MCP tools |
| --- | --- | --- | --- | --- |
| `GET /api/game/state` | none | `rom_loaded`, `active_emulator`, `rom_path`, `rom_name`, `frame_count`, `ai_running`, `current_goal`, `fps`, `speed_multiplier`, `current_provider`, `current_model` | Returns `200` with the current in-memory game-state snapshot | `get_state` |
| `GET /api/agent/context` | none | `loaded`, `rom_name`, `frame`, `game_mode`, `position`, `party`, `inventory`, `battle`, `health_summary`, `recommendations`, `active_session_id`, `active_routine`, `available_tools`, `memory_summary`, `guardrails`, `next_recommended_action`, `timestamp` | Returns `200` with a safe empty snapshot | `get_agent_context` |
| `POST /api/agent/act` | none | request: `action`, `frames`; response: `success`, `action`, `frames`, `observation`, `changes`, `timestamp` | Returns `400` with `error: "No ROM loaded"` | `act_and_observe` |
| `GET /api/agent/toolbelt` | none | `active_session_id`, `active_routine`, `available_tools`, `tool_groups`, `memory_summary`, `guardrails`, `next_recommended_action`, `auto_learning_signals`, `timestamp` | Returns `200` with safe empty/default metadata | `get_agent_toolbelt` |
| `GET /api/agent/routines` | none | `active_session_id`, `active_routine`, `routines`, `suggested_routines`, `skill_drafts`, `timestamp` | Returns `200` even if no session is active | `get_agent_routines` |
| `GET /api/agent/guardrails` | none | `active_session_id`, `active_routine`, `guardrails`, `recent_failure_reflections`, `timestamp` | Returns `200` with safe empty/default metadata | `get_agent_guardrails` |
| `POST /api/agent/routines` | none | request: `name`, `steps`, optional `description`, `tags`, `session_id`; response: `routine`, `active_routine`, `timestamp` | Returns `400` if no active or supplied session exists | none |
| `POST /api/save_state` | `POST /save_state` | request body accepted; current route stores one slot per active emulator in memory | Returns `400` with `error: "No ROM loaded"` | `save_state`, `quick_save` |
| `POST /api/load_state` | `POST /load_state` | request body accepted; current route restores the active emulator slot from memory | Returns `400` with `error: "No ROM loaded"` or `error: "No saved state available"` | `load_state`, `quick_load` |
| `GET /api/screen` | none | `image`, `shape`, `timestamp`, `pyboy_frame`, `performance`, optional `optimization` | Returns `400` with `error: "No ROM loaded"` | `get_screen`, `screenshot` |
| `GET /api/stream` | none | SSE prelude: `status`, `fps`; frame event: `image`, `timestamp`, `frame`, `fps`; error event: `error`, `recoverable`, `consecutive_errors` | Returns `200` and emits a single SSE error event when no ROM is loaded | `get_screen`, `screenshot` |
| `POST /api/game/button` | `POST /api/game/action`, `POST /api/action` | request: `button` or `action`, optional `frames`; success: `message`, `action`, `frames`, `history_length` | Returns `400` with `error: "No ROM loaded"` | `press_a`, `press_b`, `press_up`, `press_down`, `press_left`, `press_right`, `press_start`, `press_select`, `press_button`, `press_button_combo`, `hold_button` |

---

## Overview

These endpoints follow OpenClaw conventions for:
- Agent state tracking (mode, goal, actions, errors)
- Component health monitoring (runtime, emulator, stream)
- Status summaries for dashboards and MCP tools

All endpoints return JSON with stable, agent-friendly shapes.

---

## Failure Learning And Guardrails

The backend capability adapter now exposes structured failure-learning metadata
alongside success-oriented memory summaries.

- Persistent memory type: `failure_reflection`
- Record fields: `trigger`, `error`, `consequence`, `defense`, `severity`, `source`, `timestamp`
- Read-only capability surfaces:
  - `GET /api/agent/context`
  - `GET /api/agent/state`
  - `GET /api/agent/toolbelt`
  - `GET /api/agent/guardrails`
- Matching MCP tool:
  - `get_agent_guardrails`

Auto-learning remains conservative:

- A failed `POST /api/agent/act` can append one `failure_reflection` record for the active session.
- The reflection is metadata-only and does not trigger any extra emulator actions.
- Guardrails are advisory planner context intended to reduce repeated mistakes, not enforce control flow.

---

## Health Endpoints

### `GET /health`

Basic health check for monitoring and load balancers.

**Response Shape:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "service": "ai-game-server",
  "version": "3.0.0",
  "python_version": "3.11.0",
  "platform": "macOS-14.0-arm64",
  "uptime_seconds": 3600.5,
  "timestamp": "2026-03-19T20:00:00.000Z",
  "checks": {
    "flask": "ok" | "error",
    "pyboy": "ok" | "not_available" | "error",
    "mcp": "ok" | "not_available" | "error"
  }
}
```

---

### `GET /api/health`

Comprehensive health check for all components.

**Response Shape:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "components": {
    "runtime": {
      "status": "healthy",
      "uptime_seconds": 3600.5,
      "memory_mb": 256.3,
      "checks": {
        "flask": "ok",
        "pyboy": "ok",
        "mcp": "ok",
        "memory": "ok"
      }
    },
    "emulator": {
      "status": "healthy" | "degraded" | "unhealthy" | "not_loaded",
      "rom_loaded": true,
      "rom_name": "Pokemon Red",
      "active_emulator": "pyboy",
      "frame_count": 12345
    },
    "stream": {
      "status": "healthy" | "unhealthy",
      "websocket_running": true,
      "active_clients": 2
    },
    "agent": {
      "status": "healthy" | "degraded",
      "enabled": false,
      "mode": "manual",
      "recent_errors": 0
    }
  },
  "summary": {
    "healthy_count": 4,
    "degraded_count": 0,
    "unhealthy_count": 0,
    "unknown_count": 0
  },
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/health/runtime`

Runtime component health.

**Response Shape:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "uptime_seconds": 3600.5,
  "uptime_human": "1h 0m 0s",
  "checks": {
    "flask": "ok",
    "pyboy": "ok",
    "mcp": "ok",
    "memory": "ok"
  },
  "memory_mb": 256.3,
  "version": "3.0.0",
  "python_version": "3.11.0",
  "platform": "macOS-14.0-arm64",
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/health/emulator`

Emulator component health.

**Response Shape:**
```json
{
  "status": "healthy" | "degraded" | "unhealthy" | "not_loaded",
  "rom_loaded": true,
  "rom_name": "Pokemon Red",
  "active_emulator": "pyboy",
  "frame_count": 12345,
  "fps": 59.8,
  "last_check": "2026-03-19T20:00:00.000Z",
  "error": null,
  "performance": {
    "avg_frame_time_ms": 16.7,
    "avg_encoding_time_ms": 5.2,
    "adaptive_fps_target": 60
  },
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/health/stream`

Stream component health.

**Response Shape:**
```json
{
  "status": "healthy" | "unhealthy",
  "websocket_running": true,
  "websocket_port": 5003,
  "websocket_url": "ws://localhost:5003/api/ws/stream",
  "active_clients": 2,
  "sse_endpoint": "/api/stream",
  "last_check": "2026-03-19T20:00:00.000Z",
  "error": null,
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

## Agent State Endpoints

### `GET /api/agent/state`

Comprehensive agent state (OpenClaw-style).

**Response Shape:**
```json
{
  "mode": "manual" | "autonomous" | "ai_assisted",
  "enabled": false,
  "current_goal": "Defeat Brock",
  "current_task": "Navigate to Pewter City",
  "last_decision": {
    "timestamp": "2026-03-19T20:00:00.000Z",
    "decision": {"action": "UP", "goal": "Defeat Brock"},
    "provider": "openclaw"
  },
  "last_action": "UP",
  "last_action_time": "2026-03-19T20:00:00.000Z",
  "recent_errors": [
    {
      "timestamp": "2026-03-19T19:55:00.000Z",
      "type": "action_error",
      "message": "Failed to execute action: A",
      "context": {"action": "A", "frames": 1}
    }
  ],
  "recent_actions": [
    {
      "timestamp": "2026-03-19T20:00:00.000Z",
      "action": "UP",
      "frames": 1,
      "result": "success",
      "source": "manual"
    }
  ],
  "stats": {
    "total_actions": 1234,
    "total_decisions": 56,
    "total_errors": 3
  },
  "active_session_id": "019e1f3e-4f61-7d7b-bdd0-78ceaf0b86de",
  "active_routine": "heal_party",
  "available_tools": [
    {
      "name": "get_agent_context",
      "access": "read-only",
      "category": "context",
      "backend_route": "/api/agent/context",
      "mcp_tool": "get_agent_context",
      "description": "Get a full structured gameplay snapshot for planning."
    }
  ],
  "memory_summary": {
    "total_records": 4,
    "by_type": {"note": 2, "control_pattern": 2},
    "latest_by_type": {"note": {"type": "note"}, "control_pattern": {"type": "control_pattern"}},
    "recent_notes": [{"type": "note", "text": "Pokemon Center is north of the mart."}],
    "learned_control_patterns": [
      {"sequence": ["UP", "A"], "outcome": "Enter a doorway from the overworld"}
    ]
  },
  "next_recommended_action": {
    "action": "HEAL",
    "reason": "Heal party at Pokemon Center",
    "source": "context.health_summary"
  },
  "started_at": "2026-03-19T19:00:00.000Z",
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/agent/status`

Agent status summary (OpenClaw-style).

**Response Shape:**
```json
{
  "mode": "manual" | "autonomous" | "ai_assisted",
  "enabled": false,
  "goal": "Defeat Brock",
  "last_action": "UP",
  "last_error": null,
  "action_count": 1234,
  "error_count": 3,
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/agent/toolbelt`

Hermes-inspired adapter metadata for planners and operator UIs. This endpoint is
read-only and derives its state from the existing agent state, active session,
and structured memory records.

**Response Shape:**
```json
{
  "active_session_id": "019e1f3e-4f61-7d7b-bdd0-78ceaf0b86de",
  "active_routine": "heal_party",
  "available_tools": [
    {
      "name": "get_agent_context",
      "access": "read-only",
      "category": "context",
      "backend_route": "/api/agent/context",
      "mcp_tool": "get_agent_context"
    },
    {
      "name": "act_and_observe",
      "access": "mutating",
      "category": "actions",
      "backend_route": "/api/agent/act",
      "mcp_tool": "act_and_observe"
    }
  ],
  "tool_groups": {
    "context": ["get_agent_context", "get_game_mode"],
    "capabilities": ["get_agent_toolbelt", "get_agent_routines"],
    "actions": ["act_and_observe"]
  },
  "memory_summary": {
    "total_records": 4,
    "by_type": {"note": 2, "control_pattern": 2},
    "recent_notes": [{"type": "note", "text": "Pokemon Center is north of the mart."}],
    "learned_control_patterns": [
      {"sequence": ["UP", "A"], "outcome": "Enter a doorway from the overworld"}
    ]
  },
  "next_recommended_action": {
    "action": "HEAL",
    "reason": "Heal party at Pokemon Center",
    "source": "context.health_summary"
  },
  "planner_hint": {
    "action": "HEAL",
    "reason": "Heal party at Pokemon Center",
    "source": "context.health_summary"
  },
  "auto_learning_signals": {
    "control_patterns_observed": 2,
    "suggested_routine_count": 1,
    "skill_draft_count": 2
  },
  "timestamp": "2026-06-23T20:00:00.000Z"
}
```

---

### `GET /api/agent/routines`

Lists operator-authored routines plus generated playbook suggestions derived
from learned control patterns. The route is read-only and does not advance the
emulator.

**Response Shape:**
```json
{
  "active_session_id": "019e1f3e-4f61-7d7b-bdd0-78ceaf0b86de",
  "active_routine": "heal_party",
  "routines": [
    {
      "id": "d2a67ce727f0",
      "name": "heal_party",
      "kind": "playbook",
      "origin": "operator",
      "status": "ready",
      "steps": [{"action": "UP", "frames": 1}]
    }
  ],
  "suggested_routines": [
    {
      "id": "learned-f1396fd95444",
      "name": "enter_a_doorway_from_the_overworld",
      "kind": "generated_playbook",
      "origin": "memory",
      "status": "suggested",
      "steps": [{"action": "UP", "frames": 1}, {"action": "A", "frames": 1}],
      "summary": "Enter a doorway from the overworld"
    }
  ],
  "skill_drafts": [
    {
      "id": "skill-f1396fd95444",
      "name": "Enter a doorway from the overworld",
      "source": "memory.control_pattern",
      "status": "draft"
    }
  ],
  "timestamp": "2026-06-23T20:00:00.000Z"
}
```

---

### `POST /api/agent/routines`

Creates or updates a reusable routine in the active or supplied session. This
persists session metadata only; it does not press buttons, tick frames, or
write emulator save-state.

**Request Shape:**
```json
{
  "session_id": "019e1f3e-4f61-7d7b-bdd0-78ceaf0b86de",
  "name": "pewter_entry",
  "description": "Line up on the city gate and walk north.",
  "steps": [{"action": "UP", "frames": 4}],
  "tags": ["navigation"],
  "activate": true
}
```

**Response Shape:**
```json
{
  "ok": true,
  "session_id": "019e1f3e-4f61-7d7b-bdd0-78ceaf0b86de",
  "active_routine": "pewter_entry",
  "routine": {
    "id": "2606eb59de50",
    "name": "pewter_entry",
    "kind": "playbook",
    "origin": "operator",
    "status": "ready",
    "steps": [{"action": "UP", "frames": 4}],
    "skill_draft": {
      "id": "skill-2606eb59de50",
      "name": "pewter_entry",
      "source": "routine.upsert",
      "status": "draft"
    }
  },
  "timestamp": "2026-06-23T20:00:00.000Z"
}
```

---

### `GET /api/agent/goal`

Get current agent goal.

**Response Shape:**
```json
{
  "goal": "Defeat Brock",
  "task": "Navigate to Pewter City",
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `POST /api/agent/goal`

Set agent goal.

**Request Body:**
```json
{
  "goal": "Defeat Brock",
  "task": "Navigate to Pewter City"
}
```

**Response Shape:**
```json
{
  "ok": true,
  "goal": "Defeat Brock",
  "task": "Navigate to Pewter City",
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `POST /api/agent/mode`

Set agent mode.

**Request Body:**
```json
{
  "mode": "autonomous"
}
```

**Response Shape:**
```json
{
  "ok": true,
  "mode": "autonomous"
}
```

---

### `GET /api/agent/errors`

Get recent agent errors.

**Query Parameters:**
- `limit`: Number of errors to return (default: 10, max: 50)

**Response Shape:**
```json
{
  "errors": [
    {
      "timestamp": "2026-03-19T19:55:00.000Z",
      "type": "action_error",
      "message": "Failed to execute action: A",
      "context": {"action": "A", "frames": 1}
    }
  ],
  "count": 1,
  "total_errors": 3,
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

### `GET /api/agent/actions`

Get recent agent actions.

**Query Parameters:**
- `limit`: Number of actions to return (default: 20, max: 100)

**Response Shape:**
```json
{
  "actions": [
    {
      "timestamp": "2026-03-19T20:00:00.000Z",
      "action": "UP",
      "frames": 1,
      "result": "success",
      "source": "manual"
    }
  ],
  "count": 20,
  "total_actions": 1234,
  "timestamp": "2026-03-19T20:00:00.000Z"
}
```

---

## Status Endpoint

### `GET /api/status`

Comprehensive status of the server.

**Response Shape:**
```json
{
  "rom_loaded": true,
  "active_emulator": "pyboy",
  "rom_path": "/path/to/rom.gb",
  "rom_name": "Pokemon Red",
  "fps": 60,
  "speed_multiplier": 1.0,
  "current_goal": "Defeat Brock",
  "current_provider": "openclaw",
  "current_model": "bailian/kimi-k2.5",
  "frame_count": 12345,
  "ai_providers": {
    "openclaw": {"available": true, "status": "available"},
    "lmstudio": {"available": true, "status": "available"}
  }
}
```

---

## Design Principles

### Stable Shapes
- All endpoints return consistent JSON structures
- Empty/null values are explicit, not missing fields
- Arrays default to `[]`, objects to `{}`, counts to `0`

### Agent-Friendly
- Timestamps in ISO 8601 format
- Enums are lowercase strings (e.g., `"healthy"`, `"manual"`)
- Counts are integers
- Errors include context for debugging

### OpenClaw Compatibility
- Health status values: `"healthy"`, `"degraded"`, `"unhealthy"`
- Agent modes: `"manual"`, `"autonomous"`, `"ai_assisted"`
- Action sources: `"manual"`, `"ai"`, `"autonomous"`

---

## Usage Examples

### Check if server is healthy:
```bash
curl http://localhost:5002/health
```

### Get full health status:
```bash
curl http://localhost:5002/api/health
```

### Get agent state:
```bash
curl http://localhost:5002/api/agent/state
```

### Set agent goal:
```bash
curl -X POST http://localhost:5002/api/agent/goal \
  -H "Content-Type: application/json" \
  -d '{"goal": "Defeat Brock", "task": "Navigate to Pewter City"}'
```

### Get recent actions:
```bash
curl "http://localhost:5002/api/agent/actions?limit=10"
```

### Get recent errors:
```bash
curl "http://localhost:5002/api/agent/errors?limit=5"
```

---

## Integration with MCP Tools

These endpoints are designed for easy integration with MCP tools and LM Studio:

```python
# Example: Get health status
health = mcp_call("http_get", {"url": "http://localhost:5002/api/health"})

# Example: Get agent state
state = mcp_call("http_get", {"url": "http://localhost:5002/api/agent/state"})

# Example: Set goal
result = mcp_call("http_post", {
    "url": "http://localhost:5002/api/agent/goal",
    "body": {"goal": "Defeat Brock"}
})
```

---

## Changelog

### v3.0.0 (2026-03-19)
- Added OpenClaw-style agent state endpoints
- Added component health endpoints
- Added action/error tracking
- Normalized response shapes
- Added API-CONTRACT.md documentation
