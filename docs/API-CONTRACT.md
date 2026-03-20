# API Contract - OpenClaw-Style Agent/Health/Status Endpoints

**Last Updated:** March 19, 2026
**Version:** 3.0.0

This document describes the OpenClaw-compatible agent, health, and status endpoints for the AI Game Boy Server.

---

## Overview

These endpoints follow OpenClaw conventions for:
- Agent state tracking (mode, goal, actions, errors)
- Component health monitoring (runtime, emulator, stream)
- Status summaries for dashboards and MCP tools

All endpoints return JSON with stable, agent-friendly shapes.

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