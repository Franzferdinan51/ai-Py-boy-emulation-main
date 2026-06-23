# Game Boy Agent API Contract

This document records the canonical HTTP and MCP surface for the agent platform.
The canonical paths below are the ones new callers should target. Legacy aliases
remain available for compatibility, but they are not the preferred contract.

## Canonical HTTP Matrix

| Canonical HTTP route | Supported aliases | High-level payload fields | No-ROM behavior | Related MCP tools |
| --- | --- | --- | --- | --- |
| `GET /api/game/state` | none | `rom_loaded`, `active_emulator`, `rom_path`, `rom_name`, `frame_count`, `ai_running`, `current_goal`, `fps`, `speed_multiplier`, `current_provider`, `current_model` | Returns `200` with the current in-memory game-state snapshot | `get_state` |
| `GET /api/agent/context` | none | `loaded`, `rom_name`, `frame`, `game_mode`, `position`, `party`, `inventory`, `battle`, `health_summary`, `recommendations`, `timestamp` | Returns `200` with an empty safe snapshot | `get_agent_context` |
| `POST /api/agent/act` | none | request: `action`, `frames`; response: `success`, `action`, `frames`, `observation`, `changes`, `timestamp` | Returns `400` with `error: "No ROM loaded"` | `act_and_observe` |
| `POST /api/save_state` | `POST /save_state` | request body is accepted; current route stores one slot per active emulator in memory | Returns `400` with `error: "No ROM loaded"` | `save_state`, `quick_save` |
| `POST /api/load_state` | `POST /load_state` | request body is accepted; current route restores the active emulator slot from memory | Returns `400` with `error: "No ROM loaded"` or `error: "No saved state available"` | `load_state`, `quick_load` |
| `GET /api/screen` | none | `image`, `shape`, `timestamp`, `pyboy_frame`, `performance`, optional `optimization` | Returns `400` with `error: "No ROM loaded"` | `get_screen`, `screenshot` |
| `GET /api/stream` | none | SSE prelude: `status`, `fps`; frame event: `image`, `timestamp`, `frame`, `fps`; error event: `error`, `recoverable`, `consecutive_errors` | Returns `200` and emits a single SSE error event when no ROM is loaded | `get_screen`, `screenshot` |
| `POST /api/game/button` | `POST /api/game/action`, `POST /api/action` | request: `button` or `action`, optional `frames`; success: `message`, `action`, `frames`, `history_length` | Returns `400` with `error: "No ROM loaded"` | `press_a`, `press_b`, `press_up`, `press_down`, `press_left`, `press_right`, `press_start`, `press_select`, `press_button`, `press_button_combo`, `hold_button` |

## Named Save Semantics

- The canonical save/load endpoints are the `/api/*` routes.
- The backend currently keeps save-state bytes in memory, keyed by the active
  emulator id. That means the effective slot is the active emulator, not an
  arbitrary user-supplied name.
- MCP tools may send a logical `name` such as `quick_save`, but the current HTTP
  route implementation does not branch on the request name.
- The bare `/save_state` and `/load_state` routes are legacy compatibility
  shims. They return placeholder success bodies when a ROM is loaded, but they
  are not the canonical persistence path.

## Stream Contract

`GET /api/stream` is server-sent events over `text/event-stream`.

- Startup event: `{"status":"stream_started","fps":30}`
- Frame event: `{"image":"...","timestamp":..., "frame":..., "fps":30}`
- Error event: `{"error":"...", "recoverable":true, "consecutive_errors":N}`

Without a ROM, the endpoint returns `200` and emits a single SSE error event
with `error: "No ROM loaded"`.

## MCP Mapping

The generic MCP wrapper in `ai-game-server/generic_mcp_server.py` routes the
core tools to the HTTP contract above:

- `get_state` -> `GET /api/game/state`
- `get_agent_context` -> `GET /api/agent/context`
- `act_and_observe` -> `POST /api/agent/act`
- `save_state` -> `POST /api/save_state`
- `load_state` -> `POST /api/load_state`
- `quick_save` -> `POST /api/save_state` with `name="quick_save"`
- `quick_load` -> `POST /api/load_state` with `name="quick_save"`
- `get_screen` and `screenshot` -> `GET /api/screen`
- `press_a`, `press_b`, `press_up`, `press_down`, `press_left`, `press_right`,
  `press_start`, `press_select` -> `POST /api/game/button`

The MCP wrapper also exposes `press_button`, `press_button_combo`, and
`hold_button`, which fan into the same canonical button route.
