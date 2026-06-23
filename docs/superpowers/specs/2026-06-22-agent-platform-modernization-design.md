# Agent Platform Modernization Design

## Goal

Modernize the Game Boy operator platform without changing emulator authority.
PyBoy remains the only component that advances frames or restores emulator
state. The web app, MCP server, OpenClaw, and a future Hermes adapter use one
stable backend contract.

## Decisions

- OpenClaw stays the first-class remote agent gateway.
- Hermes-inspired capabilities are portable in-repo primitives: durable memory,
  routines, structured goals/tasks, and action traces. Hermes is not a required
  runtime dependency in this phase.
- Every agent action emits an observation/result record. Read-only tools are
  separate from emulator-mutating tools.
- Compatibility aliases remain available until frontend and MCP callers use
  canonical endpoint names.

## Architecture

The backend exposes a versioned run contract around the existing agent routes:
`GameObservationV1`, `GameActionV1`, and append-only `GameRunEventV1` records.
Goals contain ordered tasks and may reference reusable declarative routines.

Streaming reads the latest rendered frame and never ticks PyBoy. Save/load uses
one named-state implementation for web and MCP callers, including legacy aliases.

## Operator UI

The frontend defaults to backend `:5002`, uses a shared API client, consumes the
actual server SSE frame shape, and adds a compact run panel for goals, tasks,
approval policy, latest observation, and recent action traces.

## Verification

- Flask tests cover canonical response shapes, named save/load, and event traces.
- MCP tests prove core tools reach real backend endpoints.
- Frontend tests cover SSE decoding and API mapping.
- A live-ROM smoke path verifies load, multiple stream frames, save/load restore,
  and matching MCP calls.

## Non-goals

- Installing or vendoring Hermes Agent or OpenClaw.
- Adding a second emulation loop or direct MCP emulator access.
- Adding a database migration in the first modernization phase.
