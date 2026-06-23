"""
agent_features — additive, optional features for the AI-Py-Boy server.

Modules:
  - sessions: named playthroughs with persistent state
  - memory:   structured agent memory / KnowledgeBase (RAG-ready)
  - events:   reasoning event stream (THINK/DECIDE/ACT/MILESTONE/ALERT)
  - telemetry: stuck-meter, blackouts, position tracking, action metrics
  - collision: RAM-derived walkability grid + labeled overlay

Use `register_all(app, emulator_ref_getter=...)` to add all routes to a Flask
app at once. Each module also exposes its own register_*_routes(app) function
for selective wiring.

These modules intentionally do NOT touch the legacy `game_state` /
`agent_state` globals in `backend.server` — they are additive and parallel.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import sessions, memory, events, telemetry, collision, run_ledger, agent_capabilities

__all__ = [
    "sessions",
    "memory",
    "events",
    "telemetry",
    "collision",
    "run_ledger",
    "agent_capabilities",
    "register_all",
]


def register_all(
    app,
    *,
    emulator_ref_getter: Optional[Callable[[], Any]] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, int]:
    """Register all agent_features routes on the given Flask app.

    Returns a dict of {module_name: routes_registered} for visibility.
    """
    counts: Dict[str, int] = {}

    before = len(app.url_map._rules)

    sessions.register_sessions_routes(app, data_dir=data_dir)
    counts["sessions"] = len(app.url_map._rules) - before
    before = len(app.url_map._rules)

    memory.register_memory_routes(app, data_dir=data_dir)
    counts["memory"] = len(app.url_map._rules) - before
    before = len(app.url_map._rules)

    events.register_events_routes(app)
    counts["events"] = len(app.url_map._rules) - before
    before = len(app.url_map._rules)

    telemetry.register_telemetry_routes(app)
    counts["telemetry"] = len(app.url_map._rules) - before
    before = len(app.url_map._rules)

    collision.register_collision_routes(app, emulator_ref_getter=emulator_ref_getter)
    counts["collision"] = len(app.url_map._rules) - before

    return counts
