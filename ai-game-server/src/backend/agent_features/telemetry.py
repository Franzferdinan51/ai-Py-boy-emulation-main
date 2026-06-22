"""
Agent Telemetry — Stuck-meter, blackouts, position tracking, action metrics.

Inspired by NousResearch/pokemon-agent's "Instruments" panel:
  - gym badges counter
  - three-tier objectives
  - stuck-meter
  - blackout / caught / action counters
  - milestone timeline

This module tracks:
  - position_history: last N positions for stuck detection
  - stuck_meter: increments when position doesn't change
  - blackouts: count of "all Pokémon fainted" events
  - actions_per_minute, success_rate
  - last_action / last_decision / last_error
  - session duration
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .sessions import get_session, update_session, DEFAULT_DATA_DIR
from . import events as evt

_lock = threading.RLock()

# Per-session telemetry state (in-memory mirror of manifest telemetry)
_state: Dict[str, Dict[str, Any]] = {}

# Position history per session (in-memory, last N)
_POSITION_WINDOW = 32
_positions: Dict[str, Deque[Tuple[int, int, Optional[int]]]] = {}

# Stuck detection: how many recent positions are identical
_STUCK_THRESHOLD = 12


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_state(session_id: str) -> Dict[str, Any]:
    """Get-or-create in-memory telemetry state for a session."""
    with _lock:
        if session_id not in _state:
            sess = get_session(session_id)
            persisted = (sess or {}).get("telemetry", {}) or {}
            _state[session_id] = {
                "session_id": session_id,
                "stuck_meter": int(persisted.get("stuck_meter", 0)),
                "blackouts": int(persisted.get("blackouts", 0)),
                "last_position": persisted.get("last_position"),
                "started_at": sess.get("created_at") if sess else _now_iso(),
                "last_action_at": None,
                "actions_total": 0,
                "actions_success": 0,
                "errors_total": 0,
                "frames_played": 0,
                "current_goal": (sess or {}).get("current_goal", ""),
                "in_battle": False,
                "lowest_hp_percent": 100,
                "battles_won": 0,
                "battles_lost": 0,
            }
            _positions[session_id] = deque(maxlen=_POSITION_WINDOW)
        return _state[session_id]


def report_position(
    session_id: str,
    x: int,
    y: int,
    map_id: Optional[int] = None,
    *,
    threshold: int = _STUCK_THRESHOLD,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Report the agent's current position; update stuck-meter if stuck."""
    state = _ensure_state(session_id)
    pos = (int(x), int(y), map_id)

    with _lock:
        history = _positions.setdefault(session_id, deque(maxlen=_POSITION_WINDOW))
        history.append(pos)
        state["last_position"] = {"x": x, "y": y, "map_id": map_id, "timestamp": _now_iso()}

        # Stuck detection: how many of the last N are identical
        recent = list(history)[-threshold:]
        if len(recent) >= threshold and len(set(recent)) == 1:
            new_stuck = min(100, state["stuck_meter"] + 5)
            if new_stuck == 100 and state["stuck_meter"] < 100:
                evt.alert(
                    f"Agent is stuck at ({x},{y}) — {threshold}+ identical positions",
                    session_id=session_id,
                    x=x, y=y, map_id=map_id,
                )
            state["stuck_meter"] = new_stuck
        else:
            # Decay slowly if moving
            state["stuck_meter"] = max(0, state["stuck_meter"] - 1)

        # Persist
        update_session(session_id, {
            "telemetry": {
                "stuck_meter": state["stuck_meter"],
                "blackouts": state["blackouts"],
                "last_position": state["last_position"],
            },
        }, data_dir=data_dir)

    return {
        "ok": True,
        "stuck_meter": state["stuck_meter"],
        "is_stuck": state["stuck_meter"] >= 80,
        "position": state["last_position"],
    }


def report_action(
    session_id: str,
    success: bool = True,
    action: str = "",
    error: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Record an action result for telemetry."""
    state = _ensure_state(session_id)
    with _lock:
        state["actions_total"] += 1
        state["last_action_at"] = _now_iso()
        if success:
            state["actions_success"] += 1
        else:
            state["errors_total"] += 1
        if error:
            state["last_error"] = {"message": error, "timestamp": _now_iso(), "action": action}
        # Update manifest stats
        sess = get_session(session_id, data_dir)
        if sess:
            stats = dict(sess.get("stats", {}))
            stats["total_actions"] = state["actions_total"]
            stats["total_errors"] = state["errors_total"]
            update_session(session_id, {"stats": stats}, data_dir=data_dir)
    return {"ok": True, "actions_total": state["actions_total"], "success_rate": _success_rate(state)}


def report_battle_outcome(
    session_id: str,
    won: bool,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    state = _ensure_state(session_id)
    with _lock:
        if won:
            state["battles_won"] += 1
            evt.milestone(f"Battle won", session_id=session_id)
        else:
            state["battles_lost"] += 1
            evt.alert(f"Battle lost", session_id=session_id)
    return {"ok": True, "won": state["battles_won"], "lost": state["battles_lost"]}


def report_blackout(session_id: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
    state = _ensure_state(session_id)
    with _lock:
        state["blackouts"] += 1
        update_session(session_id, {
            "telemetry": {
                "stuck_meter": state["stuck_meter"],
                "blackouts": state["blackouts"],
                "last_position": state["last_position"],
            },
        }, data_dir=data_dir)
    evt.alert(f"Blackout #{state['blackouts']}", session_id=session_id)
    return {"ok": True, "blackouts": state["blackouts"]}


def report_health(
    session_id: str,
    lowest_hp_percent: int,
    in_battle: bool = False,
) -> Dict[str, Any]:
    state = _ensure_state(session_id)
    with _lock:
        state["lowest_hp_percent"] = int(lowest_hp_percent)
        state["in_battle"] = bool(in_battle)
    return {"ok": True, "lowest_hp_percent": state["lowest_hp_percent"], "in_battle": state["in_battle"]}


def report_frames(session_id: str, frames: int) -> Dict[str, Any]:
    state = _ensure_state(session_id)
    with _lock:
        state["frames_played"] = int(frames)
    return {"ok": True, "frames_played": state["frames_played"]}


def get_telemetry(session_id: str) -> Dict[str, Any]:
    state = _ensure_state(session_id)
    with _lock:
        return {
            "ok": True,
            "telemetry": {
                **state,
                "success_rate": _success_rate(state),
                "is_stuck": state["stuck_meter"] >= 80,
                "position_history_size": len(_positions.get(session_id, [])),
                "session_duration_seconds": _duration_seconds(state.get("started_at")),
                "timestamp": _now_iso(),
            },
        }


def reset_telemetry(session_id: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        if session_id in _state:
            _state[session_id]["stuck_meter"] = 0
            _state[session_id]["errors_total"] = 0
        _positions.pop(session_id, None)
        update_session(session_id, {
            "telemetry": {
                "stuck_meter": 0,
                "blackouts": _state.get(session_id, {}).get("blackouts", 0),
                "last_position": None,
            },
        }, data_dir=data_dir)
    return {"ok": True, "reset": True}


def _success_rate(state: Dict[str, Any]) -> float:
    total = state.get("actions_total", 0)
    if not total:
        return 0.0
    return round(100.0 * state.get("actions_success", 0) / total, 1)


def _duration_seconds(started_at: Optional[str]) -> float:
    if not started_at:
        return 0.0
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - started).total_seconds()
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Flask registration
# ---------------------------------------------------------------------------

def register_telemetry_routes(app):
    from flask import jsonify, request

    @app.route("/api/agent/telemetry", methods=["GET"])
    def _telemetry_get():
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(get_telemetry(session_id))

    @app.route("/api/agent/telemetry/position", methods=["POST"])
    def _telemetry_position():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id or "x" not in p or "y" not in p:
            return jsonify({"ok": False, "error": "session_id, x, y required"}), 400
        return jsonify(report_position(session_id, p["x"], p["y"], p.get("map_id")))

    @app.route("/api/agent/telemetry/action", methods=["POST"])
    def _telemetry_action():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(report_action(
            session_id,
            success=p.get("success", True),
            action=p.get("action", ""),
            error=p.get("error"),
        ))

    @app.route("/api/agent/telemetry/battle", methods=["POST"])
    def _telemetry_battle():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(report_battle_outcome(session_id, bool(p.get("won", False))))

    @app.route("/api/agent/telemetry/blackout", methods=["POST"])
    def _telemetry_blackout():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(report_blackout(session_id))

    @app.route("/api/agent/telemetry/health", methods=["POST"])
    def _telemetry_health():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id or "lowest_hp_percent" not in p:
            return jsonify({"ok": False, "error": "session_id, lowest_hp_percent required"}), 400
        return jsonify(report_health(session_id, p["lowest_hp_percent"], p.get("in_battle", False)))

    @app.route("/api/agent/telemetry/reset", methods=["POST"])
    def _telemetry_reset():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(reset_telemetry(session_id))
