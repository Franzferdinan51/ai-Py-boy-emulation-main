"""
Agent Reasoning Events — THINK / DECIDE / ACT / MILESTONE / ALERT stream.

Inspired by NousResearch/pokemon-agent's "Field Log" reasoning stream.
Each event has a kind, a message, optional metadata, and a timestamp.
Events can be broadcast over WebSocket / SSE (later) and queried via REST.

This is intentionally global (not per-session) so the live field log can
follow the active session and survive session switches.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

from .sessions import get_session, get_current_session, DEFAULT_DATA_DIR

VALID_KINDS: Set[str] = {"THINK", "DECIDE", "ACT", "MILESTONE", "ALERT", "OBSERVE", "REFLECT"}

# Ring buffer of recent events (in-memory)
_MAX_BUFFER = int(os.environ.get("PYBOY_EVENT_BUFFER", "1000"))
_events: Deque[Dict[str, Any]] = deque(maxlen=_MAX_BUFFER)
_events_lock = threading.RLock()

# Subscribers (for future SSE/WS broadcast): list of threading.Event
_subscribers: List["threading.Event"] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_event(
    kind: str,
    message: str,
    *,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    persist: bool = True,
) -> Dict[str, Any]:
    """Emit a reasoning event to the field log."""
    kind = (kind or "").upper()
    if kind not in VALID_KINDS:
        kind = "OBSERVE"  # graceful default

    record = {
        "id": str(uuid.uuid4()),
        "kind": kind,
        "message": message,
        "timestamp": _now_iso(),
        "session_id": session_id,
        "metadata": metadata or {},
    }

    with _events_lock:
        _events.append(record)
        # Wake subscribers
        for ev in list(_subscribers):
            try:
                ev.set()
            except Exception:  # noqa: BLE001
                pass

    # Optionally persist per-session
    if persist and session_id:
        try:
            sess = get_session(session_id)
            if sess:
                target = Path(sess.get("data_dir", str(Path(DEFAULT_DATA_DIR) / "games" / session_id))) / "events" / "field.log.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    return {"ok": True, "event": record}


def get_events(
    kind: Optional[str] = None,
    limit: int = 100,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get recent events, optionally filtered by kind and/or session."""
    with _events_lock:
        snapshot = list(_events)

    if kind:
        snapshot = [e for e in snapshot if e.get("kind") == kind.upper()]
    if session_id:
        snapshot = [e for e in snapshot if e.get("session_id") == session_id]

    # Most recent first
    snapshot.reverse()
    snapshot = snapshot[:limit]

    return {
        "ok": True,
        "events": snapshot,
        "count": len(snapshot),
        "buffer_size": len(_events),
        "timestamp": _now_iso(),
    }


def clear_events() -> Dict[str, Any]:
    with _events_lock:
        _events.clear()
    return {"ok": True, "cleared": True, "timestamp": _now_iso()}


def get_event_stats() -> Dict[str, Any]:
    with _events_lock:
        snapshot = list(_events)
    by_kind: Dict[str, int] = {}
    by_session: Dict[str, int] = {}
    for e in snapshot:
        k = e.get("kind", "OBSERVE")
        by_kind[k] = by_kind.get(k, 0) + 1
        sid = e.get("session_id") or "_none"
        by_session[sid] = by_session.get(sid, 0) + 1
    return {
        "ok": True,
        "total": len(snapshot),
        "by_kind": by_kind,
        "by_session": by_session,
        "buffer_capacity": _MAX_BUFFER,
        "timestamp": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Convenience helpers used by other modules
# ---------------------------------------------------------------------------

def think(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("THINK", message, session_id=session_id, metadata=meta)


def decide(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("DECIDE", message, session_id=session_id, metadata=meta)


def act(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("ACT", message, session_id=session_id, metadata=meta)


def milestone(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("MILESTONE", message, session_id=session_id, metadata=meta)


def alert(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("ALERT", message, session_id=session_id, metadata=meta)


def observe(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("OBSERVE", message, session_id=session_id, metadata=meta)


def reflect(message: str, session_id: Optional[str] = None, **meta) -> Dict[str, Any]:
    return emit_event("REFLECT", message, session_id=session_id, metadata=meta)


# ---------------------------------------------------------------------------
# Flask registration
# ---------------------------------------------------------------------------

def register_events_routes(app):
    from flask import jsonify, request, Response

    @app.route("/api/agent/events", methods=["GET"])
    def _events_list():
        kind = request.args.get("kind")
        limit = int(request.args.get("limit", 100))
        session_id = request.args.get("session_id")
        return jsonify(get_events(kind=kind, limit=limit, session_id=session_id))

    @app.route("/api/agent/events", methods=["POST"])
    def _events_emit():
        p = request.get_json(silent=True) or {}
        kind = p.get("kind", "OBSERVE")
        message = p.get("message", "")
        session_id = p.get("session_id")
        if not message:
            return jsonify({"ok": False, "error": "message required"}), 400
        return jsonify(emit_event(
            kind,
            message,
            session_id=session_id,
            metadata=p.get("metadata"),
            persist=p.get("persist", True),
        ))

    @app.route("/api/agent/events/clear", methods=["POST"])
    def _events_clear():
        return jsonify(clear_events())

    @app.route("/api/agent/events/stats", methods=["GET"])
    def _events_stats():
        return jsonify(get_event_stats())

    @app.route("/api/agent/events/stream", methods=["GET"])
    def _events_stream():
        """SSE stream of new events.

        Query params:
          session_id (optional): filter to specific session
          kinds (optional): comma-separated kind whitelist
        """
        session_id = request.args.get("session_id")
        kinds_param = request.args.get("kinds", "")
        allowed_kinds = {k.strip().upper() for k in kinds_param.split(",") if k.strip()}

        def gen():
            last_seen_ts = ""
            while True:
                # Pull events newer than last_seen_ts
                with _events_lock:
                    snapshot = [e for e in _events if e.get("timestamp", "") > last_seen_ts]
                for ev in snapshot:
                    if session_id and ev.get("session_id") != session_id:
                        continue
                    if allowed_kinds and ev.get("kind") not in allowed_kinds:
                        continue
                    last_seen_ts = ev.get("timestamp", last_seen_ts)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                time.sleep(0.5)

        return Response(gen(), mimetype="text/event-stream")
