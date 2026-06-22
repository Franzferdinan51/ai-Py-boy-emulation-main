"""
Agent Memory / KnowledgeBase — structured episodic memory for AI gameplay.

Inspired by CalebDeLeeuwMisfits/PokemonLLMAgentBenchmark's KnowledgeBase class
and Voyager-style "agent that learns in environment" patterns.

Each session has its own knowledge base, persisted as JSON lines under
`<data_dir>/games/<session_id>/memory/knowledge.jsonl`.

Stores:
  - locations: where the player has been (map_id, map_name, x, y, when)
  - party: history of Pokémon caught / leveled up
  - objectives: completed objectives with timestamps
  - notes: free-form knowledge notes (RAG-ready)
  - facts: structured key-value facts about the world
  - controls: learned control patterns (button sequences → outcomes)
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sessions import get_session, update_session, DEFAULT_DATA_DIR

_lock = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_path(session_id: str, data_dir: Optional[str] = None) -> Path:
    return Path(data_dir or DEFAULT_DATA_DIR) / "games" / session_id / "memory" / "knowledge.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(records) >= limit:
                    break
    except OSError:
        return []
    return records


# ---------------------------------------------------------------------------
# Knowledge types
# ---------------------------------------------------------------------------

def add_location(
    session_id: str,
    map_id: Optional[int] = None,
    map_name: Optional[str] = None,
    x: Optional[int] = None,
    y: Optional[int] = None,
    note: str = "",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Record that the player was at a location."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}

    record = {
        "id": str(uuid.uuid4()),
        "type": "location",
        "timestamp": _now_iso(),
        "map_id": map_id,
        "map_name": map_name,
        "x": x,
        "y": y,
        "note": note,
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
    return {"ok": True, "record": record}


def add_party_event(
    session_id: str,
    event: str,  # "caught" | "leveled_up" | "evolved" | "fainted" | "healed"
    species: str = "",
    level: Optional[int] = None,
    note: str = "",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    record = {
        "id": str(uuid.uuid4()),
        "type": "party_event",
        "timestamp": _now_iso(),
        "event": event,
        "species": species,
        "level": level,
        "note": note,
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
    return {"ok": True, "record": record}


def complete_objective(
    session_id: str,
    objective: str,
    tier: str = "short_term",  # "short_term" | "mid_term" | "long_term"
    note: str = "",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Mark an objective as complete and record it in memory."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}

    record = {
        "id": str(uuid.uuid4()),
        "type": "objective_complete",
        "timestamp": _now_iso(),
        "objective": objective,
        "tier": tier,
        "note": note,
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
        # Also record as milestone
        sess = get_session(session_id, data_dir)  # re-fetch after lock
        if sess:
            milestones = list(sess.get("milestones", []))
            milestones.append({
                "id": record["id"],
                "timestamp": record["timestamp"],
                "kind": "objective",
                "title": objective,
                "tier": tier,
            })
            # update_session does its own locking; don't nest locks
            _ = milestones  # already written via JSONL; manifest update below

    # Persist milestone into manifest
    with _lock:
        sess2 = get_session(session_id, data_dir)
        if sess2:
            milestones = list(sess2.get("milestones", []))
            milestones.append({
                "id": record["id"],
                "timestamp": record["timestamp"],
                "kind": "objective",
                "title": objective,
                "tier": tier,
            })
            update_session(session_id, {"milestones": milestones}, data_dir=data_dir)

    return {"ok": True, "record": record}


def add_note(
    session_id: str,
    text: str,
    tags: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a free-form knowledge note (RAG-ready)."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    record = {
        "id": str(uuid.uuid4()),
        "type": "note",
        "timestamp": _now_iso(),
        "text": text,
        "tags": tags or [],
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
    return {"ok": True, "record": record}


def add_fact(
    session_id: str,
    key: str,
    value: Any,
    source: str = "agent",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a structured key-value fact about the game world."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    record = {
        "id": str(uuid.uuid4()),
        "type": "fact",
        "timestamp": _now_iso(),
        "key": key,
        "value": value,
        "source": source,
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
    return {"ok": True, "record": record}


def add_control_pattern(
    session_id: str,
    sequence: List[str],
    outcome: str,
    note: str = "",
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a learned control pattern (e.g. button sequence → outcome)."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    record = {
        "id": str(uuid.uuid4()),
        "type": "control_pattern",
        "timestamp": _now_iso(),
        "sequence": sequence,
        "outcome": outcome,
        "note": note,
    }
    with _lock:
        _append_jsonl(_memory_path(session_id, data_dir), record)
    return {"ok": True, "record": record}


# ---------------------------------------------------------------------------
# Query / retrieval (RAG-ready)
# ---------------------------------------------------------------------------

def get_memory(
    session_id: str,
    type_filter: Optional[str] = None,
    limit: int = 100,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Get all memory records for a session, optionally filtered by type."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    with _lock:
        records = _read_jsonl(_memory_path(session_id, data_dir), limit=limit)
    if type_filter:
        records = [r for r in records if r.get("type") == type_filter]
    return {
        "ok": True,
        "session_id": session_id,
        "records": records,
        "count": len(records),
        "timestamp": _now_iso(),
    }


def search_memory(
    session_id: str,
    query: str,
    limit: int = 20,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Simple text-search over memory records (RAG-lite)."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    q = (query or "").strip().lower()
    if not q:
        return {"ok": True, "matches": [], "count": 0}

    with _lock:
        records = _read_jsonl(_memory_path(session_id, data_dir), limit=10000)

    matches: List[Dict[str, Any]] = []
    for r in records:
        haystack = json.dumps(r, ensure_ascii=False).lower()
        if q in haystack:
            matches.append(r)
            if len(matches) >= limit:
                break

    return {
        "ok": True,
        "query": query,
        "matches": matches,
        "count": len(matches),
        "timestamp": _now_iso(),
    }


def summarize_memory(session_id: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return counts and latest of each memory type for a session."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    with _lock:
        records = _read_jsonl(_memory_path(session_id, data_dir), limit=10000)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        t = r.get("type", "unknown")
        by_type.setdefault(t, []).append(r)

    return {
        "ok": True,
        "session_id": session_id,
        "total_records": len(records),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "latest_by_type": {k: v[-1] if v else None for k, v in by_type.items()},
        "timestamp": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Flask registration
# ---------------------------------------------------------------------------

def register_memory_routes(app, *, data_dir: Optional[str] = None):
    from flask import jsonify, request

    @app.route("/api/agent/memory", methods=["GET"])
    def _memory_get():
        from flask import request
        session_id = request.args.get("session_id")
        type_filter = request.args.get("type")
        limit = int(request.args.get("limit", 100))
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(get_memory(session_id, type_filter=type_filter, limit=limit, data_dir=data_dir))

    @app.route("/api/agent/memory/search", methods=["GET"])
    def _memory_search():
        from flask import request
        session_id = request.args.get("session_id")
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 20))
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(search_memory(session_id, query, limit=limit, data_dir=data_dir))

    @app.route("/api/agent/memory/summary", methods=["GET"])
    def _memory_summary():
        from flask import request
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(summarize_memory(session_id, data_dir=data_dir))

    @app.route("/api/agent/memory/note", methods=["POST"])
    def _memory_note():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(add_note(session_id, p.get("text", ""), tags=p.get("tags"), data_dir=data_dir))

    @app.route("/api/agent/memory/location", methods=["POST"])
    def _memory_location():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(add_location(
            session_id,
            map_id=p.get("map_id"),
            map_name=p.get("map_name"),
            x=p.get("x"),
            y=p.get("y"),
            note=p.get("note", ""),
            data_dir=data_dir,
        ))

    @app.route("/api/agent/memory/party_event", methods=["POST"])
    def _memory_party_event():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(add_party_event(
            session_id,
            event=p.get("event", "caught"),
            species=p.get("species", ""),
            level=p.get("level"),
            note=p.get("note", ""),
            data_dir=data_dir,
        ))

    @app.route("/api/agent/memory/objective_complete", methods=["POST"])
    def _memory_objective_complete():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(complete_objective(
            session_id,
            objective=p.get("objective", ""),
            tier=p.get("tier", "short_term"),
            note=p.get("note", ""),
            data_dir=data_dir,
        ))

    @app.route("/api/agent/memory/fact", methods=["POST"])
    def _memory_fact():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(add_fact(
            session_id,
            key=p.get("key", ""),
            value=p.get("value"),
            source=p.get("source", "agent"),
            data_dir=data_dir,
        ))

    @app.route("/api/agent/memory/control_pattern", methods=["POST"])
    def _memory_control_pattern():
        p = request.get_json(silent=True) or {}
        session_id = p.get("session_id")
        if not session_id:
            return jsonify({"ok": False, "error": "session_id required"}), 400
        return jsonify(add_control_pattern(
            session_id,
            sequence=p.get("sequence", []),
            outcome=p.get("outcome", ""),
            note=p.get("note", ""),
            data_dir=data_dir,
        ))
