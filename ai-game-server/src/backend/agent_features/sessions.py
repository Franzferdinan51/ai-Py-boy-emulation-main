"""
Game Sessions — Named playthroughs with persistent state.

Inspired by NousResearch/pokemon-agent's session model: each session binds
together the emulator save-state, the agent brain (LLM session id), objectives,
milestones, and stats, all persisted to disk under `<data_dir>/games/<id>/`.

This module is intentionally additive: it can be wired into the existing
Flask app via register_sessions_routes(app) without breaking the legacy
game_state / agent_state globals.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = os.environ.get(
    "PYBOY_DATA_DIR",
    str(Path.home() / ".ai-py-boy" / "data"),
)

_sessions_lock = threading.RLock()
# In-memory index: {session_id: session_dict}
_sessions_index: Dict[str, Dict[str, Any]] = {}
# The currently active session id (None if no session)
_active_session_id: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(name: str) -> str:
    """Sanitize session names for filesystem use."""
    name = (name or "").strip()
    if not name:
        return ""
    # Allow letters, digits, dash, underscore, space → keep alnum + dash + underscore
    return re.sub(r"[^A-Za-z0-9 _-]", "", name)[:80].strip() or "session"


def _session_dir(session_id: str, data_dir: Optional[str] = None) -> Path:
    base = Path(data_dir or DEFAULT_DATA_DIR)
    return base / "games" / session_id


def _manifest_path(session_id: str, data_dir: Optional[str] = None) -> Path:
    return _session_dir(session_id, data_dir) / "manifest.json"


def _save_state_path(session_id: str, data_dir: Optional[str] = None) -> Path:
    return _session_dir(session_id, data_dir) / "save.state"


def _init_session_dirs(session_id: str, data_dir: Optional[str] = None) -> Path:
    sdir = _session_dir(session_id, data_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "memory").mkdir(exist_ok=True)
    (sdir / "events").mkdir(exist_ok=True)
    (sdir / "objectives").mkdir(exist_ok=True)
    return sdir


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def create_session(
    name: str = "",
    rom_path: Optional[str] = None,
    brain_session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new named playthrough session."""
    with _sessions_lock:
        session_id = str(uuid.uuid4())
        clean_name = _safe_name(name) or f"Run-{int(time.time())}"
        created_at = _now_iso()

        session = {
            "id": session_id,
            "name": clean_name,
            "created_at": created_at,
            "updated_at": created_at,
            "rom_path": rom_path,
            "brain_session_id": brain_session_id,
            "current_goal": "",
            "objectives": {
                "long_term": [],
                "mid_term": [],
                "short_term": [],
            },
            "milestones": [],
            "stats": {
                "total_actions": 0,
                "total_decisions": 0,
                "total_errors": 0,
                "total_events": 0,
                "frames_played": 0,
            },
            "telemetry": {
                "stuck_meter": 0,
                "blackouts": 0,
                "last_position": None,
                "position_history": [],
            },
            "tags": [],
            "data_dir": str(_session_dir(session_id, data_dir)),
        }

        # Persist manifest
        _init_session_dirs(session_id, data_dir)
        manifest_path = _manifest_path(session_id, data_dir)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        _sessions_index[session_id] = session

        return {
            "ok": True,
            "session": session,
            "manifest_path": str(manifest_path),
            "timestamp": created_at,
        }


def list_sessions(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """List all persisted sessions (loads from disk if needed)."""
    with _sessions_lock:
        base = Path(data_dir or DEFAULT_DATA_DIR) / "games"
        if not base.exists():
            return {"ok": True, "sessions": [], "count": 0, "active_session_id": _active_session_id}

        # Hydrate index from disk if empty
        if not _sessions_index:
            _hydrate_index(data_dir)

        items = sorted(
            _sessions_index.values(),
            key=lambda s: s.get("updated_at") or s.get("created_at") or "",
            reverse=True,
        )

        # Compact list view
        compact = [
            {
                "id": s["id"],
                "name": s["name"],
                "rom_path": s.get("rom_path"),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at"),
                "current_goal": s.get("current_goal", ""),
                "milestones_count": len(s.get("milestones", [])),
                "actions": s.get("stats", {}).get("total_actions", 0),
                "tags": s.get("tags", []),
            }
            for s in items
        ]

        return {
            "ok": True,
            "sessions": compact,
            "count": len(compact),
            "active_session_id": _active_session_id,
            "timestamp": _now_iso(),
        }


def get_session(session_id: str, data_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _sessions_lock:
        if session_id not in _sessions_index:
            _hydrate_index(data_dir)
        return _sessions_index.get(session_id)


def get_current_session(data_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if _active_session_id is None:
        return None
    return get_session(_active_session_id, data_dir)


def set_active_session(session_id: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
    global _active_session_id
    with _sessions_lock:
        sess = get_session(session_id, data_dir)
        if not sess:
            return {"ok": False, "error": f"Session {session_id!r} not found"}
        _active_session_id = session_id
        return {
            "ok": True,
            "active_session": {
                "id": sess["id"],
                "name": sess["name"],
                "rom_path": sess.get("rom_path"),
                "brain_session_id": sess.get("brain_session_id"),
                "updated_at": sess.get("updated_at"),
            },
            "timestamp": _now_iso(),
        }


def update_session(session_id: str, patch: Dict[str, Any], data_dir: Optional[str] = None,
                   persist: bool = True) -> Optional[Dict[str, Any]]:
    """Update session fields and (optionally) persist manifest."""
    with _sessions_lock:
        sess = get_session(session_id, data_dir)
        if not sess:
            return None
        for key, value in (patch or {}).items():
            if key in {"id", "created_at"}:
                continue  # immutable
            sess[key] = value
        sess["updated_at"] = _now_iso()
        if persist:
            _persist_manifest(sess, data_dir)
        return sess


def delete_session(session_id: str, data_dir: Optional[str] = None) -> Dict[str, Any]:
    global _active_session_id
    with _sessions_lock:
        sess = get_session(session_id, data_dir)
        if not sess:
            return {"ok": False, "error": f"Session {session_id!r} not found"}
        _sessions_index.pop(session_id, None)
        if _active_session_id == session_id:
            _active_session_id = None
        sdir = _session_dir(session_id, data_dir)
        if sdir.exists():
            try:
                import shutil
                shutil.rmtree(sdir)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"Deleted in-memory but failed to remove dir: {e}"}
        return {"ok": True, "deleted": session_id, "timestamp": _now_iso()}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_manifest(session: Dict[str, Any], data_dir: Optional[str] = None) -> None:
    manifest_path = _manifest_path(session["id"], data_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2)


def _hydrate_index(data_dir: Optional[str] = None) -> None:
    base = Path(data_dir or DEFAULT_DATA_DIR) / "games"
    if not base.exists():
        return
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        manifest = entry / "manifest.json"
        if not manifest.exists():
            continue
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                session = json.load(f)
            _sessions_index[session["id"]] = session
        except (json.JSONDecodeError, OSError, KeyError):
            continue


def save_emulator_state_for_session(session_id: str, state_bytes: bytes,
                                    data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Persist emulator save-state bytes for a session."""
    sess = get_session(session_id, data_dir)
    if not sess:
        return {"ok": False, "error": f"Session {session_id!r} not found"}
    target = _save_state_path(session_id, data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(state_bytes)
    update_session(session_id, {"last_saved_at": _now_iso()}, data_dir=data_dir)
    return {"ok": True, "session_id": session_id, "bytes": len(state_bytes), "path": str(target)}


def load_emulator_state_for_session(session_id: str, data_dir: Optional[str] = None) -> Optional[bytes]:
    sess = get_session(session_id, data_dir)
    if not sess:
        return None
    target = _save_state_path(session_id, data_dir)
    if not target.exists():
        return None
    with open(target, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Flask blueprint registration (additive, optional)
# ---------------------------------------------------------------------------

def register_sessions_routes(app, *, data_dir: Optional[str] = None):
    """Register Game Sessions REST routes on the given Flask app.

    Endpoints:
      POST /api/games/new              → create_session
      GET  /api/games                  → list_sessions
      GET  /api/games/current          → get_current_session
      GET  /api/games/<sid>            → get_session
      POST /api/games/<sid>/activate   → set_active_session
      POST /api/games/<sid>/update     → update_session
      DELETE /api/games/<sid>          → delete_session
      GET  /api/games/<sid>/save_state → load saved state bytes (metadata)
      POST /api/games/<sid>/save_state → save state bytes from request body
    """
    from flask import jsonify, request, send_file

    _ = data_dir  # reserved for future per-route override

    @app.route("/api/games/new", methods=["POST"])
    def _games_new():
        payload = request.get_json(silent=True) or {}
        return jsonify(create_session(
            name=payload.get("name", ""),
            rom_path=payload.get("rom_path"),
            brain_session_id=payload.get("brain_session_id"),
        ))

    @app.route("/api/games", methods=["GET"])
    def _games_list():
        return jsonify(list_sessions())

    @app.route("/api/games/current", methods=["GET"])
    def _games_current():
        sess = get_current_session()
        if not sess:
            return jsonify({"ok": False, "active": False, "session": None}), 200
        return jsonify({"ok": True, "active": True, "session": sess})

    @app.route("/api/games/<session_id>", methods=["GET"])
    def _games_get(session_id):
        sess = get_session(session_id)
        if not sess:
            return jsonify({"ok": False, "error": "not found"}), 404
        return jsonify({"ok": True, "session": sess})

    @app.route("/api/games/<session_id>/activate", methods=["POST"])
    def _games_activate(session_id):
        return jsonify(set_active_session(session_id))

    @app.route("/api/games/<session_id>/update", methods=["POST"])
    def _games_update(session_id):
        payload = request.get_json(silent=True) or {}
        sess = update_session(session_id, payload)
        if not sess:
            return jsonify({"ok": False, "error": "not found"}), 404
        return jsonify({"ok": True, "session": sess})

    @app.route("/api/games/<session_id>", methods=["DELETE"])
    def _games_delete(session_id):
        return jsonify(delete_session(session_id))

    @app.route("/api/games/<session_id>/save_state", methods=["POST"])
    def _games_save_state(session_id):
        # Accept either raw bytes or JSON {"bytes_b64": "..."}
        raw = request.get_data(cache=False, as_text=False)
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            b64 = payload.get("bytes_b64")
            if b64:
                import base64
                raw = base64.b64decode(b64)
        if not raw:
            return jsonify({"ok": False, "error": "empty body"}), 400
        return jsonify(save_emulator_state_for_session(session_id, raw))

    @app.route("/api/games/<session_id>/save_state", methods=["GET"])
    def _games_get_save_state(session_id):
        blob = load_emulator_state_for_session(session_id)
        if blob is None:
            return jsonify({"ok": False, "error": "no save state"}), 404
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "bytes": len(blob),
            "bytes_b64": __import__("base64").b64encode(blob).decode("ascii"),
        })
