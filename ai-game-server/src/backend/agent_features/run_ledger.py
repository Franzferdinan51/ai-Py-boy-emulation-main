"""
Versioned run ledger for agent observations, actions, and run events.

The ledger is additive and read-only with respect to the emulator. It records
successful agent actions as JSON-safe v1 dictionaries and keeps a bounded
in-memory mirror for no-session / no-ROM scenarios.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .sessions import DEFAULT_DATA_DIR, get_current_session, get_session

_lock = threading.RLock()
_recent_events: Deque[Dict[str, Any]] = deque(
    maxlen=int(os.environ.get("PYBOY_RUN_LEDGER_BUFFER", "1000"))
)

_MAX_TEXT = 160
_MAX_LIST_ITEMS = 16
_MAX_DICT_ITEMS = 16
_MAX_DEPTH = 3
_MAX_LIMIT = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_text(value: Any, limit: int = _MAX_TEXT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, (bytes, bytearray)):
        return _bounded_text(value.decode("utf-8", errors="replace"))
    if depth >= _MAX_DEPTH:
        return _bounded_text(value)
    if isinstance(value, dict):
        safe: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_DICT_ITEMS:
                break
            safe[_bounded_text(key, 64)] = _json_safe(item, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:_MAX_LIST_ITEMS]]
    return _bounded_text(value)


def _session_ledger_path(session: Dict[str, Any]) -> Path:
    data_dir = session.get("data_dir")
    if data_dir:
        return Path(data_dir) / "events" / "run-ledger.jsonl"
    session_id = session.get("id", "unknown")
    return Path(DEFAULT_DATA_DIR) / "games" / str(session_id) / "events" / "run-ledger.jsonl"


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
                if len(records) >= limit:
                    break
    except OSError:
        return []
    return records


def GameActionV1(
    *,
    source: str,
    action: Any,
    frames: Any = 1,
    success: Any = None,
    observation: Any = None,
    changes: Any = None,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "version": "v1",
        "kind": "action",
        "timestamp": timestamp or _now_iso(),
        "source": _bounded_text(source, 64),
        "action": _bounded_text(action, 64) if not isinstance(action, dict) else _json_safe(action),
        "frames": 1,
    }
    try:
        record["frames"] = max(1, min(int(frames or 1), 99))
    except (TypeError, ValueError):
        record["frames"] = 1
    if session_id is not None:
        record["session_id"] = _bounded_text(session_id, 64)
    if success is not None:
        record["success"] = bool(success)
    if observation is not None:
        record["observation"] = _json_safe(observation)
    if changes is not None:
        record["changes"] = _json_safe(changes)
    payload = dict(data or {})
    payload.setdefault("action", record.get("action"))
    payload.setdefault("frames", record["frames"])
    if success is not None:
        payload.setdefault("success", bool(success))
    record["data"] = _json_safe(payload)
    return record


def GameObservationV1(
    *,
    source: str,
    observation: Any,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    action: Any = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "version": "v1",
        "kind": "observation",
        "timestamp": timestamp or _now_iso(),
        "source": _bounded_text(source, 64),
        "observation": _json_safe(observation),
    }
    if session_id is not None:
        record["session_id"] = _bounded_text(session_id, 64)
    if action is not None:
        record["action"] = _json_safe(action)
    payload = dict(data or {})
    payload.setdefault("observation", record["observation"])
    if action is not None:
        payload.setdefault("action", _json_safe(action))
    record["data"] = _json_safe(payload)
    return record


def GameRunEventV1(
    *,
    source: str,
    action: Any,
    observation: Any,
    changes: Any,
    session_id: Optional[str] = None,
    success: Any = True,
    timestamp: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action_record = action if isinstance(action, dict) else GameActionV1(
        source=source,
        action=action,
        session_id=session_id,
        success=success,
    )
    observation_record = observation if isinstance(observation, dict) else GameObservationV1(
        source=source,
        observation=observation,
        session_id=session_id,
        action=action_record,
    )
    record: Dict[str, Any] = {
        "version": "v1",
        "kind": "run_event",
        "timestamp": timestamp or _now_iso(),
        "source": _bounded_text(source, 64),
        "action": _json_safe(action_record),
        "observation": _json_safe(observation_record),
        "changes": _json_safe(changes),
        "success": bool(success),
    }
    if session_id is not None:
        record["session_id"] = _bounded_text(session_id, 64)
    payload = dict(data or {})
    payload.setdefault("action", record["action"])
    payload.setdefault("observation", record["observation"])
    payload.setdefault("changes", record["changes"])
    payload.setdefault("success", record["success"])
    record["data"] = _json_safe(payload)
    return record


def record_run_event(
    *,
    source: str,
    action: Any,
    observation: Any,
    changes: Any,
    session_id: Optional[str] = None,
    success: Any = True,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    active_session = get_current_session()
    resolved_session_id = session_id or (active_session or {}).get("id")
    event = GameRunEventV1(
        source=source,
        action=action,
        observation=observation,
        changes=changes,
        session_id=resolved_session_id,
        success=success,
        data=data,
    )

    with _lock:
        _recent_events.append(event)

    if active_session and active_session.get("id") == resolved_session_id:
        try:
            _append_jsonl(_session_ledger_path(active_session), event)
        except OSError:
            pass

    return {"ok": True, "event": event}


def get_run_events(
    limit: int = 20,
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, _MAX_LIMIT))

    active_session = get_current_session()
    resolved_session = None
    if session_id:
        resolved_session = get_session(session_id)
    elif active_session:
        resolved_session = active_session

    records: List[Dict[str, Any]] = []
    if resolved_session:
        try:
            records.extend(_read_jsonl(_session_ledger_path(resolved_session), limit=1000))
        except OSError:
            pass

    with _lock:
        records.extend(list(_recent_events))

    deduped: List[Dict[str, Any]] = []
    seen_ids = set()
    for event in sorted(records, key=lambda item: item.get("timestamp", ""), reverse=True):
        if not isinstance(event, dict):
            continue
        if session_id and event.get("session_id") != session_id:
            continue
        event_id = event.get("id")
        dedupe_key = event_id or json.dumps(event, sort_keys=True, ensure_ascii=False)
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        deduped.append(_json_safe(event))
        if len(deduped) >= limit:
            break

    return {
        "ok": True,
        "events": deduped,
        "count": len(deduped),
        "limit": limit,
        "session_id": session_id or (resolved_session or {}).get("id"),
        "loaded": bool((resolved_session or active_session) and (resolved_session or active_session).get("id")),
        "timestamp": _now_iso(),
    }


def register_run_ledger_routes(app, *, game_state_getter=None):
    from flask import jsonify, request

    @app.route("/api/agent/runs/events", methods=["GET"])
    def _runs_events():
        current_state = game_state_getter() if callable(game_state_getter) else {}
        session_id = request.args.get("session_id")
        limit = request.args.get("limit", 20)
        payload = get_run_events(limit=limit, session_id=session_id)
        payload["loaded"] = bool(current_state.get("rom_loaded"))
        payload["active_emulator"] = current_state.get("active_emulator")
        payload["rom_loaded"] = bool(current_state.get("rom_loaded"))
        return jsonify(payload), 200
