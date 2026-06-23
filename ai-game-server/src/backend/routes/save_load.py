"""
routes/save_load — Game state save/load endpoints (legacy compat + /api aliases).

Extracted from server.py. Supports both the bare /save_state, /load_state routes
and the /api/save_state, /api/load_state endpoints.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

from flask import jsonify

logger = logging.getLogger(__name__)
_VALID_SAVE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _coerce_state_bytes(state_data: Any) -> Optional[bytes]:
    if isinstance(state_data, bytes):
        return state_data
    if isinstance(state_data, bytearray):
        return bytes(state_data)
    if isinstance(state_data, memoryview):
        return state_data.tobytes()
    return None


def _normalize_slot_name(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if "name" in payload:
        raw_name = payload.get("name")
    elif "save_name" in payload:
        raw_name = payload.get("save_name")
    else:
        return "quick_save", None

    if raw_name is None:
        return None, "Invalid save name"
    if not isinstance(raw_name, str):
        return None, "Invalid save name"

    name = raw_name.strip()
    if not name or not _VALID_SAVE_NAME.fullmatch(name):
        return None, "Invalid save name"
    return name, None


def _active_emulator_key(active: Any) -> Any:
    try:
        hash(active)
    except TypeError:
        return str(active)
    return active


def _slot_store(saved_states: Dict[str, Any], active_key: Any) -> Dict[str, bytes]:
    existing = saved_states.get(active_key)
    if isinstance(existing, dict):
        return existing
    if existing is None:
        slots: Dict[str, bytes] = {}
    else:
        legacy_bytes = _coerce_state_bytes(existing)
        slots = {"quick_save": legacy_bytes} if legacy_bytes is not None else {}
    saved_states[active_key] = slots
    return slots


def _resolve_payload_name(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    return _normalize_slot_name(payload)


def _load_request_payload() -> Dict[str, Any]:
    from flask import request

    payload: Dict[str, Any] = {}
    json_payload = request.get_json(silent=True) or {}
    if isinstance(json_payload, dict):
        payload.update(json_payload)
    if request.form:
        payload.update(request.form.to_dict())
    return payload


def _get_emulator_context(
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Optional[Dict[str, Any]], Optional[Any]]:
    current_state = game_state_getter() or {}
    active = current_state.get("active_emulator")
    if not current_state.get("rom_loaded") or not active:
        return current_state, active, None, "No ROM loaded"

    emulators = emulators_getter() or {}
    emulator = emulators.get(active)
    if emulator is None:
        return current_state, active, emulators, "No ROM loaded"
    return current_state, active, emulators, None


def _save_state_impl(
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    saved_states: Dict[str, Any],
) -> Tuple[Any, int]:
    payload = _load_request_payload()
    slot_name, error = _resolve_payload_name(payload)
    if error:
        return jsonify({"error": error}), 400

    _, active, emulators, rom_error = _get_emulator_context(
        game_state_getter=game_state_getter,
        emulators_getter=emulators_getter,
    )
    if rom_error:
        return jsonify({"error": rom_error}), 400

    emulator = emulators[active]
    if not hasattr(emulator, "save_state"):
        return jsonify({"error": "Emulator does not support save state"}), 400

    try:
        state_data = _coerce_state_bytes(emulator.save_state())
        if not state_data:
            return jsonify({"error": "Failed to save state data"}), 500

        active_key = _active_emulator_key(active)
        slots = _slot_store(saved_states, active_key)
        slots[slot_name] = state_data
        return (
            jsonify(
                {
                    "success": True,
                    "message": "State saved successfully",
                    "name": slot_name,
                    "save_name": slot_name,
                    "bytes": len(state_data),
                }
            ),
            200,
        )
    except Exception as exc:
        logger.error("Error saving state: %s", exc)
        return jsonify({"error": "Failed to save state", "details": str(exc)}), 500


def _load_state_impl(
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    saved_states: Dict[str, Any],
) -> Tuple[Any, int]:
    payload = _load_request_payload()
    slot_name, error = _resolve_payload_name(payload)
    if error:
        return jsonify({"error": error}), 400

    _, active, emulators, rom_error = _get_emulator_context(
        game_state_getter=game_state_getter,
        emulators_getter=emulators_getter,
    )
    if rom_error:
        return jsonify({"error": rom_error}), 400

    active_key = _active_emulator_key(active)
    slot_store = saved_states.get(active_key)
    state_data: Optional[bytes] = None
    if isinstance(slot_store, dict):
        state_data = _coerce_state_bytes(slot_store.get(slot_name))
    elif slot_name == "quick_save":
        state_data = _coerce_state_bytes(slot_store)

    if not state_data:
        return jsonify({"error": "No saved state available"}), 400

    emulator = emulators[active]
    if not hasattr(emulator, "load_state"):
        return jsonify({"error": "Emulator does not support load state"}), 400

    try:
        ok = emulator.load_state(state_data)
        if not ok:
            return jsonify({"error": "Emulator failed to load saved state"}), 500
        return (
            jsonify(
                {
                    "success": True,
                    "message": "State loaded successfully",
                    "name": slot_name,
                    "save_name": slot_name,
                    "bytes": len(state_data),
                }
            ),
            200,
        )
    except Exception as exc:
        logger.error("Error loading state: %s", exc)
        return jsonify({"error": "Failed to load state", "details": str(exc)}), 500


def register_save_load_routes(
    app,
    *,
    emulators_getter: Callable[[], Dict[str, Any]],
    game_state_getter: Callable[[], Dict[str, Any]],
    saved_states: Dict[str, Any],
) -> None:
    """Register save/load state routes."""

    @app.route("/save_state", methods=["POST"])
    def save_state():
        """Save game state for legacy compatibility."""
        return _save_state_impl(
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            saved_states=saved_states,
        )

    @app.route("/load_state", methods=["POST"])
    def load_state():
        """Load game state for legacy compatibility."""
        return _load_state_impl(
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            saved_states=saved_states,
        )

    @app.route("/api/save_state", methods=["POST"])
    def api_save_state():
        """Save game state API endpoint for frontend compatibility."""
        return _save_state_impl(
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            saved_states=saved_states,
        )

    @app.route("/api/load_state", methods=["POST"])
    def api_load_state():
        """Load game state API endpoint for frontend compatibility."""
        return _load_state_impl(
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            saved_states=saved_states,
        )
