"""
routes/save_load — Game state save/load endpoints (legacy compat + /api aliases).

Extracted from server.py. Supports both the bare /save_state, /load_state routes
(placeholder compat for GLM4.5-UI) and the /api/save_state, /api/load_state
endpoints (frontend compatibility).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from flask import jsonify

logger = logging.getLogger(__name__)


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
        """Save game state (placeholder for GLM4.5-UI compatibility)"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            # Placeholder implementation
            return jsonify({"success": True, "message": "State saved"}), 200
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return jsonify({"error": "Failed to save state"}), 500

    @app.route("/load_state", methods=["POST"])
    def load_state():
        """Load game state (placeholder for GLM4.5-UI compatibility)"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            # Placeholder implementation
            return jsonify({"success": True, "message": "State loaded"}), 200
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return jsonify({"error": "Failed to load state"}), 500

    @app.route("/api/save_state", methods=["POST"])
    def api_save_state():
        """Save game state API endpoint for frontend compatibility"""
        try:
            current_state = game_state_getter()
            active = current_state.get("active_emulator")
            if not current_state.get("rom_loaded") or not active:
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter()
            emulator = emulators[active]
            if hasattr(emulator, "save_state"):
                state_data = emulator.save_state()
                if state_data:
                    saved_states[active] = state_data
                    return jsonify({
                        "success": True,
                        "message": "State saved successfully",
                        "bytes": len(state_data),
                    }), 200
                return jsonify({"error": "Failed to save state data"}), 500
            return jsonify({"error": "Emulator does not support save state"}), 400

        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return jsonify({"error": "Failed to save state", "details": str(e)}), 500

    @app.route("/api/load_state", methods=["POST"])
    def api_load_state():
        """Load game state API endpoint for frontend compatibility"""
        try:
            current_state = game_state_getter()
            active = current_state.get("active_emulator")
            if not current_state.get("rom_loaded") or not active:
                return jsonify({"error": "No ROM loaded"}), 400
            if active not in saved_states:
                return jsonify({"error": "No saved state available"}), 400

            emulators = emulators_getter()
            emulator = emulators[active]
            if hasattr(emulator, "load_state"):
                ok = emulator.load_state(saved_states[active])
                if ok:
                    return jsonify({
                        "success": True,
                        "message": "State loaded successfully",
                        "bytes": len(saved_states[active]),
                    }), 200
                return jsonify({"error": "Emulator failed to load saved state"}), 500
            return jsonify({"error": "Emulator does not support load state"}), 400

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return jsonify({"error": "Failed to load state", "details": str(e)}), 500
