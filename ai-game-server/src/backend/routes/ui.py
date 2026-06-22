"""
routes/ui — UI process launch/stop/restart/status endpoints.

Extracted from server.py. Controls the emulator's UI subprocess (if supported).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from flask import jsonify

logger = logging.getLogger(__name__)


def register_ui_routes(
    app,
    *,
    emulators_getter: Callable[[], Dict[str, Any]],
    game_state_getter: Callable[[], Dict[str, Any]],
) -> None:
    """Register UI control routes."""

    @app.route("/api/ui/launch", methods=["POST"])
    def launch_ui():
        """Launch UI process for the current ROM"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]

            if not hasattr(emulator, "launch_ui"):
                return jsonify({"error": "UI control not supported by this emulator"}), 400

            success = emulator.launch_ui()

            if success:
                ui_status = emulator.get_ui_status()
                logger.info("UI process launched successfully")
                return jsonify({
                    "message": "UI launched successfully",
                    "ui_status": ui_status,
                }), 200
            else:
                logger.error("Failed to launch UI process")
                return jsonify({"error": "Failed to launch UI"}), 500

        except Exception as e:
            logger.error(f"Error launching UI: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/ui/stop", methods=["POST"])
    def stop_ui():
        """Stop the UI process"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]

            if not hasattr(emulator, "stop_ui"):
                return jsonify({"error": "UI control not supported by this emulator"}), 400

            success = emulator.stop_ui()

            if success:
                logger.info("UI process stopped successfully")
                return jsonify({"message": "UI stopped successfully"}), 200
            else:
                logger.error("Failed to stop UI process")
                return jsonify({"error": "Failed to stop UI"}), 500

        except Exception as e:
            logger.error(f"Error stopping UI: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/ui/restart", methods=["POST"])
    def restart_ui():
        """Restart the UI process"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]

            if not hasattr(emulator, "restart_ui"):
                return jsonify({"error": "UI control not supported by this emulator"}), 400

            success = emulator.restart_ui()

            if success:
                ui_status = emulator.get_ui_status()
                logger.info("UI process restarted successfully")
                return jsonify({
                    "message": "UI restarted successfully",
                    "ui_status": ui_status,
                }), 200
            else:
                logger.error("Failed to restart UI process")
                return jsonify({"error": "Failed to restart UI"}), 500

        except Exception as e:
            logger.error(f"Error restarting UI: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/ui/status", methods=["GET"])
    def get_ui_status():
        """Get UI process status"""
        try:
            current_state = game_state_getter()
            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]

            if not hasattr(emulator, "get_ui_status"):
                return jsonify({"error": "UI control not supported by this emulator"}), 400

            ui_status = emulator.get_ui_status()

            return jsonify({
                "ui_status": ui_status,
                "rom_loaded": current_state["rom_loaded"],
                "active_emulator": current_state["active_emulator"],
            }), 200

        except Exception as e:
            logger.error(f"Error getting UI status: {e}")
            return jsonify({"error": "Internal server error"}), 500
