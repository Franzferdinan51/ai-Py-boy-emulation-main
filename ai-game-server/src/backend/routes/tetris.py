"""
routes/tetris — Tetris genetic AI train/status/save/load endpoints.

Extracted from server.py. Provides control over the genetic-AI Tetris trainer
when the tetris-genetic provider is configured.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from flask import jsonify, request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline validators (originally from server.py: validate_string_input /
# sanitize_filename / validate_json_data). Stable, self-contained.
# ---------------------------------------------------------------------------

_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._/-]")


def _validate_string_input(value, field_name, min_length=0, max_length=1000):
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters")
    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    return value


def _sanitize_filename(filename):
    name = (filename or "").strip()
    if not name:
        return ""
    name = name.replace("..", "_")
    return _FILENAME_SAFE.sub("_", name)[:255]


def _validate_json_data(raw, label):
    if not raw:
        raise ValueError(f"{label}: empty body")
    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{label}: invalid JSON ({e})")
    if not isinstance(data, dict):
        raise ValueError(f"{label}: expected object")
    return data


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_tetris_routes(app, *, ai_provider_manager: Any) -> None:
    """Register tetris routes."""

    def _get_tetris_ai():
        if ai_provider_manager is None:
            return None
        providers = getattr(ai_provider_manager, "providers", {}) or {}
        return providers.get("tetris-genetic", {}).get("connector")

    @app.route("/api/tetris/train", methods=["POST"])
    def train_tetris_ai():
        """Train the Tetris genetic AI"""
        try:
            data = request.get_json() or {}
            population_size = data.get("population_size", 20)
            generations = data.get("generations", 5)

            tetris_ai = _get_tetris_ai()
            if not tetris_ai:
                return jsonify({
                    "success": False,
                    "error": "Tetris genetic AI not available",
                }), 400

            # Training is launched asynchronously; we don't depend on game_state
            # directly (the trainer uses its own emulator wrapper).
            logger.info(
                f"Starting Tetris AI training: population_size={population_size}, "
                f"generations={generations}"
            )

            def _train_async():
                try:
                    results = tetris_ai.train_generation(population_size, generations)
                    logger.info(f"Tetris AI training completed: {results}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Tetris AI training failed: {e}")

            training_thread = threading.Thread(target=_train_async, daemon=True)
            training_thread.start()

            return jsonify({
                "success": True,
                "message": "Tetris AI training started",
                "population_size": population_size,
                "generations": generations,
                "provider_status": tetris_ai.get_status(),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error starting Tetris AI training: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
            }), 500

    @app.route("/api/tetris/status", methods=["GET"])
    def get_tetris_status():
        """Get Tetris genetic AI status"""
        try:
            tetris_ai = _get_tetris_ai()
            if not tetris_ai:
                return jsonify({"available": False, "error": "Tetris genetic AI not available"})
            return jsonify({
                "success": True,
                "status": tetris_ai.get_status(),
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error getting Tetris AI status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/tetris/save", methods=["POST"])
    def save_tetris_model():
        """Save Tetris AI model"""
        try:
            data = _validate_json_data(request.get_data(as_text=True), "tetris save request")
            filepath = data.get("filepath")
            if not filepath:
                return jsonify({"success": False, "error": "Filepath required"}), 400
            try:
                filepath = _validate_string_input(filepath, "filepath", min_length=1, max_length=500)
                filepath = _sanitize_filename(filepath)
                if not filepath.endswith((".pkl", ".model", ".dat")):
                    raise ValueError(
                        "Filepath must have a valid model extension (.pkl, .model, .dat)"
                    )
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400

            tetris_ai = _get_tetris_ai()
            if not tetris_ai:
                return jsonify({"success": False, "error": "Tetris genetic AI not available"}), 400

            success = tetris_ai.save_training_state(filepath)
            return jsonify({
                "success": success,
                "message": "Model saved successfully" if success else "Failed to save model",
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error saving Tetris model: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/tetris/load", methods=["POST"])
    def load_tetris_model():
        """Load Tetris AI model"""
        try:
            data = _validate_json_data(request.get_data(as_text=True), "tetris load request")
            filepath = data.get("filepath")
            if not filepath:
                return jsonify({"success": False, "error": "Filepath required"}), 400
            try:
                filepath = _validate_string_input(filepath, "filepath", min_length=1, max_length=500)
                filepath = _sanitize_filename(filepath)
                if not filepath.endswith((".pkl", ".model", ".dat")):
                    raise ValueError(
                        "Filepath must have a valid model extension (.pkl, .model, .dat)"
                    )
            except ValueError as e:
                return jsonify({"success": False, "error": str(e)}), 400

            tetris_ai = _get_tetris_ai()
            if not tetris_ai:
                return jsonify({"success": False, "error": "Tetris genetic AI not available"}), 400

            success = tetris_ai.load_training_state(filepath)
            return jsonify({
                "success": success,
                "message": "Model loaded successfully" if success else "Failed to load model",
            })
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error loading Tetris model: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
