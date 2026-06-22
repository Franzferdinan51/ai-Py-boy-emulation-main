"""
routes/config — Configuration validation and introspection endpoints.

Extracted from server.py (was @app.route('/api/config/validate'), /api/config).
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from flask import jsonify


def register_config_routes(
    app,
    *,
    secure_config: Any = None,
    secure_config_available: bool = False,
    host: str = "0.0.0.0",
    port: int = 5002,
    debug: bool = False,
) -> None:
    """Register config-related routes."""

    @app.route("/api/config/validate", methods=["GET"])
    def validate_configuration():
        """Validate system configuration and environment variables"""
        if secure_config_available and secure_config is not None:
            validation = secure_config.validate_environment_variables()
            safe_config = secure_config.get_safe_config()

            response_data = {
                "validation": validation,
                "configuration": safe_config,
                "timestamp": time.time(),
            }

            if not validation["valid"]:
                return jsonify(response_data), 400
            elif validation["warnings"]:
                return jsonify(response_data), 200  # OK but with warnings
            else:
                return jsonify(response_data), 200
        else:
            return jsonify({
                "error": "Configuration validation not available",
                "fallback": os.environ.get("FLASK_ENV", "development"),
            }), 503

    @app.route("/api/config", methods=["GET"])
    def get_configuration():
        """Get safe configuration information"""
        if secure_config_available and secure_config is not None:
            return jsonify(secure_config.get_safe_config()), 200
        else:
            return jsonify({
                "error": "Configuration manager not available",
                "basic_config": {
                    "host": host,
                    "port": port,
                    "debug": debug,
                    "environment": os.environ.get("FLASK_ENV", "development"),
                },
            }), 503
