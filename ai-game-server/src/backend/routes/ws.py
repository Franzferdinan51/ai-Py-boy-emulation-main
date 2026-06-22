"""
routes/ws — WebSocket server control endpoints.

Extracted from server.py. Provides status/start/stop control over the WebSocket
streaming server. Depends on a websocket_runner object that exposes:

    - running()       -> bool
    - port()          -> int
    - clients()       -> int
    - start()         -> None  (raises on failure)
    - stop()          -> None
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from flask import jsonify

logger = logging.getLogger(__name__)


class WebSocketRunner:
    """Adapter for the existing websocket server globals.

    Wraps the module-level ws_server_running / WS_PORT / ws_clients / start /
    stop functions in server.py into a single callable interface.
    """

    def __init__(self, *, is_running, get_port, get_clients, start_fn, stop_fn):
        self._is_running = is_running
        self._get_port = get_port
        self._get_clients = get_clients
        self._start = start_fn
        self._stop = stop_fn

    def running(self) -> bool:
        try:
            return bool(self._is_running())
        except Exception:  # noqa: BLE001
            return False

    def port(self) -> int:
        try:
            return int(self._get_port())
        except Exception:  # noqa: BLE001
            return 5003

    def clients(self) -> int:
        try:
            return int(self._get_clients())
        except Exception:  # noqa: BLE001
            return 0

    def start(self) -> None:
        self._start()

    def stop(self) -> None:
        self._stop()


def register_ws_routes(
    app,
    *,
    websocket_runner: Optional[Any] = None,
) -> None:
    """Register websocket control routes.

    websocket_runner should be a WebSocketRunner instance. If None, a stub is
    used that always reports "not running" — useful for tests.
    """
    if websocket_runner is None:
        class _StubRunner:
            def running(self) -> bool:
                return False

            def port(self) -> int:
                return 5003

            def clients(self) -> int:
                return 0

            def start(self) -> None:
                raise RuntimeError("WebSocket runner not configured")

            def stop(self) -> None:
                return None

        websocket_runner = _StubRunner()

    @app.route("/api/ws/status", methods=["GET"])
    def get_websocket_status():
        """Get WebSocket server status."""
        port = websocket_runner.port()
        return jsonify({
            "running": websocket_runner.running(),
            "port": port,
            "url": f"ws://localhost:{port}/api/ws/stream",
            "clients": websocket_runner.clients(),
            "timestamp": datetime.now().isoformat(),
        }), 200

    @app.route("/api/ws/start", methods=["POST"])
    def start_websocket_endpoint():
        """Start the WebSocket server."""
        try:
            websocket_runner.start()
            port = websocket_runner.port()
            return jsonify({
                "success": True,
                "message": "WebSocket server started",
                "url": f"ws://localhost:{port}/api/ws/stream",
                "port": port,
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/ws/stop", methods=["POST"])
    def stop_websocket_endpoint():
        """Stop the WebSocket server."""
        try:
            websocket_runner.stop()
            return jsonify({
                "success": True,
                "message": "WebSocket server stopped",
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Failed to stop WebSocket server: {e}")
            return jsonify({"error": str(e)}), 500
