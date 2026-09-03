"""Regression tests for backend startup wiring."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))
os.environ.setdefault("BACKEND_PORT", "5002")
os.environ.setdefault("FLASK_ENV", "development")

import backend.server as server  # noqa: E402


def test_websocket_compatibility_lifecycle_is_safe_when_disabled():
    """Missing optional websocket implementation must not crash startup."""
    assert server.start_websocket_server() is False
    assert server.stop_websocket_server() is False
    assert server.ws_server_running is False
    assert server.ws_clients == set()


def test_refactored_routes_are_registered_once():
    """Import-time blueprint registration must not be repeated by main()."""
    seen = set()
    duplicates = []
    for rule in server.app.url_map.iter_rules():
        methods = tuple(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        key = (rule.rule, methods)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    assert not duplicates


def test_websocket_status_reports_not_running():
    response = server.app.test_client().get("/api/ws/status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["running"] is False
    assert payload["clients"] == 0
    assert payload["port"] == 5003
