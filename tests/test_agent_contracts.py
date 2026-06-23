from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))
os.environ.setdefault("BACKEND_PORT", "5002")
os.environ.setdefault("FLASK_ENV", "development")

import backend.server as srv  # noqa: E402

client = srv.app.test_client()


def _request(method: str, path: str, **kwargs):
    return getattr(client, method.lower())(path, **kwargs)


def test_canonical_game_agent_contract_matrix():
    canonical_routes = [
        ("get", "/api/game/state", 200, ("rom_loaded", "active_emulator", "rom_path")),
        (
            "get",
            "/api/agent/context",
            200,
            ("loaded", "game_mode", "position", "party", "inventory", "battle"),
        ),
        ("post", "/api/agent/act", 400, ("error", "timestamp")),
        ("post", "/api/save_state", 400, ("error",)),
        ("post", "/api/load_state", 400, ("error",)),
        ("get", "/api/screen", 400, ("error",)),
        ("get", "/api/stream", 200, ()),
        ("post", "/api/game/button", 400, ("error",)),
    ]

    for method, path, expected_status, required_keys in canonical_routes:
        response = _request(method, path, json={} if method == "post" else None)
        assert response.status_code == expected_status, (path, response.status_code)
        if path == "/api/stream":
            body = response.get_data(as_text=True)
            assert response.content_type.startswith("text/event-stream")
            assert '"error": "No ROM loaded"' in body
            continue
        data = response.get_json() or {}
        for key in required_keys:
            assert key in data, (path, key, data)


def test_supported_legacy_aliases_remain_registered():
    alias_routes = [
        ("post", "/api/action", 400),
        ("post", "/api/game/action", 400),
        ("post", "/save_state", 400),
        ("post", "/load_state", 400),
        ("post", "/api/rom/load", 400),
    ]

    for method, path, expected_status in alias_routes:
        response = _request(method, path, json={} if method == "post" else None)
        assert response.status_code == expected_status, (path, response.status_code)
