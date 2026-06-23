from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))
os.environ.setdefault("BACKEND_PORT", "5002")
os.environ.setdefault("FLASK_ENV", "development")

import backend.server as srv  # noqa: E402

client = srv.app.test_client()


class _LedgerStubEmulator:
    def __init__(self) -> None:
        self._step_calls = 0

    def step(self, action: str, frames: int) -> bool:
        self._step_calls += 1
        return True

    def get_position(self):
        return {"x": 7, "y": 9, "map_id": 3, "map_name": "Route 1"}

    def get_battle_info(self):
        return {"in_battle": False, "battle_type": "none"}

    def get_party_info(self):
        return [{"species": "Pikachu", "hp_percent": 100}]


def _request(method: str, path: str, **kwargs):
    return getattr(client, method.lower())(path, **kwargs)


def test_agent_act_attaches_v1_event_and_events_endpoint_lists_it():
    stub = _LedgerStubEmulator()
    original = srv.emulators.get("ledger_stub")
    srv.emulators["ledger_stub"] = stub
    srv.game_state.update(
        {
            "rom_loaded": True,
            "active_emulator": "ledger_stub",
            "rom_name": "Test ROM",
            "rom_path": "/tmp/test-rom.gb",
        }
    )

    try:
        response = _request("post", "/api/agent/act", json={"action": "A", "frames": 2})
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json() or {}
        assert data.get("success") is True
        assert data.get("action") == "A"
        assert "event" in data

        event = data["event"]
        assert event["version"] == "v1"
        assert event["source"] == "api.agent.act"
        assert event["action"]["version"] == "v1"
        assert event["observation"]["version"] == "v1"
        assert event["timestamp"]

        events_response = _request("get", "/api/agent/runs/events?limit=1")
        assert events_response.status_code == 200, events_response.get_data(as_text=True)
        events_data = events_response.get_json() or {}
        assert events_data["count"] == 1
        assert events_data["events"][0]["version"] == "v1"
        assert events_data["events"][0]["action"]["action"] == "A"
    finally:
        if original is None:
            srv.emulators.pop("ledger_stub", None)
        else:
            srv.emulators["ledger_stub"] = original


def test_run_ledger_events_endpoint_is_safe_without_rom():
    srv.game_state.update({"rom_loaded": False, "active_emulator": None})

    response = _request("get", "/api/agent/runs/events?limit=500")
    assert response.status_code == 200, response.get_data(as_text=True)
    data = response.get_json() or {}
    assert data["limit"] == 100
    assert data["count"] >= 0
    assert isinstance(data.get("events"), list)
    json.dumps(data)
