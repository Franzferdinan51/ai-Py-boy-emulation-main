from __future__ import annotations

import os
import sys

import pytest
from flask import Flask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))

from backend.routes.save_load import register_save_load_routes  # noqa: E402


class FakeEmulator:
    def __init__(self, initial_value: int = 7):
        self.value = initial_value
        self.save_calls = 0
        self.load_calls = 0

    def save_state(self) -> bytes:
        self.save_calls += 1
        return bytes([self.value])

    def load_state(self, state: bytes) -> bool:
        self.load_calls += 1
        if not state:
            return False
        self.value = state[0]
        return True


@pytest.fixture()
def contract_client():
    app = Flask(__name__)
    app.config["TESTING"] = True

    emulators = {"pyboy": FakeEmulator()}
    game_state = {"rom_loaded": True, "active_emulator": "pyboy"}
    saved_states: dict[str, object] = {}

    register_save_load_routes(
        app,
        emulators_getter=lambda: emulators,
        game_state_getter=lambda: game_state,
        saved_states=saved_states,
    )

    return app.test_client(), emulators, game_state, saved_states


def test_named_slots_are_real_and_shared_across_aliases(contract_client):
    client, emulators, game_state, saved_states = contract_client
    emulator = emulators["pyboy"]

    save_response = client.post("/api/save_state", json={"save_name": "slot_alpha"})
    assert save_response.status_code == 200
    save_data = save_response.get_json()
    assert save_data["success"] is True
    assert save_data["name"] == "slot_alpha"
    assert save_data["bytes"] == 1
    assert emulator.save_calls == 1
    assert saved_states == {"pyboy": {"slot_alpha": b"\x07"}}

    emulator.value = 99

    load_response = client.post("/load_state", json={"name": "slot_alpha"})
    assert load_response.status_code == 200
    load_data = load_response.get_json()
    assert load_data["success"] is True
    assert load_data["name"] == "slot_alpha"
    assert load_data["bytes"] == 1
    assert emulator.load_calls == 1
    assert emulator.value == 7

    emulator.value = 23

    legacy_save_response = client.post("/save_state", json={})
    assert legacy_save_response.status_code == 200
    legacy_save_data = legacy_save_response.get_json()
    assert legacy_save_data["name"] == "quick_save"
    assert legacy_save_data["bytes"] == 1
    assert emulator.save_calls == 2
    assert saved_states["pyboy"]["quick_save"] == b"\x17"

    emulator.value = 41

    api_load_response = client.post("/api/load_state", json={})
    assert api_load_response.status_code == 200
    api_load_data = api_load_response.get_json()
    assert api_load_data["name"] == "quick_save"
    assert api_load_data["bytes"] == 1
    assert emulator.load_calls == 2
    assert emulator.value == 23

    assert game_state["rom_loaded"] is True
    assert game_state["active_emulator"] == "pyboy"


def test_invalid_save_name_is_rejected_safely(contract_client):
    client, emulators, _, saved_states = contract_client
    emulator = emulators["pyboy"]

    response = client.post("/api/save_state", json={"name": "../escape"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]
    assert emulator.save_calls == 0
    assert saved_states == {}

    response = client.post("/api/load_state", json={"save_name": "bad/name"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]
    assert emulator.load_calls == 0
