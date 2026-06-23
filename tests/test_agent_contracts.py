from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))
os.environ.setdefault("BACKEND_PORT", "5002")
os.environ.setdefault("FLASK_ENV", "development")

import backend.server as srv  # noqa: E402
from backend.agent_features.sessions import create_session, delete_session, set_active_session

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
            (
                "loaded",
                "game_mode",
                "position",
                "party",
                "inventory",
                "battle",
                "active_session_id",
                "active_routine",
                "available_tools",
                "memory_summary",
                "guardrails",
                "next_recommended_action",
            ),
        ),
        ("post", "/api/agent/act", 400, ("error", "timestamp")),
        ("post", "/api/save_state", 400, ("error",)),
        ("post", "/api/load_state", 400, ("error",)),
        ("get", "/api/screen", 400, ("error",)),
        ("get", "/api/stream", 200, ()),
        ("post", "/api/game/button", 400, ("error",)),
        (
            "get",
            "/api/agent/state",
            200,
            (
                "active_session_id",
                "active_routine",
                "available_tools",
                "memory_summary",
                "guardrails",
                "next_recommended_action",
            ),
        ),
        ("get", "/api/agent/toolbelt", 200, ("available_tools", "tool_groups", "active_session_id")),
        ("get", "/api/agent/routines", 200, ("routines", "active_routine", "active_session_id")),
        ("get", "/api/agent/guardrails", 200, ("active_session_id", "guardrails", "recent_failure_reflections")),
        (
            "get",
            "/api/agent/skills/workshop",
            200,
            ("active_session_id", "drafts", "workspace_skills_root", "install_route"),
        ),
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


def test_agent_routines_post_is_registered_and_uses_safe_defaults():
    response = _request(
        "post",
        "/api/agent/routines",
        json={"name": "Viridian setup", "steps": [{"action": "UP", "frames": 1}]},
    )
    assert response.status_code in {200, 400}, response.get_data(as_text=True)
    data = response.get_json() or {}
    assert "timestamp" in data
    if response.status_code == 400:
        assert "error" in data
    else:
        assert "routine" in data


class _FailureLearningStubEmulator:
    def step(self, action: str, frames: int) -> bool:
        return False

    def get_position(self):
        return {"x": 3, "y": 5, "map_id": 1, "map_name": "Viridian City"}

    def get_battle_info(self):
        return {"in_battle": False}

    def get_party_info(self):
        return [{"species": "Bulbasaur", "hp_percent": 100}]


def test_failed_agent_action_persists_read_only_guardrail_metadata(tmp_path):
    stub = _FailureLearningStubEmulator()
    original = srv.emulators.get("failure_learning_stub")
    original_game_state = dict(srv.game_state)
    created = create_session(name="Failure Learning", data_dir=str(tmp_path))
    session_id = created["session"]["id"]
    set_active_session(session_id, data_dir=str(tmp_path))
    srv.emulators["failure_learning_stub"] = stub
    srv.game_state.update(
        {
            "rom_loaded": True,
            "active_emulator": "failure_learning_stub",
            "rom_name": "Failure Learning ROM",
            "rom_path": "/tmp/failure-learning.gb",
        }
    )

    try:
        response = _request("post", "/api/agent/act", json={"action": "A", "frames": 1})
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json() or {}
        assert data["success"] is False
        assert "failure_reflection" in data
        assert data["failure_reflection"]["defense"]

        guardrails = _request("get", "/api/agent/guardrails")
        assert guardrails.status_code == 200, guardrails.get_data(as_text=True)
        guardrail_data = guardrails.get_json() or {}
        assert guardrail_data["active_session_id"] == session_id
        assert guardrail_data["guardrails"][0]["trigger"] == "Action A did not advance emulator state"
        assert guardrail_data["recent_failure_reflections"][0]["consequence"]
    finally:
        delete_session(session_id, data_dir=str(tmp_path))
        if original is None:
            srv.emulators.pop("failure_learning_stub", None)
        else:
            srv.emulators["failure_learning_stub"] = original
        srv.game_state.clear()
        srv.game_state.update(original_game_state)


def test_agent_skill_workshop_preview_and_install_routes(monkeypatch, tmp_path):
    workspace_skills_dir = tmp_path / "skills"
    monkeypatch.setenv("PYBOY_WORKSPACE_SKILLS_DIR", str(workspace_skills_dir))

    created = create_session(name="Workshop Route Test", data_dir=str(tmp_path))
    session_id = created["session"]["id"]
    set_active_session(session_id, data_dir=str(tmp_path))

    try:
        upsert = _request(
            "post",
            "/api/agent/routines",
            json={
                "session_id": session_id,
                "name": "Pewter Gate",
                "description": "Walk north and confirm the gate transition.",
                "steps": [{"action": "UP", "frames": 3}, {"action": "A", "frames": 1}],
                "tags": ["navigation"],
            },
        )
        assert upsert.status_code == 200, upsert.get_data(as_text=True)
        draft_id = (upsert.get_json() or {})["routine"]["skill_draft"]["id"]

        listing = _request("get", "/api/agent/skills/workshop")
        assert listing.status_code == 200, listing.get_data(as_text=True)
        listing_data = listing.get_json() or {}
        assert listing_data["workspace_skills_root"] == str(workspace_skills_dir.resolve())
        assert any(draft["id"] == draft_id for draft in listing_data["drafts"])

        preview = _request("get", f"/api/agent/skills/workshop/{draft_id}")
        assert preview.status_code == 200, preview.get_data(as_text=True)
        preview_data = preview.get_json() or {}
        assert preview_data["artifact"]["relative_install_dir"] == "generated/pewter-gate"
        assert preview_data["artifact"]["install_path"].endswith("skills/generated/pewter-gate/SKILL.md")
        assert preview_data["artifact"]["frontmatter"]["name"] == "pewter-gate"

        install = _request(
            "post",
            "/api/agent/skills/workshop/install",
            json={"draft_id": draft_id},
        )
        assert install.status_code == 200, install.get_data(as_text=True)
        install_data = install.get_json() or {}
        assert install_data["ok"] is True
        assert install_data["installed"] is True
        assert install_data["artifact"]["relative_install_dir"] == "generated/pewter-gate"
        installed_path = workspace_skills_dir / "generated" / "pewter-gate" / "SKILL.md"
        assert installed_path.exists()
        contents = installed_path.read_text(encoding="utf-8")
        assert "name: pewter-gate" in contents
        assert "Walk north and confirm the gate transition." in contents

        second_install = _request(
            "post",
            "/api/agent/skills/workshop/install",
            json={"draft_id": draft_id},
        )
        assert second_install.status_code == 409, second_install.get_data(as_text=True)
        conflict = second_install.get_json() or {}
        assert conflict["ok"] is False
        assert "already exists" in conflict["error"]
    finally:
        delete_session(session_id, data_dir=str(tmp_path))
