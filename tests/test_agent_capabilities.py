from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-game-server", "src"))

from backend.agent_features.memory import (
    add_control_pattern,
    add_failure_reflection,
    add_note,
)
from backend.agent_features.sessions import (
    create_session,
    delete_session,
    set_active_session,
    update_session,
)


def test_capability_snapshot_uses_active_session_memory_and_routines(tmp_path):
    from backend.agent_features.agent_capabilities import build_capability_snapshot

    created = create_session(name="Capability Test", data_dir=str(tmp_path))
    session = created["session"]
    session_id = session["id"]
    set_active_session(session_id, data_dir=str(tmp_path))
    update_session(
        session_id,
        {
            "active_routine": "heal_party",
            "routines": [
                {
                    "id": "heal_party",
                    "name": "heal_party",
                    "status": "ready",
                    "steps": [{"action": "UP", "frames": 1}],
                }
            ],
        },
        data_dir=str(tmp_path),
    )
    add_note(session_id, "Pokemon Center is one screen north.", data_dir=str(tmp_path))
    add_control_pattern(
        session_id,
        ["UP", "A"],
        "Enter a doorway from the overworld",
        data_dir=str(tmp_path),
    )

    try:
        snapshot = build_capability_snapshot(
            agent_state={"current_goal": "Heal the party", "current_task": "Reach the Pokemon Center"},
            game_context={
                "loaded": True,
                "game_mode": "exploration",
                "position": {"map_name": "Viridian City", "x": 12, "y": 8},
                "recommendations": ["Heal party at Pokemon Center"],
                "battle": {"in_battle": False},
                "health_summary": {"needs_healing": True},
            },
            session_id=session_id,
            data_dir=str(tmp_path),
        )
    finally:
        delete_session(session_id, data_dir=str(tmp_path))

    assert snapshot["active_session_id"] == session_id
    assert snapshot["active_routine"] == "heal_party"
    assert snapshot["next_recommended_action"]["action"] == "HEAL"
    assert snapshot["memory_summary"]["total_records"] >= 2
    assert snapshot["memory_summary"]["learned_control_patterns"][0]["outcome"] == "Enter a doorway from the overworld"
    assert snapshot["skill_drafts"][0]["source"] == "memory.control_pattern"


def test_routine_upsert_creates_reusable_playbook_metadata(tmp_path):
    from backend.agent_features.agent_capabilities import upsert_session_routine

    created = create_session(name="Routine Test", data_dir=str(tmp_path))
    session_id = created["session"]["id"]
    set_active_session(session_id, data_dir=str(tmp_path))

    try:
        result = upsert_session_routine(
            session_id=session_id,
            payload={
                "name": "pewter_entry",
                "description": "Line up on the city gate and walk north.",
                "steps": [{"action": "UP", "frames": 4}],
                "tags": ["navigation"],
            },
            data_dir=str(tmp_path),
        )
    finally:
        delete_session(session_id, data_dir=str(tmp_path))

    routine = result["routine"]
    assert result["ok"] is True
    assert routine["name"] == "pewter_entry"
    assert routine["kind"] == "playbook"
    assert routine["origin"] == "operator"
    assert routine["steps"][0]["action"] == "UP"
    assert routine["skill_draft"]["status"] == "draft"


def test_capability_snapshot_includes_recent_failure_guardrails(tmp_path):
    from backend.agent_features.agent_capabilities import build_capability_snapshot

    created = create_session(name="Guardrail Test", data_dir=str(tmp_path))
    session_id = created["session"]["id"]
    set_active_session(session_id, data_dir=str(tmp_path))
    add_failure_reflection(
        session_id,
        trigger="Repeated A press into dialogue lock",
        error="step returned false",
        consequence="Agent wasted three turns without advancing text.",
        defense="Check dialogue or menu state before retrying A more than once.",
        severity="high",
        source="api.agent.act",
        data_dir=str(tmp_path),
    )

    try:
        snapshot = build_capability_snapshot(
            agent_state={"current_goal": "Leave dialogue safely"},
            game_context={"loaded": True, "game_mode": "dialogue"},
            session_id=session_id,
            data_dir=str(tmp_path),
        )
    finally:
        delete_session(session_id, data_dir=str(tmp_path))

    assert snapshot["memory_summary"]["by_type"]["failure_reflection"] == 1
    assert snapshot["memory_summary"]["recent_failure_reflections"][0]["trigger"] == "Repeated A press into dialogue lock"
    assert snapshot["guardrails"][0]["defense"] == "Check dialogue or menu state before retrying A more than once."
    assert snapshot["guardrails"][0]["severity"] == "high"
    assert snapshot["auto_learning_signals"]["failure_reflection_count"] == 1
    assert any(tool["name"] == "get_agent_guardrails" for tool in snapshot["available_tools"])


def test_capability_snapshot_exposes_skill_workshop_payload_and_draft_preview(tmp_path):
    from backend.agent_features.agent_capabilities import (
        build_capability_snapshot,
        get_skill_workshop_draft,
    )

    workspace_skills_dir = tmp_path / "workspace-skills"
    created = create_session(name="Workshop Test", data_dir=str(tmp_path))
    session_id = created["session"]["id"]
    set_active_session(session_id, data_dir=str(tmp_path))
    update_session(
        session_id,
        {
            "routines": [
                {
                    "id": "viridian_entry",
                    "name": "viridian_entry",
                    "description": "Move through the city gate and confirm the transition.",
                    "status": "ready",
                    "steps": [{"action": "UP", "frames": 4}, {"action": "A", "frames": 1}],
                    "skill_draft": {
                        "id": "skill-viridian-entry",
                        "name": "viridian_entry",
                        "source": "routine.upsert",
                        "status": "draft",
                        "summary": "Move through the city gate and confirm the transition.",
                    },
                }
            ],
        },
        data_dir=str(tmp_path),
    )
    add_note(session_id, "The north gate triggers a short confirmation prompt.", data_dir=str(tmp_path))
    add_control_pattern(
        session_id,
        ["UP", "A"],
        "Enter Viridian gate cleanly",
        note="Use A once after the movement to clear the prompt.",
        data_dir=str(tmp_path),
    )
    add_failure_reflection(
        session_id,
        trigger="Repeated movement into blocked tile",
        error="step returned false",
        consequence="The agent burned extra turns without crossing the gate.",
        defense="Check the position delta before repeating UP more than twice.",
        severity="medium",
        source="api.agent.act",
        data_dir=str(tmp_path),
    )

    try:
        snapshot = build_capability_snapshot(
            agent_state={"current_goal": "Leave Viridian City"},
            game_context={"loaded": True, "game_mode": "exploration"},
            session_id=session_id,
            data_dir=str(tmp_path),
            workspace_skills_dir=str(workspace_skills_dir),
        )
        detail = get_skill_workshop_draft(
            draft_id="skill-viridian-entry",
            agent_state={"current_goal": "Leave Viridian City"},
            game_context={"loaded": True, "game_mode": "exploration"},
            session_id=session_id,
            data_dir=str(tmp_path),
            workspace_skills_dir=str(workspace_skills_dir),
        )
    finally:
        delete_session(session_id, data_dir=str(tmp_path))

    workshop = snapshot["skill_workshop"]
    assert workshop["draft_count"] >= 2
    assert workshop["workspace_precedence"] == "repo-local"
    assert workshop["workspace_skills_root"] == str(workspace_skills_dir.resolve())
    assert workshop["install_route"] == "/api/agent/skills/workshop/install"
    assert detail["ok"] is True
    assert detail["draft"]["id"] == "skill-viridian-entry"
    assert detail["artifact"]["relative_install_dir"] == "generated/viridian-entry"
    assert detail["artifact"]["install_path"].endswith("workspace-skills/generated/viridian-entry/SKILL.md")
    assert detail["artifact"]["frontmatter"]["name"] == "viridian-entry"
    assert "Check the position delta before repeating UP more than twice." in detail["artifact"]["content"]
    assert "get_agent_context" in detail["artifact"]["content"]
