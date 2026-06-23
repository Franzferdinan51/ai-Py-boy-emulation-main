"""
Hermes-inspired, adapter-first agent capability helpers.

This module is intentionally read-mostly. It derives planner-friendly
capability state from the existing session, memory, and run data without
introducing a second emulator authority.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import get_memory, summarize_memory
from .sessions import get_current_session, get_session, update_session

_TOOLBELT = [
    {
        "name": "get_agent_context",
        "access": "read-only",
        "category": "context",
        "backend_route": "/api/agent/context",
        "mcp_tool": "get_agent_context",
        "description": "Get a full structured gameplay snapshot for planning.",
    },
    {
        "name": "get_game_mode",
        "access": "read-only",
        "category": "context",
        "backend_route": "/api/agent/mode",
        "mcp_tool": "get_game_mode",
        "description": "Check whether the game is in exploration, battle, menu, or title mode.",
    },
    {
        "name": "get_agent_toolbelt",
        "access": "read-only",
        "category": "capabilities",
        "backend_route": "/api/agent/toolbelt",
        "mcp_tool": "get_agent_toolbelt",
        "description": "Inspect available routines, memory summary, and planner signals.",
    },
    {
        "name": "get_agent_routines",
        "access": "read-only",
        "category": "capabilities",
        "backend_route": "/api/agent/routines",
        "mcp_tool": "get_agent_routines",
        "description": "List saved routines plus generated routine suggestions.",
    },
    {
        "name": "get_agent_guardrails",
        "access": "read-only",
        "category": "capabilities",
        "backend_route": "/api/agent/guardrails",
        "mcp_tool": "get_agent_guardrails",
        "description": "Review recent failure reflections and do-not-repeat guardrails before acting.",
    },
    {
        "name": "get_agent_skill_workshop",
        "access": "read-only",
        "category": "capabilities",
        "backend_route": "/api/agent/skills/workshop",
        "mcp_tool": "get_agent_skill_workshop",
        "description": "Inspect generated OpenClaw-compatible skill drafts, previews, and install metadata.",
    },
    {
        "name": "act_and_observe",
        "access": "mutating",
        "category": "actions",
        "backend_route": "/api/agent/act",
        "mcp_tool": "act_and_observe",
        "description": "Execute one canonical action and return the resulting observation.",
    },
]

_TOOL_GROUPS = {
    "context": ["get_agent_context", "get_game_mode"],
    "capabilities": ["get_agent_toolbelt", "get_agent_routines", "get_agent_guardrails", "get_agent_skill_workshop"],
    "actions": ["act_and_observe"],
}

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_WORKSPACE_SKILLS_DIR = _REPO_ROOT / "skills"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:12]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or "generated-skill"


def _workspace_skills_root(workspace_skills_dir: Optional[str] = None) -> Path:
    configured = workspace_skills_dir or os.environ.get("PYBOY_WORKSPACE_SKILLS_DIR")
    root = Path(configured).expanduser() if configured else _DEFAULT_WORKSPACE_SKILLS_DIR
    return root.resolve()


def _normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("button") or "NOOP").upper()
        try:
            frames = max(1, min(int(step.get("frames", 1)), 120))
        except (TypeError, ValueError):
            frames = 1
        normalized.append(
            {
                "action": action,
                "frames": frames,
                "notes": step.get("notes", ""),
            }
        )
    return normalized


def _session_routines(session: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not session:
        return []
    routines = session.get("routines", [])
    if not isinstance(routines, list):
        return []
    return [routine for routine in routines if isinstance(routine, dict)]


def _memory_summary_payload(session_id: Optional[str], data_dir: Optional[str] = None) -> Dict[str, Any]:
    if not session_id:
        return {
            "total_records": 0,
            "by_type": {},
            "latest_by_type": {},
            "recent_notes": [],
            "learned_control_patterns": [],
            "recent_failure_reflections": [],
        }

    summary = summarize_memory(session_id, data_dir=data_dir)
    memory = get_memory(session_id, limit=100, data_dir=data_dir)
    records = memory.get("records", []) if memory.get("ok") else []

    notes = [record for record in records if record.get("type") == "note"][-3:]
    control_patterns = [
        {
            "sequence": record.get("sequence", []),
            "outcome": record.get("outcome", ""),
            "note": record.get("note", ""),
            "timestamp": record.get("timestamp"),
        }
        for record in records
        if record.get("type") == "control_pattern"
    ][-5:]
    failure_reflections = [
        {
            "trigger": record.get("trigger", ""),
            "error": record.get("error", ""),
            "consequence": record.get("consequence", ""),
            "defense": record.get("defense", ""),
            "severity": record.get("severity", "medium"),
            "source": record.get("source", "agent"),
            "timestamp": record.get("timestamp"),
        }
        for record in records
        if record.get("type") == "failure_reflection"
    ][-5:]

    return {
        "total_records": summary.get("total_records", 0) if summary.get("ok") else 0,
        "by_type": summary.get("by_type", {}) if summary.get("ok") else {},
        "latest_by_type": summary.get("latest_by_type", {}) if summary.get("ok") else {},
        "recent_notes": notes,
        "learned_control_patterns": list(reversed(control_patterns)),
        "recent_failure_reflections": list(reversed(failure_reflections)),
    }


def _guardrails(memory_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    guardrails: List[Dict[str, Any]] = []
    for reflection in memory_summary.get("recent_failure_reflections", []):
        guardrails.append(
            {
                "kind": "failure_reflection",
                "trigger": reflection.get("trigger", ""),
                "error": reflection.get("error", ""),
                "consequence": reflection.get("consequence", ""),
                "defense": reflection.get("defense", ""),
                "severity": reflection.get("severity", "medium"),
                "source": reflection.get("source", "agent"),
                "timestamp": reflection.get("timestamp"),
            }
        )
    return guardrails


def _skill_drafts(memory_summary: Dict[str, Any], routines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drafts: List[Dict[str, Any]] = []
    for pattern in memory_summary.get("learned_control_patterns", []):
        sequence = pattern.get("sequence", [])
        outcome = pattern.get("outcome", "")
        drafts.append(
            {
                "id": f"skill-{_stable_id(sequence, outcome)}",
                "name": outcome or "learned_pattern",
                "source": "memory.control_pattern",
                "status": "draft",
                "sequence": sequence,
                "outcome": outcome,
                "summary": pattern.get("note") or outcome,
            }
        )
    for routine in routines:
        skill_draft = routine.get("skill_draft")
        if isinstance(skill_draft, dict):
            drafts.append(skill_draft)
    return drafts[:6]


def _find_skill_draft(skill_drafts: List[Dict[str, Any]], draft_id: str) -> Optional[Dict[str, Any]]:
    for draft in skill_drafts:
        if draft.get("id") == draft_id:
            return draft
    return None


def _draft_steps(
    draft: Dict[str, Any],
    routines: List[Dict[str, Any]],
    memory_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    draft_name = str(draft.get("name") or "").strip()
    for routine in routines:
        routine_name = str(routine.get("name") or "").strip()
        skill_draft = routine.get("skill_draft") if isinstance(routine.get("skill_draft"), dict) else {}
        if skill_draft.get("id") == draft.get("id") or routine_name == draft_name:
            return _normalize_steps(routine.get("steps"))

    sequence = draft.get("sequence")
    if isinstance(sequence, list) and sequence:
        return _normalize_steps([{"action": action, "frames": 1} for action in sequence])

    for pattern in memory_summary.get("learned_control_patterns", []):
        if str(pattern.get("outcome") or "").strip() == draft_name:
            sequence = pattern.get("sequence") or []
            return _normalize_steps([{"action": action, "frames": 1} for action in sequence])
    return []


def _build_skill_artifact(
    draft: Dict[str, Any],
    *,
    session_id: Optional[str],
    active_routine: Optional[str],
    available_tools: List[Dict[str, Any]],
    memory_summary: Dict[str, Any],
    guardrails: List[Dict[str, Any]],
    routines: List[Dict[str, Any]],
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    skill_name = _slugify(str(draft.get("name") or draft.get("outcome") or draft.get("id") or "generated-skill"))
    relative_install_dir = f"generated/{skill_name}"
    install_path = _workspace_skills_root(workspace_skills_dir) / relative_install_dir / "SKILL.md"
    steps = _draft_steps(draft, routines, memory_summary)
    summary = str(draft.get("summary") or draft.get("outcome") or draft.get("name") or skill_name).strip()
    recent_note = ""
    for note in memory_summary.get("recent_notes", []):
        text = str(note.get("text") or "").strip()
        if text:
            recent_note = text
            break

    frontmatter = {
        "name": skill_name,
        "description": summary or f"Generated gameplay skill for {skill_name}",
        "metadata": {
            "openclaw": {"emoji": "🎮", "os": ["darwin", "linux", "win32"]},
            "source": {
                "session_id": session_id,
                "draft_id": draft.get("id"),
                "active_routine": active_routine,
            },
        },
    }

    tool_names = [tool.get("name") for tool in available_tools[:5] if isinstance(tool, dict) and tool.get("name")]
    lines = [
        "---",
        f"name: {frontmatter['name']}",
        f"description: {frontmatter['description']}",
        f"metadata: {frontmatter['metadata']}",
        "---",
        "",
        f"# {draft.get('name') or skill_name}",
        "",
        "## Purpose",
        summary or "Generated from learned gameplay patterns.",
        "",
        "## When To Use",
        f"- Use when the active session is working toward: {summary or skill_name}",
    ]
    if active_routine:
        lines.append(f"- Prefer this when the active routine is `{active_routine}` or a close match.")
    if recent_note:
        lines.extend(["", "## Session Note", recent_note])
    lines.extend(["", "## Recommended Tools"])
    if tool_names:
        lines.extend([f"- `{tool_name}`" for tool_name in tool_names])
    else:
        lines.append("- `get_agent_context`")

    lines.extend(["", "## Suggested Steps"])
    if steps:
        for index, step in enumerate(steps, start=1):
            action = step.get("action") or "NOOP"
            frames = step.get("frames") or 1
            note = str(step.get("notes") or "").strip()
            suffix = f" - {note}" if note else ""
            lines.append(f"{index}. Press `{action}` for `{frames}` frame(s){suffix}")
    else:
        lines.append("1. Refresh `get_agent_context` before acting.")

    if guardrails:
        lines.extend(["", "## Guardrails"])
        for guardrail in guardrails[:3]:
            defense = str(guardrail.get("defense") or "").strip()
            if defense:
                lines.append(f"- {defense}")

    lines.extend(
        [
            "",
            "## Operator Notes",
            "- This draft is generated from session memory and routines.",
            "- Keep PyBoy actions on canonical backend routes such as `act_and_observe`.",
            "- Re-check `get_agent_context` after any failed or ambiguous step.",
            "",
        ]
    )

    content = "\n".join(lines)
    return {
        "frontmatter": frontmatter,
        "content": content,
        "relative_install_dir": relative_install_dir,
        "install_path": str(install_path),
        "installed": install_path.exists(),
    }


def _skill_workshop_entry(
    draft: Dict[str, Any],
    *,
    snapshot: Dict[str, Any],
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    artifact = _build_skill_artifact(
        draft,
        session_id=snapshot.get("active_session_id"),
        active_routine=snapshot.get("active_routine"),
        available_tools=snapshot.get("available_tools", []),
        memory_summary=snapshot.get("memory_summary", {}),
        guardrails=snapshot.get("guardrails", []),
        routines=snapshot.get("routines", []),
        workspace_skills_dir=workspace_skills_dir,
    )
    return {
        **draft,
        "artifact": artifact,
        "preview_markdown": artifact["content"],
        "preview_excerpt": "\n".join(artifact["content"].splitlines()[:8]),
        "installed": artifact["installed"],
    }


def _suggested_routines(memory_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for pattern in memory_summary.get("learned_control_patterns", []):
        sequence = pattern.get("sequence", [])
        outcome = pattern.get("outcome", "")
        if not sequence or not outcome:
            continue
        suggestion_id = f"learned-{_stable_id(sequence, outcome)}"
        suggestions.append(
            {
                "id": suggestion_id,
                "name": outcome.lower().replace(" ", "_")[:48] or suggestion_id,
                "kind": "generated_playbook",
                "origin": "memory",
                "status": "suggested",
                "steps": [{"action": str(action).upper(), "frames": 1} for action in sequence],
                "summary": outcome,
            }
        )
    return suggestions[:5]


def _next_recommended_action(
    agent_state: Dict[str, Any],
    game_context: Dict[str, Any],
    active_routine: Optional[str],
) -> Dict[str, Any]:
    recommendations = game_context.get("recommendations") or []
    health_summary = game_context.get("health_summary") or {}
    battle = game_context.get("battle") or {}
    current_task = agent_state.get("current_task") or ""
    current_goal = agent_state.get("current_goal") or ""

    if battle.get("in_battle"):
        return {
            "action": "BATTLE",
            "reason": "Battle is active; use the canonical action path for tactical turns.",
            "source": "context.battle",
        }
    if health_summary.get("needs_healing"):
        return {
            "action": "HEAL",
            "reason": recommendations[0] if recommendations else "Party health is low; route to healing before exploration.",
            "source": "context.health_summary",
        }
    if active_routine:
        return {
            "action": "FOLLOW_ROUTINE",
            "target": active_routine,
            "reason": f"Continue the active routine: {active_routine}.",
            "source": "session.active_routine",
        }
    if current_task:
        return {
            "action": "ADVANCE_TASK",
            "target": current_task,
            "reason": f"Advance the current task: {current_task}.",
            "source": "agent_state.current_task",
        }
    if current_goal:
        return {
            "action": "PLAN",
            "target": current_goal,
            "reason": f"Break the current goal into a short sequence: {current_goal}.",
            "source": "agent_state.current_goal",
        }
    return {
        "action": "OBSERVE",
        "reason": "No goal, task, or urgent state detected; refresh context before acting.",
        "source": "fallback",
    }


def build_capability_snapshot(
    *,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    state = dict(agent_state or {})
    context = dict(game_context or {})
    session = None
    if session_id:
        session = get_session(session_id, data_dir=data_dir)
    if session is None:
        session = get_current_session(data_dir=data_dir)

    active_session_id = session.get("id") if session else None
    routines = _session_routines(session)
    active_routine = state.get("active_routine") or (session or {}).get("active_routine")
    memory_summary = _memory_summary_payload(active_session_id, data_dir=data_dir)
    guardrails = _guardrails(memory_summary)
    suggested_routines = _suggested_routines(memory_summary)
    next_action = _next_recommended_action(state, context, active_routine)

    return {
        "active_session_id": active_session_id,
        "active_routine": active_routine,
        "available_tools": list(_TOOLBELT),
        "tool_groups": dict(_TOOL_GROUPS),
        "memory_summary": memory_summary,
        "guardrails": guardrails,
        "next_recommended_action": next_action,
        "planner_hint": next_action,
        "routines": routines,
        "suggested_routines": suggested_routines,
        "skill_drafts": _skill_drafts(memory_summary, routines),
        "skill_workshop": {
            "active_session_id": active_session_id,
            "workspace_precedence": "repo-local",
            "workspace_skills_root": str(_workspace_skills_root(workspace_skills_dir)),
            "install_route": "/api/agent/skills/workshop/install",
            "draft_count": len(_skill_drafts(memory_summary, routines)),
            "drafts": [
                _skill_workshop_entry(
                    draft,
                    snapshot={
                        "active_session_id": active_session_id,
                        "active_routine": active_routine,
                        "available_tools": list(_TOOLBELT),
                        "memory_summary": memory_summary,
                        "guardrails": guardrails,
                        "routines": routines,
                    },
                    workspace_skills_dir=workspace_skills_dir,
                )
                for draft in _skill_drafts(memory_summary, routines)
            ],
        },
        "auto_learning_signals": {
            "control_patterns_observed": len(memory_summary.get("learned_control_patterns", [])),
            "failure_reflection_count": len(memory_summary.get("recent_failure_reflections", [])),
            "guardrail_count": len(guardrails),
            "suggested_routine_count": len(suggested_routines),
            "skill_draft_count": len(_skill_drafts(memory_summary, routines)),
        },
        "timestamp": _now_iso(),
    }


def get_toolbelt_snapshot(
    *,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = build_capability_snapshot(
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    return {
        "active_session_id": snapshot["active_session_id"],
        "active_routine": snapshot["active_routine"],
        "available_tools": snapshot["available_tools"],
        "tool_groups": snapshot["tool_groups"],
        "memory_summary": snapshot["memory_summary"],
        "guardrails": snapshot["guardrails"],
        "next_recommended_action": snapshot["next_recommended_action"],
        "planner_hint": snapshot["planner_hint"],
        "auto_learning_signals": snapshot["auto_learning_signals"],
        "timestamp": snapshot["timestamp"],
    }


def get_routines_snapshot(
    *,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = build_capability_snapshot(
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    return {
        "active_session_id": snapshot["active_session_id"],
        "active_routine": snapshot["active_routine"],
        "routines": snapshot["routines"],
        "suggested_routines": snapshot["suggested_routines"],
        "skill_drafts": snapshot["skill_drafts"],
        "timestamp": snapshot["timestamp"],
    }


def get_guardrails_snapshot(
    *,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = build_capability_snapshot(
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    return {
        "active_session_id": snapshot["active_session_id"],
        "active_routine": snapshot["active_routine"],
        "guardrails": snapshot["guardrails"],
        "recent_failure_reflections": snapshot["memory_summary"]["recent_failure_reflections"],
        "timestamp": snapshot["timestamp"],
    }


def get_skill_workshop_snapshot(
    *,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = build_capability_snapshot(
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    workshop = snapshot["skill_workshop"]
    return {
        "active_session_id": snapshot["active_session_id"],
        "active_routine": snapshot["active_routine"],
        "workspace_precedence": workshop["workspace_precedence"],
        "workspace_skills_root": workshop["workspace_skills_root"],
        "install_route": workshop["install_route"],
        "draft_count": workshop["draft_count"],
        "drafts": workshop["drafts"],
        "timestamp": snapshot["timestamp"],
    }


def get_skill_workshop_draft(
    *,
    draft_id: str,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
) -> Dict[str, Any]:
    snapshot = build_capability_snapshot(
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    draft = _find_skill_draft(snapshot["skill_workshop"]["drafts"], draft_id)
    if not draft:
        return {
            "ok": False,
            "error": f"Skill draft {draft_id!r} not found",
            "timestamp": snapshot["timestamp"],
        }
    return {
        "ok": True,
        "active_session_id": snapshot["active_session_id"],
        "active_routine": snapshot["active_routine"],
        "draft": {key: value for key, value in draft.items() if key not in {"artifact", "preview_markdown", "preview_excerpt", "installed"}},
        "artifact": draft["artifact"],
        "preview_markdown": draft["preview_markdown"],
        "preview_excerpt": draft["preview_excerpt"],
        "timestamp": snapshot["timestamp"],
    }


def install_skill_workshop_draft(
    *,
    draft_id: str,
    agent_state: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    data_dir: Optional[str] = None,
    workspace_skills_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    detail = get_skill_workshop_draft(
        draft_id=draft_id,
        agent_state=agent_state,
        game_context=game_context,
        session_id=session_id,
        data_dir=data_dir,
        workspace_skills_dir=workspace_skills_dir,
    )
    if not detail.get("ok"):
        return detail

    artifact = dict(detail["artifact"])
    install_path = Path(str(artifact["install_path"]))
    if install_path.exists() and not overwrite:
        return {
            "ok": False,
            "error": f"Skill artifact already exists at {install_path}",
            "artifact": artifact,
            "timestamp": _now_iso(),
        }

    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(str(artifact["content"]), encoding="utf-8")
    artifact["installed"] = True
    return {
        "ok": True,
        "installed": True,
        "draft": detail["draft"],
        "artifact": artifact,
        "timestamp": _now_iso(),
    }


def upsert_session_routine(
    *,
    session_id: str,
    payload: Optional[Dict[str, Any]],
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    session = get_session(session_id, data_dir=data_dir)
    if not session:
        return {"ok": False, "error": f"Session {session_id!r} not found", "timestamp": _now_iso()}

    body = dict(payload or {})
    name = str(body.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "Routine name is required", "timestamp": _now_iso()}

    steps = _normalize_steps(body.get("steps"))
    if not steps:
        return {"ok": False, "error": "Routine steps are required", "timestamp": _now_iso()}

    routine_id = str(body.get("id") or _stable_id(session_id, name))
    routine = {
        "id": routine_id,
        "name": name,
        "description": str(body.get("description") or ""),
        "kind": "playbook",
        "origin": body.get("origin") or "operator",
        "status": body.get("status") or "ready",
        "tags": list(body.get("tags") or []),
        "steps": steps,
        "updated_at": _now_iso(),
        "skill_draft": {
            "id": f"skill-{routine_id}",
            "name": name,
            "source": "routine.upsert",
            "status": "draft",
            "summary": str(body.get("description") or name),
        },
    }

    routines = _session_routines(session)
    replaced = False
    for index, existing in enumerate(routines):
        if existing.get("id") == routine_id or existing.get("name") == name:
            routines[index] = {**existing, **routine}
            replaced = True
            break
    if not replaced:
        routines.append(routine)

    patch: Dict[str, Any] = {"routines": routines}
    if body.get("activate", True):
        patch["active_routine"] = routine["name"]
    update_session(session_id, patch, data_dir=data_dir)

    return {
        "ok": True,
        "session_id": session_id,
        "routine": routine,
        "active_routine": patch.get("active_routine"),
        "timestamp": _now_iso(),
    }
