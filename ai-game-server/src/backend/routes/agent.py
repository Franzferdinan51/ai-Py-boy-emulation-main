"""
routes/agent — Agent state, status, goal/chat, errors, actions, context,
mode (GET/POST), act, dialogue, menu, and floating-chat instruction parsing.

Extracted from server.py. Exposes:

  GET              /api/agent/state
  GET              /api/agent/status
  GET/POST         /api/agent/goal
  POST             /api/agent/mode
  GET              /api/agent/mode
  POST             /api/agent/chat             (floating chat + intent parser)
  GET              /api/agent/errors
  GET              /api/agent/actions
  GET              /api/agent/context           (composite snapshot)
  POST             /api/agent/act               (act + observe)
  GET              /api/agent/dialogue
  GET              /api/agent/menu

All endpoints degrade gracefully when no emulator / no ROM is loaded or the
AI provider manager is unavailable. The chat endpoint is fully OpenClaw-aware
and supports the `goal:`, `task:`, and `?status` instruction patterns.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from flask import jsonify, request

from backend.agent_features.run_ledger import (
    GameActionV1,
    GameObservationV1,
    get_run_events,
    record_run_event,
)
from backend.agent_features.agent_capabilities import (
    build_capability_snapshot,
    get_routines_snapshot,
    get_toolbelt_snapshot,
    upsert_session_routine,
)
from backend.agent_features.sessions import get_current_session

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_agent_routes(
    app,
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    agent_state_getter: Callable[[], Dict[str, Any]],
    agent_state_mutate: Optional[Callable[[Dict[str, Any]], None]] = None,
    ai_apis_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    get_action_history: Callable[[], List[Any]] = lambda: [],
) -> None:
    """Register agent state / context / chat endpoints.

    ``agent_state_getter`` returns a mutable reference to the agent-state dict
    so handlers can do field-level mutations (``agent_state_getter()['mode'] =
    'auto'``). ``agent_state_mutate`` is used for atomic updates where we want
    to merge a dict of changes atomically; defaults to in-place dict.update on
    the getter's reference.
    """

    def _state() -> Dict[str, Any]:
        return agent_state_getter() or {}

    def _mutate(updates: Dict[str, Any]) -> None:
        if agent_state_mutate is not None:
            agent_state_mutate(updates)
        else:
            _state().update(updates)

    def _emulators():
        return emulators_getter() or {}

    def _active_session_id() -> Optional[str]:
        session = get_current_session()
        return session.get("id") if session else None

    def _capabilities(game_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return build_capability_snapshot(
            agent_state=_state(),
            game_context=game_context or {},
            session_id=_active_session_id(),
        )

    # ------------------------------------------------------------------
    # Local helpers (were module-level in server.py)
    # ------------------------------------------------------------------

    INTENT_PATTERNS = {
        "goal": ["goal:", "objective:", "my goal is ", "i want to ", "go for "],
        "task": ["task:", "do: ", "please ", "can you "],
        "query": ["?status", "?state", "status", "state:"],
    }

    def parse_instruction(message: str) -> dict:
        """Detect ``goal:`` / ``task:`` / ``?status`` / plain chat."""
        if not message:
            return {"intent": "chat", "value": ""}
        lower = message.lower().strip()
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if lower.startswith(pattern):
                    return {
                        "intent": intent,
                        "value": message[len(pattern):].strip(),
                    }
        return {"intent": "chat", "value": message}

    def format_agent_status_response() -> str:
        state = _state()
        mode = state.get("mode", "manual")
        goal = state.get("current_goal", "(none)")
        task = state.get("current_task", "(none)")
        last_action = state.get("last_action", "None")
        last_time = state.get("last_action_time", "Never")
        total_actions = state.get("stats", {}).get("total_actions", 0)
        response = f"Agent Status:\n"
        response += f"- Mode: {mode}\n"
        response += f"- Goal: {goal}\n"
        response += f"- Task: {task}\n"
        response += f"- Last action: {last_action}"
        if last_time != "Never":
            response += f" ({last_time})"
        response += f"\n- Total actions: {total_actions}"
        return response

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/agent/state", methods=["GET"])
    def api_agent_state():
        """Comprehensive agent state (OpenClaw-style)."""
        state = _state()
        now = datetime.now().isoformat()
        capabilities = _capabilities()
        return jsonify(
            {
                "mode": state.get("mode", "manual"),
                "enabled": state.get("enabled", False),
                "current_goal": state.get("current_goal", ""),
                "current_task": state.get("current_task", ""),
                "last_decision": state.get("last_decision"),
                "last_action": state.get("last_action"),
                "last_action_time": state.get("last_action_time"),
                "recent_errors": state.get("errors", [])[-10:],
                "recent_actions": state.get("actions", [])[-20:],
                "stats": state.get(
                    "stats",
                    {"total_actions": 0, "total_decisions": 0, "total_errors": 0},
                ),
                "started_at": state.get("started_at"),
                "active_session_id": capabilities["active_session_id"],
                "active_routine": capabilities["active_routine"],
                "available_tools": capabilities["available_tools"],
                "memory_summary": capabilities["memory_summary"],
                "next_recommended_action": capabilities["next_recommended_action"],
                "timestamp": now,
            }
        ), 200

    @app.route("/api/agent/status", methods=["GET"])
    def api_agent_status():
        """Concise agent status summary."""
        state = _state()
        errors = state.get("errors", [])
        last_error = errors[-1] if errors else None
        return jsonify(
            {
                "mode": state.get("mode", "manual"),
                "enabled": state.get("enabled", False),
                "goal": state.get("current_goal", ""),
                "last_action": state.get("last_action"),
                "last_error": last_error,
                "action_count": state.get("stats", {}).get("total_actions", 0),
                "error_count": state.get("stats", {}).get("total_errors", 0),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/agent/goal", methods=["POST", "GET"])
    def api_agent_goal():
        """Set or get the current agent goal/task."""
        state = _state()
        if request.method == "GET":
            return jsonify(
                {
                    "goal": state.get("current_goal", ""),
                    "task": state.get("current_task", ""),
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        data = request.get_json(silent=True) or {}
        goal = data.get("goal", "")
        task = data.get("task", "")
        _mutate({"current_goal": goal, "current_task": task})
        logger.info(f"Agent goal set: {goal}, task: {task}")
        return jsonify(
            {
                "ok": True,
                "goal": goal,
                "task": task,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/agent/mode", methods=["POST"])
    def api_agent_mode_set():
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "manual")
        _mutate({"mode": mode})
        return jsonify({"ok": True, "mode": mode}), 200

    @app.route("/api/agent/chat", methods=["POST"])
    def api_agent_chat():
        """Agent-aware chat endpoint that can set goals/tasks via chat."""
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify(
                    {
                        "ok": False,
                        "error": "No ROM loaded",
                        "chat_response": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                ), 400

            data = request.get_json(silent=True) or {}
            user_message = data.get("message", "")

            if not user_message:
                return jsonify(
                    {
                        "ok": False,
                        "error": "Message is required",
                        "chat_response": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                ), 400

            parsed = parse_instruction(user_message)
            intent = parsed["intent"]
            value = parsed["value"]

            goal_updated = None
            task_updated = None
            chat_response = None

            if intent == "goal" and value:
                _mutate({"current_goal": value})
                logger.info(f"Agent goal set via chat: {value}")
                goal_updated = value
                chat_response = (
                    f"I've updated your goal to: {value}. "
                    f"What would you like me to help you with?"
                )
            elif intent == "task" and value:
                _mutate({"current_task": value})
                logger.info(f"Agent task set via chat: {value}")
                task_updated = value
                chat_response = (
                    f"I've set your current task to: {value}. "
                    f"I'll focus on this now."
                )
            elif intent == "query":
                chat_response = format_agent_status_response()
            else:
                # Regular chat - get AI response
                api_name = data.get("api_name", "openclaw")
                api_key = data.get("api_key")
                api_endpoint = data.get("api_endpoint")
                model = data.get("model")

                current_state = game_state_getter()
                emulators = _emulators()
                emulator = emulators.get(current_state["active_emulator"])
                if emulator is None:
                    return jsonify(
                        {
                            "ok": False,
                            "error": "Active emulator not found",
                            "chat_response": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ), 400
                img_bytes = emulator.get_screen_bytes()

                if len(img_bytes) == 0:
                    return jsonify(
                        {
                            "ok": False,
                            "error": "Failed to capture screen",
                            "chat_response": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ), 500

                state = _state()
                context = {
                    "current_goal": state.get("current_goal", ""),
                    "current_task": state.get("current_task", ""),
                    "action_history": get_action_history()[-10:],
                    "game_type": current_state["active_emulator"].upper(),
                }

                try:
                    ai_apis = ai_apis_getter() if ai_apis_getter else {}
                    if api_name and api_name in ai_apis:
                        ai_connector = ai_apis[api_name]
                        chat_response = ai_connector.chat_with_ai(
                            user_message, img_bytes, context
                        )
                    elif api_name == "openclaw" or not api_name:
                        from backend.ai_apis.openclaw_ai_provider import (
                            OpenClawAIProvider,
                        )

                        oc_provider = OpenClawAIProvider(
                            endpoint=api_endpoint
                            or os.environ.get(
                                "OPENCLAW_ENDPOINT", "http://localhost:18789"
                            ),
                            api_key=api_key or os.environ.get("OPENCLAW_API_KEY", ""),
                        )
                        chat_response = oc_provider.chat_with_ai(
                            user_message, img_bytes, context
                        )
                    else:
                        chat_response = (
                            f"Provider '{api_name}' not available. "
                            f"Use: {', '.join(ai_apis.keys())}"
                        )
                except Exception as e:
                    logger.error(f"AI chat error: {e}")
                    chat_response = (
                        f"I couldn't process that request. Error: {str(e)}"
                    )

            state = _state()
            errors = state.get("errors", [])
            last_error = errors[-1] if errors else None
            return jsonify(
                {
                    "ok": True,
                    "intent_detected": intent,
                    "goal_updated": goal_updated,
                    "task_updated": task_updated,
                    "chat_response": chat_response,
                    "agent_state": {
                        "mode": state.get("mode", "manual"),
                        "enabled": state.get("enabled", False),
                        "current_goal": state.get("current_goal", ""),
                        "current_task": state.get("current_task", ""),
                        "last_action": state.get("last_action"),
                        "last_action_time": state.get("last_action_time"),
                        "last_error": last_error,
                        "action_count": state.get("stats", {}).get(
                            "total_actions", 0
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            logger.error(f"Error in agent chat: {e}")
            return jsonify(
                {
                    "ok": False,
                    "error": str(e),
                    "chat_response": None,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 500

    @app.route("/api/agent/errors", methods=["GET"])
    def api_agent_errors():
        try:
            limit = min(int(request.args.get("limit", 10)), 50)
        except (ValueError, TypeError):
            limit = 10
        state = _state()
        errors = state.get("errors", [])[-limit:]
        return jsonify(
            {
                "errors": errors,
                "count": len(errors),
                "total_errors": state.get("stats", {}).get("total_errors", 0),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/agent/actions", methods=["GET"])
    def api_agent_actions():
        try:
            limit = min(int(request.args.get("limit", 20)), 100)
        except (ValueError, TypeError):
            limit = 20
        state = _state()
        actions = state.get("actions", [])[-limit:]
        return jsonify(
            {
                "actions": actions,
                "count": len(actions),
                "total_actions": state.get("stats", {}).get("total_actions", 0),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/agent/runs/events", methods=["GET"])
    def api_agent_runs_events():
        try:
            limit = int(request.args.get("limit", 20))
        except (ValueError, TypeError):
            limit = 20
        session_id = request.args.get("session_id")
        current_state = game_state_getter() or {}
        payload = get_run_events(limit=limit, session_id=session_id)
        payload["loaded"] = bool(current_state.get("rom_loaded"))
        payload["active_emulator"] = current_state.get("active_emulator")
        payload["rom_loaded"] = bool(current_state.get("rom_loaded"))
        return jsonify(payload), 200

    @app.route("/api/agent/toolbelt", methods=["GET"])
    def api_agent_toolbelt():
        payload = get_toolbelt_snapshot(
            agent_state=_state(),
            game_context={"loaded": bool((game_state_getter() or {}).get("rom_loaded"))},
            session_id=_active_session_id(),
        )
        return jsonify(payload), 200

    @app.route("/api/agent/routines", methods=["GET"])
    def api_agent_routines():
        payload = get_routines_snapshot(
            agent_state=_state(),
            game_context={"loaded": bool((game_state_getter() or {}).get("rom_loaded"))},
            session_id=_active_session_id(),
        )
        return jsonify(payload), 200

    @app.route("/api/agent/routines", methods=["POST"])
    def api_agent_routines_upsert():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or _active_session_id()
        if not session_id:
            return jsonify(
                {
                    "ok": False,
                    "error": "No active session. Create or activate a session before saving routines.",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 400
        result = upsert_session_routine(session_id=session_id, payload=payload)
        status = 200 if result.get("ok") else 400
        return jsonify(result), status

    @app.route("/api/agent/context", methods=["GET"])
    def api_agent_context():
        """Composite snapshot: position + party + inventory + battle + health."""
        current_state = game_state_getter()
        active = current_state.get("active_emulator")

        empty_response = {
            "loaded": False,
            "rom_name": None,
            "frame": 0,
            "game_mode": "none",
            "position": {"x": 0, "y": 0, "map_id": 0, "map_name": "none"},
            "party": {"count": 0, "pokemon": []},
            "inventory": {"money": 0, "items": []},
            "battle": {"in_battle": False, "enemy": None},
            "health_summary": {
                "party_healthy": True,
                "lowest_hp_percent": 100,
                "needs_healing": False,
            },
            "recommendations": [],
            "active_session_id": None,
            "active_routine": None,
            "available_tools": [],
            "memory_summary": {
                "total_records": 0,
                "by_type": {},
                "latest_by_type": {},
                "recent_notes": [],
                "learned_control_patterns": [],
            },
            "next_recommended_action": {
                "action": "OBSERVE",
                "reason": "No ROM loaded",
                "source": "fallback",
            },
            "timestamp": datetime.now().isoformat(),
        }

        emulators = _emulators()
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            empty_response.update(_capabilities(empty_response))
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]

            position = {"x": 0, "y": 0, "map_id": 0, "map_name": "unknown"}
            party = {"count": 0, "pokemon": []}
            inventory = {"money": 0, "items": []}
            battle = {"in_battle": False, "enemy": None}

            if hasattr(emulator, "get_position"):
                position = emulator.get_position()
            if hasattr(emulator, "get_party_info"):
                pokemon_list = emulator.get_party_info() or []
                party = {"count": len(pokemon_list), "pokemon": pokemon_list}
            if hasattr(emulator, "get_inventory_info"):
                inv = emulator.get_inventory_info() or {}
                inventory = {
                    "money": inv.get("money", 0),
                    "items": inv.get("items", []),
                }
            if hasattr(emulator, "get_battle_info"):
                battle = emulator.get_battle_info()

            game_mode = "exploration"
            if battle.get("in_battle"):
                game_mode = "battle"
            elif position.get("map_id", 255) == 255:
                game_mode = "title"

            lowest_hp = 100
            needs_healing = False
            for mon in party.get("pokemon", []):
                hp_pct = mon.get("hp_percent", 100)
                if hp_pct < lowest_hp:
                    lowest_hp = hp_pct
                if hp_pct < 30:
                    needs_healing = True

            recommendations = []
            if needs_healing:
                recommendations.append("Heal party at Pokemon Center")
            if battle.get("in_battle"):
                enemy = battle.get("enemy", {})
                if enemy.get("hp_percent", 100) < 30 and lowest_hp > 50:
                    recommendations.append("Consider catching this Pokemon")
                else:
                    recommendations.append("Battle in progress")
            if inventory.get("money", 0) > 10000:
                recommendations.append(
                    f'Consider spending money: ¥{inventory["money"]:,}'
                )

            response = {
                "loaded": True,
                "rom_name": current_state.get("rom_name", current_state.get("rom_path", "Unknown")),
                "frame": emulator.get_frame_count() if hasattr(emulator, "get_frame_count") else 0,
                "game_mode": game_mode,
                "position": position,
                "party": party,
                "inventory": inventory,
                "battle": battle,
                "health_summary": {
                    "party_healthy": not needs_healing,
                    "lowest_hp_percent": round(lowest_hp, 1),
                    "needs_healing": needs_healing,
                },
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat(),
            }
            response.update(_capabilities(response))
            return jsonify(response), 200
        except Exception as e:
            logger.debug(f"Error getting agent context: {e}")
            empty_response["error"] = str(e)
            empty_response.update(_capabilities(empty_response))
            return jsonify(empty_response), 200

    @app.route("/api/agent/mode", methods=["GET"])
    def api_get_game_mode():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        emulators = _emulators()

        empty_response = {
            "mode": "none",
            "in_battle": False,
            "in_menu": False,
            "in_dialogue": False,
            "details": {
                "battle_type": "none",
                "menu_type": "none",
                "dialogue_active": False,
            },
            "loaded": False,
            "timestamp": datetime.now().isoformat(),
        }

        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]
            battle_info = {}
            if hasattr(emulator, "get_battle_info"):
                battle_info = emulator.get_battle_info()
            in_battle = battle_info.get("in_battle", False)
            position = {}
            if hasattr(emulator, "get_position"):
                position = emulator.get_position()

            if in_battle:
                mode = "battle"
            elif position.get("map_id", 0) == 255 or (
                position.get("map_id", 0) == 0 and position.get("x", 0) == 0
            ):
                mode = "title"
            else:
                mode = "exploration"

            return jsonify(
                {
                    "mode": mode,
                    "in_battle": in_battle,
                    "in_menu": False,
                    "in_dialogue": False,
                    "details": {
                        "battle_type": battle_info.get("battle_type", "none"),
                        "menu_type": "none",
                        "dialogue_active": False,
                    },
                    "loaded": True,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting game mode: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200

    @app.route("/api/agent/act", methods=["POST"])
    def api_act_and_observe():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        emulators = _emulators()

        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(
                {
                    "success": False,
                    "error": "No ROM loaded",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 400

        try:
            data = request.get_json(silent=True) or {}
            action = data.get("action", "NOOP").upper()
            frames = data.get("frames", 1)

            valid_actions = {
                "UP",
                "DOWN",
                "LEFT",
                "RIGHT",
                "A",
                "B",
                "START",
                "SELECT",
                "NOOP",
            }
            if action not in valid_actions:
                return jsonify(
                    {
                        "success": False,
                        "error": f"Invalid action: {action}",
                        "valid_actions": list(valid_actions),
                        "timestamp": datetime.now().isoformat(),
                    }
                ), 400

            emulator = emulators[active]

            position_before = {}
            battle_before = {}
            if hasattr(emulator, "get_position"):
                position_before = emulator.get_position()
            if hasattr(emulator, "get_battle_info"):
                battle_before = emulator.get_battle_info()

            success = False
            if hasattr(emulator, "step"):
                success = emulator.step(action, frames)

            position_after = {}
            battle_after = {}
            if hasattr(emulator, "get_position"):
                position_after = emulator.get_position()
            if hasattr(emulator, "get_battle_info"):
                battle_after = emulator.get_battle_info()

            position_changed = (
                position_before.get("x") != position_after.get("x")
                or position_before.get("y") != position_after.get("y")
                or position_before.get("map_id") != position_after.get("map_id")
            )
            battle_started = (
                not battle_before.get("in_battle", False)
                and battle_after.get("in_battle", False)
            )
            battle_ended = battle_before.get(
                "in_battle", False
            ) and not battle_after.get("in_battle", False)

            game_mode = "exploration"
            if battle_after.get("in_battle"):
                game_mode = "battle"

            lowest_hp = 100
            needs_healing = False
            if hasattr(emulator, "get_party_info"):
                party = emulator.get_party_info() or []
                for mon in party:
                    hp_pct = mon.get("hp_percent", 100)
                    if hp_pct < lowest_hp:
                        lowest_hp = hp_pct
                    if hp_pct < 30:
                        needs_healing = True

            observation_payload = {
                "game_mode": game_mode,
                "position": position_after,
                "battle": battle_after,
                "health_summary": {
                    "party_healthy": not needs_healing,
                    "lowest_hp_percent": round(lowest_hp, 1),
                    "needs_healing": needs_healing,
                },
            }
            changes_payload = {
                "position_changed": position_changed,
                "battle_started": battle_started,
                "battle_ended": battle_ended,
            }
            response_payload = {
                "success": success,
                "action": action,
                "frames": frames,
                "observation": observation_payload,
                "changes": changes_payload,
                "timestamp": datetime.now().isoformat(),
            }
            if success:
                run_event = record_run_event(
                    source="api.agent.act",
                    action=GameActionV1(
                        source="api.agent.act",
                        action=action,
                        frames=frames,
                        success=success,
                    ),
                    observation=GameObservationV1(
                        source="api.agent.act",
                        observation=observation_payload,
                        action=action,
                    ),
                    changes=changes_payload,
                    success=success,
                )
                response_payload["event"] = run_event["event"]

            return jsonify(response_payload), 200
        except Exception as e:
            logger.error(f"Error in act_and_observe: {e}")
            return jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ), 500

    @app.route("/api/agent/dialogue", methods=["GET"])
    def api_dialogue_state():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        emulators = _emulators()

        empty_response = {
            "active": False,
            "text": None,
            "has_options": False,
            "options": [],
            "selected_option": 0,
            "can_advance": True,
            "loaded": False,
            "timestamp": datetime.now().isoformat(),
        }
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]
            in_battle = False
            if hasattr(emulator, "get_battle_info"):
                battle = emulator.get_battle_info()
                in_battle = battle.get("in_battle", False)
            return jsonify(
                {
                    "active": False,
                    "text": None,
                    "has_options": in_battle,
                    "options": [],
                    "selected_option": 0,
                    "can_advance": True,
                    "loaded": True,
                    "note": "Dialogue detection requires memory scanning - safe defaults returned",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting dialogue state: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200

    @app.route("/api/agent/menu", methods=["GET"])
    def api_menu_state():
        current_state = game_state_getter()
        active = current_state.get("active_emulator")
        emulators = _emulators()

        empty_response = {
            "active": False,
            "type": "none",
            "selection": 0,
            "options": [],
            "can_close": True,
            "loaded": False,
            "timestamp": datetime.now().isoformat(),
        }
        if not current_state.get("rom_loaded") or not active or active not in emulators:
            return jsonify(empty_response), 200

        try:
            emulator = emulators[active]
            in_battle = False
            battle_type = "none"
            if hasattr(emulator, "get_battle_info"):
                battle = emulator.get_battle_info()
                in_battle = battle.get("in_battle", False)
                battle_type = battle.get("battle_type", "none")

            if in_battle:
                return jsonify(
                    {
                        "active": True,
                        "type": "battle",
                        "selection": 0,
                        "options": ["FIGHT", "BAG", "POKEMON", "RUN"],
                        "can_close": False,
                        "loaded": True,
                        "timestamp": datetime.now().isoformat(),
                    }
                ), 200
            return jsonify(
                {
                    "active": False,
                    "type": "none",
                    "selection": 0,
                    "options": [],
                    "can_close": True,
                    "loaded": True,
                    "note": "Menu detection requires memory scanning - safe defaults returned",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:
            logger.debug(f"Error getting menu state: {e}")
            empty_response["error"] = str(e)
            return jsonify(empty_response), 200
