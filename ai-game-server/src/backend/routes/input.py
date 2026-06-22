"""
routes/input — Emulator input + AI action + chat endpoints.

Extracted from server.py. Exposes:

  POST   /api/game/button    ┐
  POST   /api/game/action    │  shared handler (`execute_action`)
  POST   /api/action         ┘
  POST   /api/ai-action      — ask the configured AI for the next action
  POST   /api/chat           — send a chat message + screen to AI
  POST   /input              — GLM4.5-UI compatibility alias

All endpoints degrade gracefully when no emulator / no ROM is loaded
or when the AI provider manager is unavailable.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import jsonify, request

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_input_routes(
    app,
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    update_game_state: Callable[[Dict[str, Any]], None],
    get_action_history: Callable[[], List[Any]],
    add_to_action_history: Callable[[Any], None],
    record_agent_action: Optional[Callable[..., None]] = None,
    record_agent_error: Optional[Callable[..., None]] = None,
    record_agent_decision: Optional[Callable[..., None]] = None,
    validate_string_input: Optional[Callable[..., str]] = None,
    validate_integer_input: Optional[Callable[..., int]] = None,
    validate_json_data: Optional[Callable[..., dict]] = None,
    timeout_handler: Optional[Callable[[float], Callable]] = None,
    ai_provider_manager: Any = None,
    ai_runtime_state_getter: Callable[[], Dict[str, Any]] = lambda: {},
    optimization_system_manager: Any = None,
    optimization_system_available: bool = False,
    ai_request_timeout: float = 30.0,
) -> None:
    """Register emulator input + AI action endpoints.

    ``validate_*`` and ``record_*`` callbacks are optional. When missing,
    the route falls back to a no-op so the surface stays importable in
    isolation. ``ai_provider_manager`` may be ``None`` — chat / ai-action
    will return 503 in that case.
    """

    def _noop(*_args, **_kwargs):  # pragma: no cover - trivial
        return None

    _record_agent_action = record_agent_action or _noop
    _record_agent_error = record_agent_error or _noop
    _record_agent_decision = record_agent_decision or _noop
    _validate_string = validate_string_input or (lambda v, *a, **kw: str(v))
    _validate_integer = validate_integer_input or (lambda v, *a, **kw: int(v))
    _validate_json = validate_json_data or (lambda v, *a, **kw: v)
    _timeout = timeout_handler or (lambda _s: (lambda fn: fn))

    VALID_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "NOOP"}

    # ------------------------------------------------------------------
    # Shared action executor
    # ------------------------------------------------------------------

    def _execute_action_impl():
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify({"error": "No ROM loaded"}), 400
            try:
                data = request.get_json(force=True)
                if data is None:
                    return jsonify({"error": "Invalid JSON data"}), 400
            except Exception as e:  # noqa: BLE001
                return jsonify({"error": f"JSON parse error: {e}"}), 400
            action = data.get("action", data.get("button", "SELECT"))
            frames = data.get("frames", 1)
            try:
                action = _validate_string(
                    action,
                    "action",
                    min_length=1,
                    max_length=10,
                    allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            if action not in VALID_ACTIONS:
                return (
                    jsonify(
                        {
                            "error": f"Invalid action: {action}",
                            "valid_actions": sorted(VALID_ACTIONS),
                        }
                    ),
                    400,
                )
            try:
                frames = _validate_integer(
                    frames, "frames", min_value=1, max_value=100
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            emulators = emulators_getter() or {}
            emulator = emulators.get(current_state["active_emulator"])
            if emulator is None:
                return jsonify({"error": "Active emulator not found"}), 400
            success = _timeout(5.0)(emulator.step)(action, frames)
            if success:
                add_to_action_history(action)
                _record_agent_action(action, frames, result="success", source="manual")
                return (
                    jsonify(
                        {
                            "message": "Action executed successfully",
                            "action": action,
                            "frames": frames,
                            "history_length": len(get_action_history()),
                        }
                    ),
                    200,
                )
            _record_agent_error(
                "action_error",
                f"Failed to execute action: {action}",
                {"action": action, "frames": frames},
            )
            return jsonify({"error": "Failed to execute action"}), 500
        except Exception as e:  # noqa: BLE001
            logger.error("Error executing action: %s", exc_info=True)
            _record_agent_error(
                "action_error",
                str(e),
                {"action": action if "action" in locals() else "unknown"},
            )
            return jsonify({"error": f"Internal server error: {e}"}), 500

    def _options_response():
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add(
            "Access-Control-Allow-Headers", "Content-Type,Authorization"
        )
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/game/button", methods=["POST", "OPTIONS"])
    @app.route("/api/game/action", methods=["POST", "OPTIONS"])
    @app.route("/api/action", methods=["POST", "OPTIONS"])
    def execute_action():
        if request.method == "OPTIONS":
            return _options_response()
        return _execute_action_impl()

    @app.route("/api/ai-action", methods=["POST", "OPTIONS"])
    def get_ai_action():
        if request.method == "OPTIONS":
            return _options_response()
        if ai_provider_manager is None:
            return jsonify({"error": "AI provider manager not available"}), 503
        request_id = f"ai_action_{time.time_ns()}"
        start_time = time.time()
        logger.info("[%s] AI action request received", request_id)
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify({"error": "No ROM loaded"}), 400
            try:
                data = _validate_json(
                    request.get_data(as_text=True), "AI action request"
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid request data: {e}"}), 400
            api_name = data.get("api_name")
            requested_api_name = api_name
            api_key = data.get("api_key")
            api_endpoint = data.get("api_endpoint")
            model = data.get("model")
            goal = data.get("goal", "")

            # validate API params
            try:
                if api_name is not None:
                    api_name = _validate_string(
                        api_name,
                        "api_name",
                        min_length=1,
                        max_length=50,
                        pattern=r"^[a-zA-Z0-9_-]+$",
                    )
                if api_key is not None:
                    api_key = _validate_string(
                        api_key,
                        "api_key",
                        min_length=1,
                        max_length=500,
                        pattern=r"^[a-zA-Z0-9._-]+$",
                    )
                if api_endpoint is not None:
                    api_endpoint = _validate_string(
                        api_endpoint,
                        "api_endpoint",
                        min_length=1,
                        max_length=500,
                        pattern=r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?$",
                    )
                if model is not None:
                    model = _validate_string(
                        model,
                        "model",
                        min_length=1,
                        max_length=100,
                        pattern=r"^[a-zA-Z0-9._-]+$",
                    )
                goal = _validate_string(
                    goal,
                    "goal",
                    min_length=0,
                    max_length=500,
                    pattern=r"^[a-zA-Z0-9\s\.,\?\!@#\$%\^&\*\(\)_\-\+=\{\}\[\]:;\"'<>\?/~`\|\\]+$",
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid parameter: {e}"}), 400

            runtime = ai_runtime_state_getter() or {}
            api_name = api_name or runtime.get("provider")
            if not model and runtime.get("model"):
                model = runtime["model"]
            if not api_endpoint and runtime.get("api_endpoint"):
                api_endpoint = runtime["api_endpoint"]

            update_game_state(
                {
                    "current_goal": goal,
                    "current_provider": api_name,
                    "current_model": model,
                }
            )
            current_state = game_state_getter()
            emulators = emulators_getter() or {}
            emulator = emulators.get(current_state["active_emulator"])
            if emulator is None:
                return jsonify({"error": "Active emulator not found"}), 400

            try:
                if (
                    optimization_system_available
                    and optimization_system_manager is not None
                    and hasattr(optimization_system_manager, "thread_pool_manager")
                ):
                    def capture_screen():
                        return _timeout(3.0)(emulator.get_screen)()

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(capture_screen)
                        screen_array = future.result(timeout=4.0)
                else:
                    screen_array = _timeout(3.0)(emulator.get_screen)()
            except Exception as screen_error:  # noqa: BLE001
                logger.error("[%s] Timeout getting screen: %s", request_id, screen_error)
                if (
                    optimization_system_available
                    and optimization_system_manager is not None
                    and hasattr(optimization_system_manager, "error_handler")
                ):
                    optimization_system_manager.error_handler.log_error(
                        error_type="screen_capture_timeout",
                        error_message=str(screen_error),
                        context={"request_id": request_id},
                    )
                return jsonify({"error": "Screen capture timeout"}), 500

            if screen_array is None or getattr(screen_array, "size", 0) == 0:
                return jsonify({"error": "Failed to capture screen"}), 500

            cache_hit = False
            screen_hash: Optional[Any] = None
            try:
                if (
                    optimization_system_available
                    and optimization_system_manager is not None
                    and hasattr(optimization_system_manager, "ai_cache_manager")
                ):
                    game_context = {
                        "frame_count": (
                            emulator.get_frame_count()
                            if hasattr(emulator, "get_frame_count")
                            else 0
                        ),
                        "active_emulator": current_state["active_emulator"],
                        "goal": goal,
                        "timestamp": time.time(),
                    }
                    screen_hash = (
                        optimization_system_manager.ai_cache_manager.generate_screen_hash(
                            screen_array.tobytes(), game_context
                        )
                    )
                    cached_response = (
                        optimization_system_manager.ai_cache_manager.get_ai_response(
                            screen_hash, goal, api_name, model
                        )
                    )
                    if cached_response:
                        action = cached_response.response
                        actual_provider = cached_response.provider
                        cache_hit = True
                        total_time = time.time() - start_time
                        add_to_action_history(action)
                        _record_agent_action(action, 1, result="success", source="ai")
                        _record_agent_decision(
                            {"action": action, "goal": goal, "cached": True},
                            provider=actual_provider or api_name,
                        )
                        return (
                            jsonify(
                                {
                                    "action": action,
                                    "provider_used": actual_provider,
                                    "history": get_action_history()[-10:],
                                    "optimization": {
                                        "cache_hit": True,
                                        "response_time_ms": round(total_time * 1000, 2),
                                        "optimization_enabled": True,
                                    },
                                }
                            ),
                            200,
                        )
                img_bytes = _timeout(3.0)(emulator.get_screen_bytes)()
            except Exception as convert_error:  # noqa: BLE001
                logger.error(
                    "[%s] Timeout converting screen: %s",
                    request_id,
                    convert_error,
                )
                return jsonify({"error": "Screen processing timeout"}), 500

            if not img_bytes or len(img_bytes) == 0:
                return jsonify({"error": "Failed to process screen image"}), 500

            if api_key:
                if api_name == "gemini":
                    os.environ["GEMINI_API_KEY"] = api_key
                elif api_name == "openrouter":
                    os.environ["OPENROUTER_API_KEY"] = api_key
                elif api_name == "openai-compatible":
                    os.environ["OPENAI_API_KEY"] = api_key
                    if api_endpoint:
                        os.environ["OPENAI_ENDPOINT"] = api_endpoint
                elif api_name == "nvidia":
                    os.environ["NVIDIA_API_KEY"] = api_key
            if model:
                if api_name == "openai-compatible":
                    os.environ["OPENAI_MODEL"] = model
                elif api_name == "nvidia":
                    os.environ["NVIDIA_MODEL"] = model

            if api_name and api_name not in ai_provider_manager.get_available_providers():
                available_providers = ai_provider_manager.get_available_providers()
                if requested_api_name:
                    return (
                        jsonify(
                            {
                                "error": f"Provider '{api_name}' is not available",
                                "available_providers": available_providers,
                                "suggestion": (
                                    f"Please use one of the available providers: "
                                    f"{', '.join(available_providers)}"
                                ),
                            }
                        ),
                        400,
                    )
                api_name = None

            current_history = get_action_history()
            try:
                if (
                    optimization_system_available
                    and optimization_system_manager is not None
                    and hasattr(optimization_system_manager, "thread_pool_manager")
                ):
                    def call_ai_api():
                        return _timeout(ai_request_timeout)(
                            ai_provider_manager.get_next_action
                        )(img_bytes, goal, current_history, api_name, model)

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(call_ai_api)
                        action, actual_provider = future.result(
                            timeout=ai_request_timeout + 5.0
                        )
                else:
                    action, actual_provider = _timeout(ai_request_timeout)(
                        ai_provider_manager.get_next_action
                    )(img_bytes, goal, current_history, api_name, model)
            except Exception as ai_timeout_error:  # noqa: BLE001
                logger.error(
                    "[%s] AI request timeout: %s", request_id, ai_timeout_error
                )
                return jsonify({"error": "AI request timeout"}), 500

            if not action or not isinstance(action, str):
                logger.error("[%s] Invalid AI response: action=%s", request_id, action)
                return jsonify({"error": "AI returned invalid action"}), 500

            add_to_action_history(action)
            update_game_state(
                {
                    "current_provider": actual_provider or api_name,
                    "current_model": model,
                }
            )
            _record_agent_action(action, 1, result="success", source="ai")
            _record_agent_decision(
                {"action": action, "goal": goal},
                provider=actual_provider or api_name,
            )

            if (
                optimization_system_available
                and optimization_system_manager is not None
                and hasattr(optimization_system_manager, "ai_cache_manager")
                and screen_hash is not None
            ):
                try:
                    optimization_system_manager.ai_cache_manager.cache_ai_response(
                        screen_hash=screen_hash,
                        user_goal=goal,
                        ai_response=action,
                        provider_name=actual_provider or api_name,
                        model_name=model,
                        response_time=time.time() - start_time,
                        confidence_score=0.8,
                    )
                except Exception as cache_error:  # noqa: BLE001
                    logger.debug("[%s] Failed to cache AI response: %s", request_id, cache_error)

            total_time = time.time() - start_time
            response_data: Dict[str, Any] = {
                "action": action,
                "provider_used": actual_provider,
                "history": get_action_history()[-10:],
            }
            if optimization_system_available:
                response_data["optimization"] = {
                    "cache_hit": cache_hit,
                    "response_time_ms": round(total_time * 1000, 2),
                    "memory_pressure": (
                        optimization_system_manager.get_memory_pressure()
                        if optimization_system_manager is not None
                        else None
                    ),
                    "optimization_enabled": True,
                }
            return jsonify(response_data), 200
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Error getting AI action: %s", request_id, e, exc_info=True)
            return jsonify({"error": f"Internal server error: {e}"}), 500

    @app.route("/api/chat", methods=["POST"])
    def ai_chat():
        if ai_provider_manager is None:
            return jsonify({"error": "AI provider manager not available"}), 503
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify({"error": "No ROM loaded"}), 400
            try:
                data = _validate_json(
                    request.get_data(as_text=True), "chat request"
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid chat data: {e}"}), 400
            user_message = data.get("message", "")
            api_name = data.get("api_name")
            api_key = data.get("api_key")
            api_endpoint = data.get("api_endpoint")
            model = data.get("model")
            try:
                user_message = _validate_string(
                    user_message,
                    "message",
                    min_length=1,
                    max_length=2000,
                    pattern=r"^[a-zA-Z0-9\s\.,\?\!@#\$%\^&\*\(\)_\-\+=\{\}\[\]:;\"'<>\?/~`\|\\]+$",
                )
                if api_name is not None:
                    api_name = _validate_string(
                        api_name, "api_name", min_length=1, max_length=50,
                        pattern=r"^[a-zA-Z0-9_-]+$",
                    )
                if api_key is not None:
                    api_key = _validate_string(
                        api_key, "api_key", min_length=1, max_length=500,
                        pattern=r"^[a-zA-Z0-9._-]+$",
                    )
                if api_endpoint is not None:
                    api_endpoint = _validate_string(
                        api_endpoint, "api_endpoint", min_length=1, max_length=500,
                        pattern=r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?$",
                    )
                if model is not None:
                    model = _validate_string(
                        model, "model", min_length=1, max_length=100,
                        pattern=r"^[a-zA-Z0-9._-]+$",
                    )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            emulators = emulators_getter() or {}
            emulator = emulators.get(current_state["active_emulator"])
            if emulator is None:
                return jsonify({"error": "Active emulator not found"}), 400
            try:
                img_bytes = emulator.get_screen_bytes()
            except Exception:  # noqa: BLE001
                img_bytes = b""
            if not img_bytes:
                return jsonify({"error": "Failed to capture screen"}), 500

            context = {
                "current_goal": current_state.get("current_goal"),
                "action_history": get_action_history()[-20:],
                "game_type": (current_state.get("active_emulator") or "").upper(),
            }
            runtime = ai_runtime_state_getter() or {}
            current_provider = current_state.get("current_provider")
            current_model = current_state.get("current_model")
            configured_provider = runtime.get("provider")
            configured_model = runtime.get("model")
            configured_endpoint = runtime.get("api_endpoint")
            api_name = api_name or configured_provider or current_provider
            model = model or configured_model or current_model
            api_endpoint = api_endpoint or configured_endpoint

            if api_key:
                if api_name == "gemini":
                    os.environ["GEMINI_API_KEY"] = api_key
                elif api_name == "openrouter":
                    os.environ["OPENROUTER_API_KEY"] = api_key
                elif api_name == "openai-compatible":
                    os.environ["OPENAI_API_KEY"] = api_key
                    if api_endpoint:
                        os.environ["OPENAI_ENDPOINT"] = api_endpoint
                elif api_name == "nvidia":
                    os.environ["NVIDIA_API_KEY"] = api_key
            if model:
                if api_name == "openai-compatible":
                    os.environ["OPENAI_MODEL"] = model
                elif api_name == "nvidia":
                    os.environ["NVIDIA_MODEL"] = model

            response_text, actual_provider = ai_provider_manager.chat_with_ai(
                user_message, img_bytes, context, api_name, model
            )
            update_game_state(
                {
                    "current_provider": actual_provider or api_name,
                    "current_model": model,
                }
            )
            return jsonify(
                {"response": response_text, "provider_used": actual_provider}
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("Error in AI chat: %s", e, exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/input", methods=["POST"])
    def send_input():
        """GLM4.5-UI compatibility: send a button press to the emulator."""
        try:
            try:
                data = request.get_json(silent=True) or {}
            except Exception:  # noqa: BLE001
                data = {}
            button = (data.get("button", "A") or "A").upper()
            press = bool(data.get("press", True))
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify({"error": "No ROM loaded"}), 400
            emulators = emulators_getter() or {}
            emulator = emulators.get(current_state["active_emulator"])
            if emulator is None:
                return jsonify({"error": "Active emulator not found"}), 400
            action_mapping = {
                "A": "A",
                "B": "B",
                "UP": "UP",
                "DOWN": "DOWN",
                "LEFT": "LEFT",
                "RIGHT": "RIGHT",
                "START": "START",
                "SELECT": "SELECT",
            }
            action = action_mapping.get(button, "NOOP")
            frames = 1 if press else 0
            success = bool(emulator.step(action, frames))
            if success:
                return jsonify({"success": True, "message": f"Input {button} sent"}), 200
            return jsonify({"error": "Failed to send input"}), 500
        except Exception as e:  # noqa: BLE001
            logger.error("Error sending input: %s", e)
            return jsonify({"error": "Failed to send input"}), 500