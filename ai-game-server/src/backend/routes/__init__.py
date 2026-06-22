"""
routes — Flask blueprint-style modules extracted from server.py.

Each module exposes a `register_*_routes(app, **deps)` function that mounts its
routes onto the Flask app. The pattern mirrors `backend.agent_features` so
existing main() wiring code can call:

    from backend.routes import register_all

    counts = register_all(
        app,
        emulators_getter=lambda: emulators,
        game_state_getter=get_game_state,
        ai_provider_manager=ai_provider_manager,
        ...
    )

The legacy globals in `backend.server` remain the source of truth; blueprints
read them lazily via the getter callables to avoid circular imports.
"""
from __future__ import annotations

import multiprocessing
import os
import time
from typing import Any, Callable, Dict, List, Optional

from . import (
    agent,
    ai_models,
    ai_runtime,
    config,
    health,
    input,
    rom,
    save_load,
    screen,
    spatial,
    ui,
    ws,
    tetris,
    vision,
)

__all__ = [
    "agent",
    "ai_models",
    "ai_runtime",
    "config",
    "health",
    "input",
    "rom",
    "save_load",
    "screen",
    "spatial",
    "ui",
    "ws",
    "tetris",
    "vision",
    "register_all",
]


def register_all(
    app,
    *,
    emulators_getter: Callable[[], Dict[str, Any]],
    game_state_getter: Callable[[], Dict[str, Any]],
    ai_provider_manager: Any,
    secure_config: Any = None,
    secure_config_available: bool = False,
    host: str = "0.0.0.0",
    port: int = 5002,
    debug: bool = False,
    saved_states: Optional[Dict[str, Any]] = None,
    websocket_runner: Optional[Any] = None,
    # health() deps — all optional with sensible defaults so the blueprint
    # can be wired without breaking the module-load-time registration site.
    health_server_start_time_getter: Optional[Callable[[], float]] = None,
    health_pyboy_available: bool = False,
    health_mcp_available: bool = False,
    health_format_uptime: Optional[Callable[[float], str]] = None,
    health_get_memory_usage: Optional[Callable[[], float]] = None,
    health_component_health: Optional[Dict[str, Any]] = None,
    health_ws_status_getter: Optional[Callable[[], Any]] = None,
    health_get_performance_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    health_agent_state_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    # ai_models() deps — optional; endpoints return 503 if missing.
    ai_models_get_model_discovery: Optional[Callable[..., Any]] = None,
    ai_models_openclaw_endpoint_getter: Optional[Callable[[], str]] = None,
    # ai_runtime() deps
    ai_runtime_state_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    ai_runtime_state_setter: Optional[Callable[[Dict[str, Any]], None]] = None,
    ai_runtime_openclaw_endpoint_setter: Optional[Callable[[str], None]] = None,
    # screen() deps
    screen_numpy_to_base64: Optional[Callable[..., Any]] = None,
    screen_update_performance_metrics: Optional[Callable[[float, float], None]] = None,
    screen_performance_monitor: Optional[Dict[str, Any]] = None,
    screen_get_memory_usage: Optional[Callable[[], float]] = None,
    screen_use_multi_process: bool = False,
    screen_optimization_system_manager: Any = None,
    screen_optimization_system_available: bool = False,
    screen_get_performance_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    # input() deps
    input_update_game_state: Optional[Callable[[Dict[str, Any]], None]] = None,
    input_get_action_history: Optional[Callable[[], list]] = None,
    input_add_to_action_history: Optional[Callable[[Any], None]] = None,
    input_record_agent_action: Optional[Callable[..., None]] = None,
    input_record_agent_error: Optional[Callable[..., None]] = None,
    input_record_agent_decision: Optional[Callable[..., None]] = None,
    input_validate_string_input: Optional[Callable[..., Any]] = None,
    input_validate_integer_input: Optional[Callable[..., Any]] = None,
    input_validate_json_data: Optional[Callable[..., Any]] = None,
    input_timeout_handler: Optional[Callable[[float], Any]] = None,
    input_ai_request_timeout: float = 30.0,
    # agent() deps
    agent_state_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    agent_state_mutate: Optional[Callable[[Dict[str, Any]], None]] = None,
    agent_ai_apis_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    agent_get_action_history: Optional[Callable[[], List[Any]]] = None,
    # rom() deps
    rom_validate_file_upload: Optional[Callable[..., Any]] = None,
    rom_validate_string_input: Optional[Callable[..., Any]] = None,
    rom_sanitize_filename: Optional[Callable[[str], str]] = None,
    rom_max_rom_size: int = 8 * 1024 * 1024,
    rom_allowed_extensions: Optional[List[str]] = None,
    rom_configure_emulator_launch_ui: Optional[Callable[..., Any]] = None,
    rom_sync_loaded_rom_state: Optional[Callable[..., Any]] = None,
    rom_ensure_emulation_loop_running: Optional[Callable[[], Any]] = None,
) -> Dict[str, int]:
    """Register all blueprint routes on the given Flask app.

    Returns a dict of {module_name: routes_registered} for visibility.
    """
    counts: Dict[str, int] = {}

    def _step(name: str, fn: Callable[[], None]) -> None:
        before = len(app.url_map._rules)
        fn()
        counts[name] = len(app.url_map._rules) - before

    _step(
        "config",
        lambda: config.register_config_routes(
            app,
            secure_config=secure_config,
            secure_config_available=secure_config_available,
            host=host,
            port=port,
            debug=debug,
        ),
    )

    _step(
        "health",
        lambda: health.register_health_routes(
            app,
            server_start_time_getter=health_server_start_time_getter
            or (lambda: time.time()),
            pyboy_available=health_pyboy_available,
            mcp_available=health_mcp_available,
            format_uptime=health_format_uptime
            or health._default_format_uptime,
            get_memory_usage=health_get_memory_usage
            or health._default_get_memory_usage,
            component_health=health_component_health,
            ws_status_getter=health_ws_status_getter
            or health._default_ws_status,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            get_performance_stats=health_get_performance_stats
            or health._default_performance_stats,
            agent_state_getter=health_agent_state_getter
            or (lambda: {"enabled": False, "mode": "manual", "errors": []}),
        ),
    )

    _step(
        "ai_models",
        lambda: ai_models.register_ai_models_routes(
            app,
            ai_provider_manager=ai_provider_manager,
            get_model_discovery=ai_models_get_model_discovery,
            openclaw_endpoint_getter=ai_models_openclaw_endpoint_getter
            or (lambda: os.environ.get("OPENCLAW_ENDPOINT", "http://localhost:18789")),
        ),
    )

    _step(
        "ai_runtime",
        lambda: ai_runtime.register_ai_runtime_routes(
            app,
            ai_provider_manager=ai_provider_manager,
            get_model_discovery=ai_models_get_model_discovery,
            ai_runtime_state_getter=ai_runtime_state_getter
            or (lambda: {}),
            ai_runtime_state_setter=ai_runtime_state_setter,
            openclaw_endpoint_getter=ai_models_openclaw_endpoint_getter
            or (lambda: os.environ.get("OPENCLAW_ENDPOINT", "http://localhost:18789")),
            openclaw_endpoint_setter=ai_runtime_openclaw_endpoint_setter,
        ),
    )

    _step(
        "save_load",
        lambda: save_load.register_save_load_routes(
            app,
            emulators_getter=emulators_getter,
            game_state_getter=game_state_getter,
            saved_states=saved_states if saved_states is not None else {},
        ),
    )

    _step(
        "screen",
        lambda: screen.register_screen_routes(
            app,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            numpy_to_base64_image=screen_numpy_to_base64,
            update_performance_metrics=screen_update_performance_metrics,
            performance_monitor=screen_performance_monitor,
            get_performance_stats=(screen_get_performance_stats or (lambda: {})),
            get_memory_usage=screen_get_memory_usage,
            use_multi_process=screen_use_multi_process,
            optimization_system_manager=screen_optimization_system_manager,
            optimization_system_available=screen_optimization_system_available,
            multiprocessing_cpu_count=multiprocessing.cpu_count,
        ),
    )

    _step(
        "input",
        lambda: input.register_input_routes(
            app,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            update_game_state=input_update_game_state or (lambda _u: None),
            get_action_history=input_get_action_history or (lambda: []),
            add_to_action_history=input_add_to_action_history or (lambda _a: None),
            record_agent_action=input_record_agent_action,
            record_agent_error=input_record_agent_error,
            record_agent_decision=input_record_agent_decision,
            validate_string_input=input_validate_string_input,
            validate_integer_input=input_validate_integer_input,
            validate_json_data=input_validate_json_data,
            timeout_handler=input_timeout_handler,
            ai_provider_manager=ai_provider_manager,
            ai_runtime_state_getter=ai_runtime_state_getter or (lambda: {}),
            optimization_system_manager=screen_optimization_system_manager,
            optimization_system_available=screen_optimization_system_available,
            ai_request_timeout=input_ai_request_timeout,
        ),
    )

    _step(
        "agent",
        lambda: agent.register_agent_routes(
            app,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            agent_state_getter=agent_state_getter or (lambda: {}),
            agent_state_mutate=agent_state_mutate,
            ai_apis_getter=agent_ai_apis_getter,
            get_action_history=agent_get_action_history or (lambda: []),
        ),
    )

    _step(
        "spatial",
        lambda: spatial.register_spatial_routes(
            app,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
        ),
    )

    _step(
        "rom",
        lambda: rom.register_rom_routes(
            app,
            game_state_getter=game_state_getter,
            emulators_getter=emulators_getter,
            validate_file_upload=rom_validate_file_upload,
            validate_string_input=rom_validate_string_input,
            sanitize_filename=rom_sanitize_filename,
            max_rom_size=rom_max_rom_size,
            allowed_rom_extensions=rom_allowed_extensions,
            configure_emulator_launch_ui=rom_configure_emulator_launch_ui,
            sync_loaded_rom_state=rom_sync_loaded_rom_state,
            ensure_emulation_loop_running=rom_ensure_emulation_loop_running,
        ),
    )

    _step(
        "ui",
        lambda: ui.register_ui_routes(
            app,
            emulators_getter=emulators_getter,
            game_state_getter=game_state_getter,
        ),
    )

    _step(
        "ws",
        lambda: ws.register_ws_routes(
            app,
            websocket_runner=websocket_runner,
        ),
    )

    _step(
        "tetris",
        lambda: tetris.register_tetris_routes(
            app,
            ai_provider_manager=ai_provider_manager,
        ),
    )

    _step(
        "vision",
        lambda: vision.register_vision_routes(
            app,
            emulators_getter=emulators_getter,
            game_state_getter=game_state_getter,
            ai_provider_manager=ai_provider_manager,
        ),
    )

    return counts
