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

from typing import Any, Callable, Dict, Optional

from . import (
    config,
    save_load,
    ui,
    ws,
    tetris,
    vision,
)

__all__ = [
    "config",
    "save_load",
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
        "save_load",
        lambda: save_load.register_save_load_routes(
            app,
            emulators_getter=emulators_getter,
            game_state_getter=game_state_getter,
            saved_states=saved_states if saved_states is not None else {},
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
