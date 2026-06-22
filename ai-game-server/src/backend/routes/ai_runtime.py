"""
routes/ai_runtime — AI runtime config + OpenClaw-specific endpoints.

Extracted from server.py. Exposes:

  GET/POST /api/ai/runtime      — current AI provider/model/endpoint selection
  GET/POST /api/openclaw/config — OpenClaw endpoint + dual-model config
  GET     /api/ai/settings     — combined AI settings payload for UI
  GET     /api/openclaw/health — trivial reachability probe

All endpoints degrade gracefully when the AI provider manager or
OpenClaw discovery service is unavailable.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from flask import jsonify, request

from . import ai_models as _ai_models

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_ai_runtime_routes(
    app,
    *,
    ai_provider_manager: Any = None,
    get_model_discovery: Optional[Callable[..., Any]] = None,
    ai_runtime_state_getter: Callable[[], Dict[str, Any]] = lambda: {},
    ai_runtime_state_setter: Optional[Callable[[Dict[str, Any]], None]] = None,
    openclaw_endpoint_getter: Callable[[], str] = lambda: app.config.get(
        "OPENCLAW_ENDPOINT", "http://localhost:18789"
    ),
    openclaw_endpoint_setter: Optional[Callable[[str], None]] = None,
) -> None:
    """Register AI runtime / OpenClaw config endpoints.

    All ``*_getter`` / ``*_setter`` callables are optional and default to
    no-op implementations so the blueprint can be wired without the
    legacy ``server`` module's globals.
    """

    def _state() -> Dict[str, Any]:
        return dict(ai_runtime_state_getter() or {})

    def _set_state(updates: Dict[str, Any]) -> None:
        if ai_runtime_state_setter is None:
            return
        merged = _state()
        merged.update({k: v for k, v in updates.items() if v is not None})
        ai_runtime_state_setter(merged)

    def _endpoint() -> str:
        return openclaw_endpoint_getter() or "http://localhost:18789"

    def _set_endpoint(value: str) -> None:
        if openclaw_endpoint_setter is not None:
            openclaw_endpoint_setter(value)

    def _manager_or_503():
        if ai_provider_manager is None:
            return None, (jsonify({"error": "AI provider manager not available"}), 503)
        return ai_provider_manager, None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/ai/runtime", methods=["GET", "POST"])
    def api_ai_runtime():
        manager, err = _manager_or_503()
        if err is not None:
            return err
        if request.method == "GET":
            try:
                provider_status = manager.get_provider_status()
                available_providers = manager.get_available_providers()
            except Exception as e:  # noqa: BLE001
                logger.error("ai/runtime GET failed: %s", e)
                return jsonify({"error": str(e)}), 500
            return jsonify(
                {
                    "state": _state(),
                    "available_providers": available_providers,
                    "provider_status": provider_status,
                    "default_provider": getattr(manager, "default_provider", None),
                    "manual_allowed": True,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        # POST
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        model = data.get("model")
        api_endpoint = data.get("api_endpoint")
        if ai_runtime_state_setter is not None:
            _set_state(
                {
                    "provider": provider,
                    "model": model,
                    "api_endpoint": api_endpoint,
                }
            )
        return jsonify(
            {
                "ok": True,
                "state": _state(),
                "message": (
                    f"Runtime updated: provider={provider or 'unchanged'}, "
                    f"model={model or 'unchanged'}"
                ),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/openclaw/config", methods=["GET", "POST"])
    def api_openclaw_config():
        manager, err = _manager_or_503()
        if err is not None:
            return err
        if get_model_discovery is None:
            return jsonify({"error": "OpenClaw model discovery not available"}), 503
        if request.method == "GET":
            discovery = get_model_discovery(_endpoint())
            try:
                available = discovery.get_available_models()
            except Exception as e:  # noqa: BLE001
                logger.debug("openclaw config GET discovery error: %s", e)
                available = []
            return jsonify(
                {
                    "endpoint": _endpoint(),
                    "dual_model": {
                        "enabled": getattr(manager, "use_dual_model", False),
                        "vision_model": getattr(manager, "vision_model", None)
                        or "bailian/kimi-k2.5",
                        "planning_model": getattr(manager, "planning_model", None)
                        or "bailian/MiniMax-M2.5",
                    },
                    "status": "available" if available else "unavailable",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        # POST
        data = request.get_json(silent=True) or {}
        result: Dict[str, Any] = {"ok": True, "changes": []}
        if "endpoint" in data:
            endpoint = data["endpoint"]
            _set_endpoint(endpoint)
            try:
                discovery = get_model_discovery(endpoint)
                if hasattr(discovery, "clear_cache"):
                    discovery.clear_cache()
            except Exception as e:  # noqa: BLE001
                logger.debug("openclaw config POST cache clear error: %s", e)
            result["changes"].append(f"endpoint: {endpoint}")
        if (
            "vision_model" in data
            or "planning_model" in data
            or "use_dual_model" in data
        ):
            try:
                cfg = manager.configure_dual_model(
                    vision_model=data.get("vision_model"),
                    planning_model=data.get("planning_model"),
                    use_dual_model=data.get("use_dual_model"),
                )
                if cfg.get("changes"):
                    result["changes"].extend(cfg["changes"])
            except Exception as e:  # noqa: BLE001
                logger.error("configure_dual_model failed: %s", e)
                result["changes"].append(f"configure_dual_model error: {e}")
        result["config"] = {
            "endpoint": _endpoint(),
            "dual_model": {
                "enabled": getattr(manager, "use_dual_model", False),
                "vision_model": getattr(manager, "vision_model", None)
                or "bailian/kimi-k2.5",
                "planning_model": getattr(manager, "planning_model", None)
                or "bailian/MiniMax-M2.5",
            },
        }
        result["timestamp"] = datetime.now().isoformat()
        return jsonify(result), 200

    @app.route("/api/ai/settings", methods=["GET"])
    def api_ai_settings():
        manager, err = _manager_or_503()
        if err is not None:
            return err
        try:
            provider_status = manager.get_provider_status() or {}
            available_providers = manager.get_available_providers()
            providers_list = []
            for provider_id, status_info in provider_status.items():
                provider_data = {
                    "id": provider_id,
                    "name": _ai_models._get_provider_display_name(provider_id),
                    "status": status_info.get("status", "unknown"),
                    "available": status_info.get("available", False),
                    "manual_allowed": _ai_models._is_provider_manual_allowed(provider_id),
                    "priority": status_info.get("priority", 99),
                    "error": status_info.get("error"),
                    "models": [],
                }
                if status_info.get("available"):
                    models = _ai_models._get_models_for_provider(provider_id, manager)
                    provider_data["models"] = models
                    provider_data["default_model"] = (
                        _ai_models._get_default_model_for_provider(provider_id, models)
                    )
                providers_list.append(provider_data)
            providers_list.sort(key=lambda p: p.get("priority", 99))

            openclaw_models: list = []
            if get_model_discovery is not None:
                try:
                    discovery = get_model_discovery(_endpoint())
                    openclaw_models = discovery.get_available_models() or []
                except Exception as e:  # noqa: BLE001
                    logger.debug("ai/settings openclaw discovery error: %s", e)

            state = _state()
            runtime_provider = state.get("provider") or "bailian"
            runtime_model = state.get("model") or "bailian/qwen3.5-plus"
            return jsonify(
                {
                    "runtime": {
                        "provider": runtime_provider,
                        "model": runtime_model,
                        "api_endpoint": _endpoint(),
                    },
                    "providers": providers_list,
                    "default_provider": "bailian",
                    "openclaw": {
                        "endpoint": _endpoint(),
                        "status": "available" if openclaw_models else "unavailable",
                        "models_count": len(openclaw_models),
                    },
                    "dual_model": {
                        "enabled": getattr(manager, "use_dual_model", False),
                        "vision_model": getattr(manager, "vision_model", None)
                        or "bailian/kimi-k2.5",
                        "planning_model": getattr(manager, "planning_model", None)
                        or "bailian/MiniMax-M2.5",
                        "available": getattr(manager, "dual_model_provider", None)
                        is not None,
                    },
                    "available_providers": available_providers,
                    "manual_allowed": True,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("ai/settings failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/openclaw/health", methods=["GET"])
    def api_openclaw_health():
        # Trivial reachability probe. Real liveness should be checked
        # via the discovery service in /api/openclaw/config GET.
        return jsonify({"ok": True, "reachable": True}), 200