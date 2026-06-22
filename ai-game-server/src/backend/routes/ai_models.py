"""
routes/ai_models — AI provider / model discovery endpoints.

Extracted from server.py. Exposes:

  GET /api/providers              — all providers with models (settings UI)
  GET /api/providers/status       — live status of all AI providers
  GET /api/models                 — models for one provider (query: provider=...)
  GET /api/openclaw/models        — OpenClaw-native model listing
  GET /api/openclaw/models/vision — vision-capable models only
  GET /api/openclaw/models/planning — planning/decision models
  GET /api/openclaw/models/recommend — recommended model for a use case

All endpoints degrade gracefully when the AI provider manager or OpenClaw
discovery service is unavailable — they return an empty list / 503 instead
of crashing, mirroring the legacy server.py semantics.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from flask import jsonify, request

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pure helpers (no globals required)
# ----------------------------------------------------------------------


def _estimate_context_window(model_id: str) -> int:
    """Estimate context window based on model name patterns."""
    model_lower = model_id.lower()
    if "128k" in model_lower:
        return 128000
    if "32k" in model_lower:
        return 32000
    if "8k" in model_lower:
        return 8192
    if "4b" in model_lower:
        return 4096
    if any(x in model_lower for x in ["27b", "35b", "70b", "max"]):
        return 32000
    if any(x in model_lower for x in ["8b", "7b"]):
        return 8192
    return 4096


def _get_model_description(model_id: str, provider_name: str) -> str:
    """Get human-readable description for a model."""
    model_lower = model_id.lower()
    if "vl" in model_lower or "vision" in model_lower:
        return "Vision model for screen analysis and image understanding"
    if "think" in model_lower:
        return "Enhanced reasoning model for complex decisions"
    if "qwen" in model_lower:
        return "Qwen language model for text generation"
    if "glm" in model_lower:
        return "GLM model for fast text generation"
    if "kimi" in model_lower:
        return "Kimi model for vision and reasoning tasks"
    if "llava" in model_lower:
        return "LLaVA vision-language model"
    if "gpt-4" in model_lower:
        return "GPT-4 model for advanced reasoning"
    if "gemini" in model_lower:
        return "Gemini multimodal model"
    return f"AI model: {model_id}"


def _get_provider_display_name(provider_id: str) -> str:
    """Get human-readable provider name."""
    names = {
        "lmstudio": "LM Studio (Local)",
        "openclaw": "OpenClaw Gateway",
        "gemini": "Google Gemini",
        "openrouter": "OpenRouter",
        "openai-compatible": "OpenAI Compatible",
        "nvidia": "NVIDIA NIM",
        "ollama": "Ollama (Local)",
        "mock": "Mock Provider (Testing)",
        "tetris-genetic": "Tetris Genetic AI",
    }
    return names.get(provider_id, provider_id.title())


def _is_provider_manual_allowed(provider_id: str) -> bool:
    """All providers allow manual model entry."""
    return True


def _get_default_model_for_provider(provider_name: str, models: list) -> Optional[str]:
    """Get the default model ID for a provider."""
    for model in models:
        if model.get("is_default"):
            return model["id"]
    env_defaults = {
        "lmstudio": os.environ.get("LM_STUDIO_THINKING_MODEL"),
        "openclaw": os.environ.get("OPENCLAW_MODEL"),
        "gemini": os.environ.get("GEMINI_MODEL"),
        "openrouter": os.environ.get("OPENROUTER_MODEL"),
        "nvidia": os.environ.get("NVIDIA_MODEL"),
        "openai-compatible": os.environ.get("OPENAI_MODEL"),
    }
    default = env_defaults.get(provider_name)
    if default:
        return default
    if models:
        return models[0]["id"]
    return None


def _get_synthetic_priority_providers() -> list:
    """Curated provider entries that should appear even if live discovery is incomplete."""
    return [
        {
            "id": "bailian",
            "name": "Alibaba Bailian",
            "status": "available",
            "available": True,
            "manual_allowed": True,
            "priority": 0,
            "error": None,
            "default_model": "bailian/qwen3.5-plus",
            "models": [
                {"id": "bailian/qwen3.5-plus", "name": "Qwen3.5 Plus", "label": "Qwen3.5 Plus \u2b50", "provider": "bailian", "category": "reasoning", "role": "primary", "capabilities": ["text", "reasoning"], "is_vision_capable": False, "is_free": False, "manual_allowed": True, "is_default": True, "context_window": 1000000, "description": "Primary default model"},
                {"id": "bailian/kimi-k2.5", "name": "Kimi K2.5", "label": "Kimi K2.5 \U0001f441\ufe0f", "provider": "bailian", "category": "vision", "role": "vision", "capabilities": ["text", "vision", "reasoning"], "is_vision_capable": True, "is_free": True, "manual_allowed": True, "is_default": False, "context_window": 196608, "description": "Default vision model"},
                {"id": "bailian/MiniMax-M2.5", "name": "MiniMax M2.5", "label": "MiniMax M2.5 \U0001f9e0", "provider": "bailian", "category": "reasoning", "role": "planning", "capabilities": ["text", "reasoning"], "is_vision_capable": False, "is_free": True, "manual_allowed": True, "is_default": False, "context_window": 196608, "description": "Default planning model"},
                {"id": "bailian/glm-5", "name": "GLM-5", "label": "GLM-5", "provider": "bailian", "category": "reasoning", "role": "general", "capabilities": ["text", "reasoning"], "is_vision_capable": False, "is_free": False, "manual_allowed": True, "is_default": False, "context_window": 128000, "description": "Fast coding / reasoning"},
            ],
        },
        {
            "id": "minimax",
            "name": "MiniMax",
            "status": "available",
            "available": True,
            "manual_allowed": True,
            "priority": 1,
            "error": None,
            "default_model": "bailian/MiniMax-M2.5",
            "models": [
                {"id": "bailian/MiniMax-M2.5", "name": "MiniMax M2.5", "label": "MiniMax M2.5", "provider": "minimax", "category": "reasoning", "role": "planning", "capabilities": ["text", "reasoning"], "is_vision_capable": False, "is_free": True, "manual_allowed": True, "is_default": True, "context_window": 196608, "description": "MiniMax via Bailian"},
            ],
        },
        {
            "id": "moonshot",
            "name": "Moonshot / Kimi",
            "status": "available",
            "available": True,
            "manual_allowed": True,
            "priority": 2,
            "error": None,
            "default_model": "bailian/kimi-k2.5",
            "models": [
                {"id": "bailian/kimi-k2.5", "name": "Kimi K2.5", "label": "Kimi K2.5", "provider": "moonshot", "category": "vision", "role": "vision", "capabilities": ["text", "vision", "reasoning"], "is_vision_capable": True, "is_free": True, "manual_allowed": True, "is_default": True, "context_window": 196608, "description": "Kimi via Bailian"},
            ],
        },
    ]


def _enrich_model_info(provider_name: str, model_id: str, index: int) -> dict:
    """Enrich model info with metadata for UI consumption."""
    model_lower = model_id.lower()
    is_vision = any(
        p in model_lower for p in ["vl", "vision", "kimi", "llava", "gpt-4v", "gpt-4o"]
    )
    is_thinking = any(p in model_lower for p in ["think", "reason", "qwen", "glm"])
    capabilities = []
    if is_vision:
        capabilities.append("vision")
    if is_thinking or not is_vision:
        capabilities.append("reasoning")
    capabilities.append("text")
    if is_vision:
        category = "vision"
    elif is_thinking:
        category = "reasoning"
    else:
        category = "general"
    name_parts = model_id.split("/")[-1] if "/" in model_id else model_id
    display_name = name_parts.replace("-", " ").replace("_", " ").title()
    is_free = provider_name in ["lmstudio", "ollama", "openclaw", "mock"]
    is_default = False
    if provider_name == "lmstudio":
        default_thinking = os.environ.get("LM_STUDIO_THINKING_MODEL", "")
        default_vision = os.environ.get("LM_STUDIO_VISION_MODEL", "")
        is_default = model_id in [default_thinking, default_vision]
    elif provider_name == "openclaw":
        is_default = index == 0
    return {
        "id": model_id,
        "name": display_name,
        "label": display_name,
        "provider": provider_name,
        "category": category,
        "capabilities": capabilities,
        "is_vision_capable": is_vision,
        "is_free": is_free,
        "manual_allowed": True,
        "is_default": is_default,
        "context_window": _estimate_context_window(model_id),
        "description": _get_model_description(model_id, provider_name),
    }


def _get_recommendation_reason(use_case: str, model) -> str:
    """Get human-readable reason for the recommendation."""
    reasons = {
        "vision": f"Best vision model available ({model.name})",
        "planning": f"Best reasoning model for game decisions ({model.name})",
        "fast": f"Fastest response time ({model.name})",
        "quality": f"Highest quality model available ({model.name})",
        "free": f"Best free model available ({model.name})",
    }
    return reasons.get(use_case, f"Recommended model: {model.name}")


def _get_models_for_provider(provider_name: str, ai_provider_manager: Any) -> list:
    """Enriched model list for a provider — exported for other blueprints."""
    if ai_provider_manager is None:
        return []
    try:
        raw = ai_provider_manager.get_models(provider_name) or []
    except Exception as e:  # noqa: BLE001
        logger.warning("get_models(%s) failed: %s", provider_name, e)
        return []
    return [_enrich_model_info(provider_name, mid, i) for i, mid in enumerate(raw)]


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_ai_models_routes(
    app,
    *,
    ai_provider_manager: Any = None,
    get_model_discovery: Optional[Callable[..., Any]] = None,
    openclaw_endpoint_getter: Callable[[], str] = lambda: os.environ.get(
        "OPENCLAW_ENDPOINT", "http://localhost:18789"
    ),
) -> None:
    """Register AI provider / model discovery endpoints.

    ``ai_provider_manager`` exposes:
      - ``get_provider_status()`` -> dict
      - ``get_models(provider_id: str)`` -> list[str]

    ``get_model_discovery`` is a callable that accepts an endpoint URL
    and returns a discovery instance with methods:
      - ``get_runtime_config()``
      - ``get_available_models(force_refresh: bool)``
      - ``get_vision_models()``
      - ``get_planning_models()``
      - ``recommend_model(use_case: str)``
      - ``_is_cache_valid()``

    Both deps are optional — endpoints return 503 if their backing
    service is missing.
    """

    def _require_manager():
        if ai_provider_manager is None:
            return None, (jsonify({"error": "AI provider manager not available"}), 503)
        return ai_provider_manager, None

    def _models_for_provider(provider_name: str) -> list:
        return _get_models_for_provider(provider_name, ai_provider_manager)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/providers/status", methods=["GET"])
    def get_providers_status():
        manager = ai_provider_manager
        if manager is None:
            return jsonify({"error": "AI provider manager not available"}), 503
        try:
            return jsonify(manager.get_provider_status()), 200
        except Exception as e:  # noqa: BLE001
            logger.error("providers/status failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/providers", methods=["GET"])
    def get_providers():
        manager = ai_provider_manager
        if manager is None:
            return jsonify({"error": "AI provider manager not available"}), 503
        try:
            provider_status = manager.get_provider_status() or {}
            providers_list: List[Dict[str, Any]] = []
            synthetic = _get_synthetic_priority_providers()
            seen_provider_ids = {p["id"] for p in synthetic}
            providers_list.extend(synthetic)
            for provider_id, status_info in provider_status.items():
                if provider_id in seen_provider_ids:
                    continue
                provider_data = {
                    "id": provider_id,
                    "name": _get_provider_display_name(provider_id),
                    "status": status_info.get("status", "unknown"),
                    "available": status_info.get("available", False),
                    "manual_allowed": _is_provider_manual_allowed(provider_id),
                    "priority": status_info.get("priority", 99),
                    "error": status_info.get("error"),
                    "models": [],
                }
                if status_info.get("available"):
                    models = _models_for_provider(provider_id)
                    provider_data["models"] = models
                    provider_data["default_model"] = _get_default_model_for_provider(
                        provider_id, models
                    )
                providers_list.append(provider_data)
            providers_list.sort(key=lambda p: p.get("priority", 99))
            return jsonify(
                {
                    "providers": providers_list,
                    "default_provider": "bailian",
                    "manual_allowed": True,
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("providers failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models", methods=["GET"])
    def get_models():
        manager = ai_provider_manager
        if manager is None:
            return jsonify({"error": "AI provider manager not available"}), 503
        provider_name = request.args.get("provider")
        if not provider_name:
            # Re-use /api/providers payload shape for "no filter"
            return get_providers()
        try:
            provider_status = manager.get_provider_status() or {}
            if provider_name not in provider_status:
                return (
                    jsonify(
                        {
                            "error": f"Provider '{provider_name}' not found",
                            "available_providers": list(provider_status.keys()),
                        }
                    ),
                    404,
                )
            status_info = provider_status[provider_name]
            models = _models_for_provider(provider_name)
            return jsonify(
                {
                    "provider": provider_name,
                    "name": _get_provider_display_name(provider_name),
                    "status": status_info.get("status", "unknown"),
                    "available": status_info.get("available", False),
                    "manual_allowed": _is_provider_manual_allowed(provider_name),
                    "models": models,
                    "default_model": _get_default_model_for_provider(
                        provider_name, models
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("models failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/openclaw/models", methods=["GET"])
    def get_openclaw_models():
        if get_model_discovery is None:
            return jsonify({"error": "OpenClaw model discovery not available"}), 503
        try:
            force_refresh = request.args.get("refresh", "false").lower() == "true"
            discovery = get_model_discovery(openclaw_endpoint_getter())
            runtime_config = discovery.get_runtime_config()
            raw_models = discovery.get_available_models(force_refresh=force_refresh)
            models_enriched = [m.to_dict() for m in raw_models]
            models_enriched.sort(key=lambda m: m.get("priority", 0), reverse=True)
            provider_status = "available" if raw_models else "unavailable"
            default_model = next(
                (m["id"] for m in models_enriched if m.get("is_default")), None
            )
            return jsonify(
                {
                    "provider": "openclaw",
                    "name": "OpenClaw Gateway",
                    "status": provider_status,
                    "available": bool(raw_models),
                    "manual_allowed": True,
                    "models": models_enriched,
                    "defaults": runtime_config.get("defaults", {}),
                    "counts": runtime_config.get("counts", {}),
                    "default_model": default_model,
                    "timestamp": datetime.now().isoformat(),
                    "cached": (not force_refresh) and discovery._is_cache_valid(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("openclaw/models failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/openclaw/models/vision", methods=["GET"])
    def get_vision_models():
        if get_model_discovery is None:
            return jsonify({"error": "OpenClaw model discovery not available"}), 503
        try:
            force_refresh = request.args.get("refresh", "false").lower() == "true"
            discovery = get_model_discovery(openclaw_endpoint_getter())
            raw_models = discovery.get_vision_models()
            models_enriched = [m.to_dict() for m in raw_models]
            default_model = models_enriched[0]["id"] if models_enriched else None
            return jsonify(
                {
                    "provider": "openclaw",
                    "name": "OpenClaw Vision Models",
                    "category": "vision",
                    "models": models_enriched,
                    "default_model": default_model,
                    "counts": {
                        "total": len(models_enriched),
                        "free": len(
                            [m for m in models_enriched if m.get("is_free")]
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("openclaw/models/vision failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/openclaw/models/planning", methods=["GET"])
    def get_planning_models():
        if get_model_discovery is None:
            return jsonify({"error": "OpenClaw model discovery not available"}), 503
        try:
            force_refresh = request.args.get("refresh", "false").lower() == "true"
            discovery = get_model_discovery(openclaw_endpoint_getter())
            raw_models = discovery.get_planning_models()
            models_enriched = [m.to_dict() for m in raw_models]
            default_model = models_enriched[0]["id"] if models_enriched else None
            return jsonify(
                {
                    "provider": "openclaw",
                    "name": "OpenClaw Planning Models",
                    "category": "planning",
                    "models": models_enriched,
                    "default_model": default_model,
                    "counts": {
                        "total": len(models_enriched),
                        "free": len(
                            [m for m in models_enriched if m.get("is_free")]
                        ),
                    },
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("openclaw/models/planning failed: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/openclaw/models/recommend", methods=["GET"])
    def recommend_model():
        if get_model_discovery is None:
            return jsonify({"error": "OpenClaw model discovery not available"}), 503
        try:
            use_case = request.args.get("use_case", "planning")
            force_refresh = request.args.get("refresh", "false").lower() == "true"
            discovery = get_model_discovery(openclaw_endpoint_getter())
            recommended_model = discovery.recommend_model(use_case)
            if recommended_model:
                all_models = discovery.get_available_models()
                alternatives = [
                    m for m in all_models if m.id != recommended_model.id
                ][:3]
                return jsonify(
                    {
                        "recommended": recommended_model.to_dict(),
                        "use_case": use_case,
                        "reason": _get_recommendation_reason(use_case, recommended_model),
                        "alternatives": [
                            {
                                "id": m.id,
                                "name": m.name,
                                "provider": m.provider,
                                "is_free": m.is_free,
                                "is_vision_capable": m.is_vision_capable,
                                "role": m.role,
                            }
                            for m in alternatives
                        ],
                        "timestamp": datetime.now().isoformat(),
                    }
                ), 200
            return (
                jsonify(
                    {
                        "error": "No model found for this use case",
                        "use_case": use_case,
                        "available_use_cases": [
                            "vision",
                            "planning",
                            "fast",
                            "quality",
                            "free",
                        ],
                    }
                ),
                404,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("openclaw/models/recommend failed: %s", e)
            return jsonify({"error": str(e)}), 500