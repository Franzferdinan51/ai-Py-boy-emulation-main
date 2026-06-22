"""
routes/vision — Screen vision analysis endpoints.

Extracted from server.py. Five routes:

  - POST /api/vision/analyze  — structured screen analysis (game_state, entities, danger, opportunities)
  - GET|POST /api/vision/describe — quick human-readable description
  - GET /api/vision/ocr       — text extraction
  - GET /api/vision/summary   — fast structured summary
  - GET /api/vision/status    — provider + endpoint metadata

All routes prefer the dual-model provider when configured and fall back to a
single-provider chat with attached image.
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict

from flask import jsonify, request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts (kept verbatim from the original server.py to preserve behaviour)
# ---------------------------------------------------------------------------

_ANALYZE_PROMPT = """Analyze this Game Boy game screen. Provide:

1. **Game State**: What's happening? (exploration, battle, menu, dialog, title screen)
2. **Description**: Brief visual description of what you see
3. **Player Position**: Where is the player on screen?
4. **Nearby Entities**: NPCs, enemies, items, obstacles (list what you can identify)
5. **UI Elements**: Menus, text boxes, health bars, indicators
6. **Danger Level**: Is there immediate danger? (low/medium/high)
7. **Opportunities**: What could the player do next?

Current objective: {goal}

Be concise but specific. Focus on actionable information for gameplay."""

_OCR_PROMPT = """Extract all visible text from this Game Boy screen.

List each distinct text element you can see:
- Dialogue text
- Menu options
- Item names
- Numbers (HP, level, money, etc.)
- Any other readable text

Format your response as:
TEXT_FOUND: [yes/no]
LINES:
- Line 1 text
- Line 2 text
...

If no text is visible, respond with:
TEXT_FOUND: no
LINES: (none)"""

_SUMMARY_PROMPT = """Quick screen analysis. Respond in this EXACT format:

STATE: [exploration/battle/menu/dialog/title]
SAFE_TO_ACT: [yes/no]
URGENCY: [low/medium/high]
RECOMMENDED: [one action: UP/DOWN/LEFT/RIGHT/A/B/WAIT]

Only respond with those 4 lines. No other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vision_model_label(ai_provider_manager) -> str:
    vision = getattr(ai_provider_manager, "vision_model", None)
    return f"vision:{vision}" if vision else "vision:default"


def _dual_model_provider(ai_provider_manager):
    """Return dual-model provider if configured and available, else None."""
    if not getattr(ai_provider_manager, "use_dual_model", False):
        return None
    dmp = getattr(ai_provider_manager, "dual_model_provider", None)
    if dmp is None:
        return None
    return dmp


def _fallback_connector(ai_provider_manager):
    """Return a single-provider connector for fallback."""
    # Prefer a vision-capable one (lmstudio / openai-compatible / gemini) when
    # caller didn't specify; otherwise fall back to default.
    provider_name = os.environ.get("VISION_PROVIDER", "lmstudio")
    if hasattr(ai_provider_manager, "get_provider"):
        try:
            conn = ai_provider_manager.get_provider(provider_name)
            if conn:
                return conn
        except Exception:  # noqa: BLE001
            pass
        try:
            return ai_provider_manager.get_provider()
        except Exception:  # noqa: BLE001
            return None
    return None


def _call_with_image(connector, prompt: str, img_bytes: bytes, context: dict) -> Any:
    """Try vision-capable methods on the connector in order of preference."""
    if hasattr(connector, "analyze_image"):
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return connector.analyze_image(image_b64, prompt)
    if hasattr(connector, "chat_with_image"):
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return connector.chat_with_image(prompt, image_b64, context)
    # Generic: text prompt + raw image bytes
    if hasattr(connector, "chat_with_ai"):
        return connector.chat_with_ai(
            f"[Image attached - screen capture]\n\n{prompt}",
            img_bytes,
            context,
        )
    raise RuntimeError("connector has no vision-capable method")


def _iso_now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_vision_routes(
    app,
    *,
    emulators_getter: Callable[[], Dict[str, Any]],
    game_state_getter: Callable[[], Dict[str, Any]],
    ai_provider_manager: Any,
) -> None:
    """Register vision analysis routes."""

    # -- /api/vision/analyze --------------------------------------------------

    @app.route("/api/vision/analyze", methods=["POST"])
    def api_vision_analyze():
        """Structured screen analysis for AI agents that can't process images."""
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get("active_emulator"):
                return jsonify({
                    "success": False,
                    "error": "No ROM loaded",
                    "analysis": None,
                    "timestamp": _iso_now(),
                }), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]

            screen_array = emulator.get_screen()
            if screen_array is None or getattr(screen_array, "size", 0) == 0:
                return jsonify({
                    "success": False,
                    "error": "Failed to capture screen",
                    "timestamp": _iso_now(),
                }), 500

            img_bytes = emulator.get_screen_bytes()
            if not img_bytes:
                return jsonify({
                    "success": False,
                    "error": "Failed to encode screen",
                    "timestamp": _iso_now(),
                }), 500

            data = request.get_json(silent=True) or {}
            custom_prompt = data.get("prompt")
            context = data.get("context", {}) or {}
            context.setdefault("goal", current_state.get("current_goal", "explore and progress"))
            context.setdefault("game_type", "Game Boy")

            prompt = custom_prompt or _ANALYZE_PROMPT.format(goal=context.get("goal"))

            dmp = _dual_model_provider(ai_provider_manager)
            if dmp is not None:
                analysis = dmp.analyze_screen(img_bytes, context)
                return jsonify({
                    "success": True,
                    "analysis": {
                        "game_state": analysis.game_state,
                        "description": analysis.raw_description,
                        "player_position": analysis.player_position,
                        "nearby_entities": analysis.nearby_entities,
                        "ui_elements": analysis.ui_elements,
                        "danger_level": analysis.danger_level,
                        "opportunities": analysis.opportunities,
                        "raw_response": analysis.raw_description,
                    },
                    "model_used": _vision_model_label(ai_provider_manager),
                    "timestamp": _iso_now(),
                }), 200

            vision_model = os.environ.get("LM_STUDIO_VISION_MODEL", "qwen3-vl-8b")
            provider_name = os.environ.get("VISION_PROVIDER", "lmstudio")
            connector = _fallback_connector(ai_provider_manager)
            if connector is not None:
                result = _call_with_image(connector, prompt, img_bytes, context)
                return jsonify({
                    "success": True,
                    "analysis": {
                        "game_state": "unknown",
                        "description": result if isinstance(result, str) else str(result),
                        "raw_response": result if isinstance(result, str) else str(result),
                    },
                    "model_used": f"{provider_name}:{vision_model}",
                    "timestamp": _iso_now(),
                }), 200

            return jsonify({
                "success": False,
                "error": "No vision-capable AI provider available",
                "hint": "Configure LM_STUDIO_VISION_MODEL or enable dual-model mode",
                "timestamp": _iso_now(),
            }), 503

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in vision analysis: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": _iso_now(),
            }), 500

    # -- /api/vision/describe ------------------------------------------------

    @app.route("/api/vision/describe", methods=["GET", "POST"])
    def api_vision_describe():
        """Lightweight screen description (text only)."""
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get("active_emulator"):
                return jsonify({
                    "success": False,
                    "error": "No ROM loaded",
                    "description": None,
                    "timestamp": _iso_now(),
                }), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]
            img_bytes = emulator.get_screen_bytes()
            if not img_bytes:
                return jsonify({
                    "success": False,
                    "error": "Failed to capture screen",
                    "timestamp": _iso_now(),
                }), 500

            data = request.get_json(silent=True) or {}
            custom_prompt = data.get(
                "prompt",
                "Describe what you see on this Game Boy screen in 2-3 sentences.",
            )

            dmp = _dual_model_provider(ai_provider_manager)
            if dmp is not None:
                analysis = dmp.analyze_screen(img_bytes, {"goal": "describe the screen"})
                return jsonify({
                    "success": True,
                    "description": analysis.raw_description,
                    "model_used": _vision_model_label(ai_provider_manager),
                    "timestamp": _iso_now(),
                }), 200

            connector = _fallback_connector(ai_provider_manager)
            if connector is not None:
                result = connector.chat_with_ai(
                    custom_prompt, img_bytes, {"goal": "describe screen"}
                )
                return jsonify({
                    "success": True,
                    "description": result,
                    "model_used": getattr(ai_provider_manager, "default_provider", "default"),
                    "timestamp": _iso_now(),
                }), 200

            return jsonify({
                "success": False,
                "error": "No AI provider available",
                "timestamp": _iso_now(),
            }), 503

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in vision describe: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": _iso_now(),
            }), 500

    # -- /api/vision/ocr -----------------------------------------------------

    @app.route("/api/vision/ocr", methods=["GET"])
    def api_vision_ocr():
        """OCR / text extraction from screen."""
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get("active_emulator"):
                return jsonify({
                    "success": False,
                    "error": "No ROM loaded",
                    "text": {"raw": "", "lines": [], "has_text": False},
                    "timestamp": _iso_now(),
                }), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]
            img_bytes = emulator.get_screen_bytes()
            if not img_bytes:
                return jsonify({
                    "success": False,
                    "error": "Failed to capture screen",
                    "text": {"raw": "", "lines": [], "has_text": False},
                    "timestamp": _iso_now(),
                }), 500

            dmp = _dual_model_provider(ai_provider_manager)
            if dmp is not None:
                analysis = dmp.analyze_screen(img_bytes, {"goal": "extract text"})
                raw_text = analysis.raw_description
                lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
                has_text = bool(lines) and "no text" not in raw_text.lower()
                return jsonify({
                    "success": True,
                    "text": {
                        "raw": raw_text,
                        "lines": lines,
                        "has_text": has_text,
                        "dialogue_active": any("dialogue" in ln.lower() for ln in lines),
                    },
                    "model_used": _vision_model_label(ai_provider_manager),
                    "timestamp": _iso_now(),
                }), 200

            connector = _fallback_connector(ai_provider_manager)
            if connector is not None and hasattr(connector, "chat_with_ai"):
                result = connector.chat_with_ai(_OCR_PROMPT, img_bytes, {"goal": "ocr"})
                lines = [ln.strip() for ln in result.split("\n") if ln.strip()]
                return jsonify({
                    "success": True,
                    "text": {
                        "raw": result,
                        "lines": lines,
                        "has_text": bool(lines),
                    },
                    "model_used": getattr(ai_provider_manager, "default_provider", "default"),
                    "timestamp": _iso_now(),
                }), 200

            return jsonify({
                "success": False,
                "error": "No vision-capable AI provider available for OCR",
                "text": {"raw": "", "lines": [], "has_text": False},
                "timestamp": _iso_now(),
            }), 503

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in OCR: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "text": {"raw": "", "lines": [], "has_text": False},
                "timestamp": _iso_now(),
            }), 500

    # -- /api/vision/summary -------------------------------------------------

    @app.route("/api/vision/summary", methods=["GET"])
    def api_vision_summary():
        """Fast structured summary."""
        try:
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get("active_emulator"):
                return jsonify({
                    "success": False,
                    "error": "No ROM loaded",
                    "summary": None,
                    "timestamp": _iso_now(),
                }), 400

            emulators = emulators_getter()
            emulator = emulators[current_state["active_emulator"]]
            img_bytes = emulator.get_screen_bytes()
            if not img_bytes:
                return jsonify({
                    "success": False,
                    "error": "Failed to capture screen",
                    "summary": None,
                    "timestamp": _iso_now(),
                }), 500

            dmp = _dual_model_provider(ai_provider_manager)
            if dmp is not None:
                analysis = dmp.analyze_screen(img_bytes, {"goal": "quick summary"})
                raw = analysis.raw_description.lower()
                if "battle" in raw:
                    state = "battle"
                elif "menu" in raw:
                    state = "menu"
                elif "dialog" in raw:
                    state = "dialog"
                elif "explor" in raw or "overworld" in raw:
                    state = "exploration"
                else:
                    state = "unknown"

                safe = "battle" not in raw and "danger" not in raw
                urgency = (
                    "high" if "danger" in raw or "critical" in raw
                    else "medium" if "battle" in raw
                    else "low"
                )
                return jsonify({
                    "success": True,
                    "summary": {
                        "state": state,
                        "safe_to_act": safe,
                        "recommended_action": (
                            analysis.opportunities[0]
                            if analysis.opportunities
                            else "explore"
                        ),
                        "urgency": urgency,
                    },
                    "model_used": _vision_model_label(ai_provider_manager),
                    "timestamp": _iso_now(),
                }), 200

            connector = _fallback_connector(ai_provider_manager)
            if connector is not None and hasattr(connector, "chat_with_ai"):
                result = connector.chat_with_ai(
                    _SUMMARY_PROMPT, img_bytes, {"goal": "summary"}
                )
                parsed = {}
                for line in result.strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed[k.strip().lower().replace(" ", "_")] = v.strip()
                return jsonify({
                    "success": True,
                    "summary": {
                        "state": parsed.get("state", "unknown"),
                        "safe_to_act": parsed.get("safe_to_act", "yes").lower() == "yes",
                        "recommended_action": parsed.get("recommended", "wait"),
                        "urgency": parsed.get("urgency", "low"),
                    },
                    "model_used": getattr(ai_provider_manager, "default_provider", "default"),
                    "timestamp": _iso_now(),
                }), 200

            return jsonify({
                "success": False,
                "error": "No AI provider available",
                "summary": None,
                "timestamp": _iso_now(),
            }), 503

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error in vision summary: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "summary": None,
                "timestamp": _iso_now(),
            }), 500

    # -- /api/vision/status --------------------------------------------------

    @app.route("/api/vision/status", methods=["GET"])
    def api_vision_status():
        """Provider + endpoint metadata for the vision API."""
        try:
            provider_status = (
                ai_provider_manager.get_provider_status()
                if hasattr(ai_provider_manager, "get_provider_status")
                else {}
            )
            dual_model_status = (
                ai_provider_manager.get_dual_model_status()
                if hasattr(ai_provider_manager, "get_dual_model_status")
                else {}
            )
        except Exception:  # noqa: BLE001
            provider_status = {}
            dual_model_status = {}

        return jsonify({
            "vision_available": dual_model_status.get("available", False),
            "dual_model_enabled": dual_model_status.get("enabled", False),
            "vision_model": dual_model_status.get("vision_model", "not configured"),
            "planning_model": dual_model_status.get("planning_model", "not configured"),
            "providers": {
                name: {
                    "available": info.get("available", False),
                    "vision_capable": name in ["openclaw", "lmstudio", "gemini"],
                }
                for name, info in provider_status.items()
            },
            "endpoints": {
                "analyze": "/api/vision/analyze",
                "describe": "/api/vision/describe",
                "ocr": "/api/vision/ocr",
                "summary": "/api/vision/summary",
            },
            "usage_guide": {
                "when_to_use_screenshot": [
                    "Displaying the game to a human user",
                    "Recording gameplay footage",
                    "Visual debugging",
                    "Frontend needs raw pixels",
                ],
                "when_to_use_vision_analysis": [
                    "AI agent needs to understand the screen",
                    "Making gameplay decisions without human",
                    "Extracting text (OCR)",
                    "Detecting game state changes",
                    "LM Studio / MCP agents that can't process images",
                ],
            },
            "timestamp": _iso_now(),
        }), 200
