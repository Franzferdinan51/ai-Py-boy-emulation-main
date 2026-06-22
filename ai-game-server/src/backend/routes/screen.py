"""
routes/screen — Screen capture, SSE stream, performance & emulator-mode endpoints.

Extracted from server.py. Exposes:

  GET  /api/screen                — single-frame base64 screen capture
  GET  /api/screen/debug          — diagnostic info about the current frame
  GET  /api/stream                — SSE stream of frames (text/event-stream)
  GET  /api/performance           — server + emulator performance stats
  GET  /api/emulator/mode         — single- vs multi-process mode
  POST /api/emulator/clear-cache  — clear screen cache + perf buffers

All endpoints degrade gracefully when no emulator / no ROM is loaded.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Optional

from flask import Response, jsonify

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_screen_routes(
    app,
    *,
    game_state_getter: Callable[[], Dict[str, Any]],
    emulators_getter: Callable[[], Dict[str, Any]],
    numpy_to_base64_image: Callable[..., Optional[str]],
    update_performance_metrics: Optional[Callable[[float, float], None]] = None,
    performance_monitor: Optional[Dict[str, Any]] = None,
    get_performance_stats: Optional[Callable[[], Dict[str, Any]]] = None,
    get_memory_usage: Callable[[], float] = lambda: 0.0,
    use_multi_process: bool = False,
    optimization_system_manager: Any = None,
    optimization_system_available: bool = False,
    multiprocessing_cpu_count: Callable[[], int] = lambda: 1,
) -> None:
    _stats = get_performance_stats or (lambda: {})
    _mem = get_memory_usage or (lambda: 0.0)
    """Register screen / stream / performance endpoints.

    ``performance_monitor`` is a mutable dict (e.g. ``{"frame_times": [],
    "encoding_times": [], "current_fps": 0, "adaptive_fps_target": 60}``);
    callers may pass ``None`` to disable performance tracking.
    """
    _perf = performance_monitor or {}

    def _perf_get(key: str, default: Any = None) -> Any:
        return _perf.get(key, default)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/api/screen", methods=["GET"])
    def get_screen():
        try:
            start_time = time.time()
            current_state = game_state_getter()
            if not current_state.get("rom_loaded") or not current_state.get(
                "active_emulator"
            ):
                return jsonify({"error": "No ROM loaded"}), 400

            emulators = emulators_getter() or {}
            emulator = emulators.get(current_state["active_emulator"])
            if emulator is None:
                return jsonify({"error": "Active emulator not found"}), 400

            screen_array = emulator.get_screen()
            if screen_array is None or getattr(screen_array, "size", 0) == 0:
                logger.error("Screen data is None or empty")
                return jsonify({"error": "Failed to capture screen"}), 500

            conversion_start = time.time()
            img_base64 = numpy_to_base64_image(screen_array)
            conversion_time = time.time() - conversion_start
            cache_hit = False

            if not img_base64:
                logger.error("Failed to convert screen to base64")
                return jsonify({"error": "Failed to process screen image"}), 500

            total_time = time.time() - start_time

            if update_performance_metrics is not None:
                try:
                    update_performance_metrics(conversion_time, total_time)
                except Exception:  # noqa: BLE001
                    pass

            response_data: Dict[str, Any] = {
                "image": img_base64,
                "shape": getattr(screen_array, "shape", None),
                "timestamp": time.time(),
                "pyboy_frame": (
                    emulator.get_frame_count()
                    if hasattr(emulator, "get_frame_count")
                    else None
                ),
                "performance": {
                    "total_time_ms": round(total_time * 1000, 2),
                    "conversion_time_ms": round(conversion_time * 1000, 2),
                    "current_fps": round(float(_perf_get("current_fps", 0)), 1),
                    "adaptive_fps_target": _perf_get("adaptive_fps_target", 60),
                },
            }
            if optimization_system_available and optimization_system_manager is not None:
                response_data["optimization"] = {
                    "cache_hit": cache_hit,
                    "memory_pressure": getattr(
                        optimization_system_manager, "get_memory_pressure", lambda: None
                    )(),
                    "optimization_enabled": True,
                }
            return jsonify(response_data), 200
        except Exception as e:  # noqa: BLE001
            logger.error("Error getting screen: %s", exc_info=True)
            return jsonify({"error": f"Internal server error: {e}"}), 500

    @app.route("/api/performance", methods=["GET"])
    def get_performance():
        try:
            stats = _stats() or {}
            emulator_stats: Dict[str, Any] = {}
            current_state = game_state_getter()
            if current_state.get("rom_loaded") and current_state.get("active_emulator"):
                emulators = emulators_getter() or {}
                emulator = emulators.get(current_state["active_emulator"])
                if emulator and hasattr(emulator, "get_performance_stats"):
                    try:
                        emulator_stats = emulator.get_performance_stats() or {}
                    except Exception:  # noqa: BLE001
                        emulator_stats = {}
            return jsonify(
                {
                    "server_performance": stats,
                    "emulator_performance": emulator_stats,
                    "system_info": {
                        "cpu_count": int(multiprocessing_cpu_count() or 1),
                        "memory_usage_mb": float(_mem() or 0),
                        "multi_process_mode": bool(use_multi_process),
                        "timestamp": time.time(),
                    },
                }
            ), 200
        except Exception as e:  # noqa: BLE001
            logger.error("Error getting performance stats: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/emulator/mode", methods=["GET"])
    def get_emulator_mode():
        return jsonify(
            {
                "multi_process_mode": bool(use_multi_process),
                "available_modes": ["single-process", "multi-process"],
                "current_mode": (
                    "multi-process" if use_multi_process else "single-process"
                ),
            }
        ), 200

    @app.route("/api/emulator/clear-cache", methods=["POST"])
    def clear_emulator_cache():
        try:
            current_state = game_state_getter()
            cleared: list = []
            if current_state.get("rom_loaded") and current_state.get("active_emulator"):
                emulators = emulators_getter() or {}
                emulator = emulators.get(current_state["active_emulator"])
                if emulator is not None and hasattr(emulator, "clear_screen_cache"):
                    try:
                        emulator.clear_screen_cache()
                        cleared.append("screen_cache")
                    except Exception:  # noqa: BLE001
                        pass
                ft = _perf.get("frame_times")
                et = _perf.get("encoding_times")
                if hasattr(ft, "clear"):
                    ft.clear()
                if hasattr(et, "clear"):
                    et.clear()
                cleared.append("performance_cache")
                logger.info("Cleared emulator caches: %s", cleared)
                return jsonify(
                    {
                        "message": "Caches cleared successfully",
                        "cleared_caches": cleared,
                        "cache_size_after": {
                            "performance_monitor": {
                                "frame_times": len(ft) if ft is not None else 0,
                                "encoding_times": len(et) if et is not None else 0,
                            }
                        },
                    }
                ), 200
            return jsonify({"error": "No emulator running"}), 400
        except Exception as e:  # noqa: BLE001
            logger.error("Error clearing caches: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/screen/debug", methods=["GET"])
    def get_screen_debug():
        try:
            current_state = game_state_getter() or {}
            debug_info: Dict[str, Any] = {
                "rom_loaded": current_state.get("rom_loaded", False),
                "active_emulator": current_state.get("active_emulator"),
                "rom_path": current_state.get("rom_path"),
                "timestamp": time.time(),
            }
            emulators = emulators_getter() or {}
            emulator = None
            if current_state.get("rom_loaded") and current_state.get("active_emulator"):
                emulator = emulators.get(current_state["active_emulator"])
            if emulator is not None:
                if hasattr(emulator, "get_info"):
                    try:
                        debug_info["emulator_info"] = emulator.get_info() or {}
                    except Exception:  # noqa: BLE001
                        debug_info["emulator_info"] = {}
                screen_array = None
                try:
                    screen_array = emulator.get_screen()
                except Exception:  # noqa: BLE001
                    screen_array = None
                debug_info.update(
                    {
                        "screen_shape": getattr(screen_array, "shape", None),
                        "screen_dtype": (
                            str(screen_array.dtype) if screen_array is not None else None
                        ),
                        "screen_min": (
                            int(screen_array.min())
                            if screen_array is not None
                            else None
                        ),
                        "screen_max": (
                            int(screen_array.max())
                            if screen_array is not None
                            else None
                        ),
                        "screen_size": getattr(screen_array, "size", None),
                    }
                )
                if screen_array is not None and getattr(screen_array, "size", 0) > 0:
                    img_base64 = numpy_to_base64_image(screen_array)
                    debug_info["base64_success"] = bool(img_base64)
                    debug_info["base64_length"] = len(img_base64) if img_base64 else 0
                    debug_info["base64_preview"] = (
                        (img_base64[:100] + "...")
                        if img_base64 and len(img_base64) > 100
                        else img_base64
                    )
                else:
                    debug_info["base64_success"] = False
                    debug_info["base64_length"] = 0
                    debug_info["base64_preview"] = None
            else:
                debug_info.update(
                    {
                        "error": "No ROM loaded",
                        "screen_shape": None,
                        "screen_dtype": None,
                        "base64_success": False,
                        "base64_length": 0,
                        "base64_preview": None,
                    }
                )
            status = 200 if current_state.get("rom_loaded") else 503
            return jsonify(debug_info), status
        except Exception as e:  # noqa: BLE001
            logger.error("Error in debug screen endpoint: %s", exc_info=True)
            return jsonify({"error": f"Debug endpoint error: {e}"}), 500

    @app.route("/api/stream", methods=["GET"])
    def stream_screen():
        """Server-Sent Events stream of base64-encoded frames."""
        target_fps = 30
        frame_interval = 1.0 / target_fps
        max_consecutive_errors = 20

        def generate():
            logger.info("SSE stream requested")
            state = game_state_getter() or {}
            if not state.get("rom_loaded") or not state.get("active_emulator"):
                yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
                return

            emulators = emulators_getter() or {}
            emulator = emulators.get(state["active_emulator"])
            if emulator is None:
                yield f"data: {json.dumps({'error': 'Active emulator not found'})}\n\n"
                return

            consecutive_errors = 0
            yield f"data: {json.dumps({'status': 'stream_started', 'fps': target_fps})}\n\n"

            while True:
                loop_start = time.time()
                try:
                    state = game_state_getter() or {}
                    if not state.get("rom_loaded") or not state.get("active_emulator"):
                        yield f"data: {json.dumps({'error': 'ROM unloaded'})}\n\n"
                        break

                    if hasattr(emulator, "step"):
                        try:
                            emulator.step("NOOP", 1)
                        except Exception:  # noqa: BLE001
                            pass

                    screen_array = emulator.get_screen() if hasattr(
                        emulator, "get_screen"
                    ) else None
                    if screen_array is None or getattr(screen_array, "size", 0) == 0:
                        raise RuntimeError("Empty screen buffer")

                    img_base64 = numpy_to_base64_image(screen_array)
                    if not img_base64:
                        raise RuntimeError("Failed to encode frame")

                    payload = {
                        "image": img_base64,
                        "timestamp": time.time(),
                        "frame": (
                            emulator.get_frame_count()
                            if hasattr(emulator, "get_frame_count")
                            else None
                        ),
                        "fps": target_fps,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    consecutive_errors = 0
                except GeneratorExit:
                    logger.info("SSE client disconnected")
                    break
                except Exception as e:  # noqa: BLE001
                    consecutive_errors += 1
                    logger.warning(
                        "SSE stream frame error (%d/%d): %s",
                        consecutive_errors,
                        max_consecutive_errors,
                        e,
                    )
                    yield f"data: {json.dumps({'error': str(e), 'recoverable': True, 'consecutive_errors': consecutive_errors})}\n\n"
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("SSE stream aborting after too many consecutive errors")
                        break
                    time.sleep(0.2)

                elapsed = time.time() - loop_start
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )