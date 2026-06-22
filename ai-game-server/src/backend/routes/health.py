"""
routes/health — System health & uptime endpoints (OpenClaw-style).

Extracted from server.py. Exposes:

  GET /health                  — basic health check (no emulation deps)
  GET /api/health              — comprehensive rollup across all components
  GET /api/health/runtime      — runtime status (Python, uptime, memory)
  GET /api/health/emulator     — emulator component status
  GET /api/health/stream       — stream / WebSocket status

All endpoints degrade gracefully when their backing service is not loaded —
they return a structured payload with ``status="unknown"`` or a 503 hint
instead of crashing. That mirrors the original server.py semantics.
"""
from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from flask import jsonify

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Default dependency implementations
# ----------------------------------------------------------------------
# These are weak fallbacks used when the blueprint is wired without all
# of its production dependencies. They never raise; they report "unknown".


def _default_format_uptime(seconds: float) -> str:
    """Format uptime seconds to a short human-readable string."""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _default_get_memory_usage() -> float:
    """Best-effort RSS in MB. Returns 0.0 if psutil is missing."""
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        try:
            import resource  # type: ignore

            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux reports kilobytes
            if sys.platform == "darwin":
                return float(rss_kb) / (1024 * 1024)
            return float(rss_kb) / 1024.0
        except Exception:  # noqa: BLE001
            return 0.0


def _default_ws_status() -> Tuple[bool, int, int]:
    """Return ``(running, port, active_clients)``."""
    return (False, 5003, 0)


def _default_performance_stats() -> Dict[str, Any]:
    return {
        "current_fps": 0,
        "avg_frame_time": 0,
        "avg_encoding_time": 0,
        "adaptive_fps_target": 60,
    }


def _default_component_health() -> Dict[str, Dict[str, Any]]:
    return {
        "runtime": {"status": "unknown", "last_check": None},
        "emulator": {"status": "unknown", "last_check": None, "error": None},
        "stream": {"status": "unknown", "last_check": None, "clients": 0},
        "agent": {"status": "unknown", "last_check": None},
    }


# Import os lazily so the helper above doesn't fail on Windows where it
# may not be present at top of file in some test runners.
import os  # noqa: E402


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_health_routes(
    app,
    *,
    server_start_time_getter: Callable[[], float] = lambda: time.time(),
    pyboy_available: bool = False,
    mcp_available: bool = False,
    format_uptime: Callable[[float], str] = _default_format_uptime,
    get_memory_usage: Callable[[], float] = _default_get_memory_usage,
    component_health: Optional[Dict[str, Dict[str, Any]]] = None,
    ws_status_getter: Callable[[], Tuple[bool, int, int]] = _default_ws_status,
    game_state_getter: Callable[[], Dict[str, Any]] = lambda: {
        "rom_loaded": False,
        "active_emulator": None,
    },
    emulators_getter: Callable[[], Dict[str, Any]] = lambda: {},
    get_performance_stats: Callable[[], Dict[str, Any]] = _default_performance_stats,
    agent_state_getter: Callable[[], Dict[str, Any]] = lambda: {
        "enabled": False,
        "mode": "manual",
        "errors": [],
    },
) -> None:
    """Register health and uptime endpoints on ``app``.

    Dependencies are passed as callables so the blueprint can be wired in
    isolation without importing the legacy ``server`` module. Defaults
    produce a working but minimal payload.
    """
    _component_health = (
        dict(component_health) if component_health else _default_component_health()
    )
    # ensure keys exist
    for key in ("runtime", "emulator", "stream", "agent"):
        _component_health.setdefault(key, {})

    # ------------------------------------------------------------------
    # Per-component helpers (private to the blueprint)
    # ------------------------------------------------------------------

    def _runtime_health_dict() -> Dict[str, Any]:
        uptime_seconds = time.time() - server_start_time_getter()
        memory_mb = float(get_memory_usage() or 0)
        memory_status = "ok"
        if memory_mb > 1000:
            memory_status = "warning"
        elif memory_mb > 2000:
            memory_status = "critical"
        checks = {
            "flask": "ok",
            "pyboy": "ok" if pyboy_available else "not_available",
            "mcp": "ok" if mcp_available else "not_available",
            "memory": memory_status,
        }
        if checks["flask"] == "error" or checks["memory"] == "critical":
            status = "unhealthy"
        elif checks["pyboy"] == "error" or checks["memory"] == "warning":
            status = "degraded"
        else:
            status = "healthy"
        return {
            "status": status,
            "uptime_seconds": round(uptime_seconds, 2),
            "memory_mb": round(memory_mb, 2),
            "checks": checks,
        }

    def _emulator_health_dict() -> Dict[str, Any]:
        state = game_state_getter()
        active = state.get("active_emulator")
        if not state.get("rom_loaded") or not active:
            return {"status": "not_loaded", "rom_loaded": False}
        try:
            emulators = emulators_getter() or {}
            emulator = emulators.get(active)
            if emulator and hasattr(emulator, "is_running") and emulator.is_running():
                frame_count = (
                    emulator.get_frame_count()
                    if hasattr(emulator, "get_frame_count")
                    else 0
                )
                return {
                    "status": "healthy",
                    "rom_loaded": True,
                    "rom_name": state.get("rom_name", state.get("rom_path")),
                    "active_emulator": active,
                    "frame_count": frame_count,
                }
            if emulator and state.get("rom_loaded"):
                return {
                    "status": "degraded",
                    "rom_loaded": True,
                    "active_emulator": active,
                }
            return {"status": "unhealthy", "rom_loaded": False}
        except Exception as e:  # noqa: BLE001
            logger.debug("Emulator health check error: %s", e)
            return {"status": "unhealthy", "error": str(e)}

    def _stream_health_dict() -> Dict[str, Any]:
        running, _port, clients = ws_status_getter()
        return {
            "status": "healthy" if running else "unhealthy",
            "websocket_running": running,
            "active_clients": clients,
        }

    def _agent_health_dict() -> Dict[str, Any]:
        astate = agent_state_getter() or {}
        enabled = bool(astate.get("enabled", False))
        mode = astate.get("mode", "manual")
        errors = astate.get("errors", []) or []
        recent_errors = [e for e in errors[-5:] if e]
        if not enabled:
            status = "healthy"  # disabled is healthy
        elif len(recent_errors) > 3:
            status = "degraded"
        else:
            status = "healthy"
        return {
            "status": status,
            "enabled": enabled,
            "mode": mode,
            "recent_errors": len(recent_errors),
        }

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health_check():
        """Basic health check endpoint (OpenClaw-style)."""
        uptime_seconds = time.time() - server_start_time_getter()
        checks = {
            "flask": "ok",
            "pyboy": "ok" if pyboy_available else "not_available",
            "mcp": "ok" if mcp_available else "not_available",
        }
        if checks["flask"] == "error":
            status = "unhealthy"
        elif checks["pyboy"] == "error":
            status = "degraded"
        else:
            status = "healthy"
        return jsonify(
            {
                "status": status,
                "service": "ai-game-server",
                "version": "3.0.0",
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "uptime_seconds": round(uptime_seconds, 2),
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
            }
        ), 200

    @app.route("/api/health/runtime", methods=["GET"])
    def api_health_runtime():
        """Runtime health (uptime, memory, dependency availability)."""
        uptime_seconds = time.time() - server_start_time_getter()
        uptime_human = format_uptime(uptime_seconds)
        runtime = _runtime_health_dict()
        _component_health["runtime"]["status"] = runtime["status"]
        _component_health["runtime"]["last_check"] = datetime.now().isoformat()
        _component_health["runtime"]["uptime_seconds"] = uptime_seconds
        return jsonify(
            {
                "status": runtime["status"],
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_human": uptime_human,
                "checks": runtime["checks"],
                "memory_mb": runtime["memory_mb"],
                "version": "3.0.0",
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/health/emulator", methods=["GET"])
    def api_health_emulator():
        """Emulator component health."""
        state = game_state_getter()
        active = state.get("active_emulator")
        emu_payload = _emulator_health_dict()
        status = emu_payload.get("status", "unknown")
        perf_stats = get_performance_stats() or {}
        frame_count = 0
        if active:
            emulators = emulators_getter() or {}
            emulator = emulators.get(active)
            if emulator and hasattr(emulator, "get_frame_count"):
                try:
                    frame_count = emulator.get_frame_count()
                except Exception:  # noqa: BLE001
                    frame_count = 0
        _component_health["emulator"]["status"] = status
        _component_health["emulator"]["last_check"] = datetime.now().isoformat()
        _component_health["emulator"]["frame_count"] = frame_count
        return jsonify(
            {
                "status": status,
                "rom_loaded": state.get("rom_loaded", False),
                "rom_name": state.get("rom_name", state.get("rom_path")),
                "active_emulator": active,
                "frame_count": frame_count,
                "fps": round(float(perf_stats.get("current_fps", 0) or 0), 2),
                "last_check": datetime.now().isoformat(),
                "error": _component_health["emulator"].get("error"),
                "performance": {
                    "avg_frame_time_ms": round(
                        float(perf_stats.get("avg_frame_time", 0) or 0) * 1000, 2
                    ),
                    "avg_encoding_time_ms": round(
                        float(perf_stats.get("avg_encoding_time", 0) or 0) * 1000, 2
                    ),
                    "adaptive_fps_target": perf_stats.get("adaptive_fps_target", 60),
                },
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/health/stream", methods=["GET"])
    def api_health_stream():
        """Stream / WebSocket component health."""
        running, port, clients = ws_status_getter()
        status = "healthy" if running else "unhealthy"
        _component_health["stream"]["status"] = status
        _component_health["stream"]["last_check"] = datetime.now().isoformat()
        _component_health["stream"]["clients"] = clients
        return jsonify(
            {
                "status": status,
                "websocket_running": running,
                "websocket_port": port,
                "websocket_url": f"ws://localhost:{port}/api/ws/stream" if running else None,
                "active_clients": clients,
                "sse_endpoint": "/api/stream",
                "last_check": datetime.now().isoformat(),
                "error": _component_health["stream"].get("error"),
                "timestamp": datetime.now().isoformat(),
            }
        ), 200

    @app.route("/api/health", methods=["GET"])
    def api_health_comprehensive():
        """Aggregate health across runtime / emulator / stream / agent."""
        components = {
            "runtime": _runtime_health_dict(),
            "emulator": _emulator_health_dict(),
            "stream": _stream_health_dict(),
            "agent": _agent_health_dict(),
        }
        statuses = [c.get("status", "unknown") for c in components.values()]
        summary = {
            "healthy_count": statuses.count("healthy"),
            "degraded_count": statuses.count("degraded"),
            "unhealthy_count": statuses.count("unhealthy"),
            "unknown_count": statuses.count("unknown")
            + statuses.count("not_loaded"),
        }
        if summary["unhealthy_count"] > 0:
            status = "unhealthy"
        elif summary["degraded_count"] > 0:
            status = "degraded"
        else:
            status = "healthy"
        return jsonify(
            {
                "status": status,
                "components": components,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
            }
        ), 200