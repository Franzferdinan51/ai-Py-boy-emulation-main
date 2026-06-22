"""
Enhanced AI Game Boy Server with Stream Stability Fixes
"""
import os
import sys
import signal
import threading
import io
import multiprocessing  # FIX: Added missing import for cpu_count()
import zlib
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import base64
import logging
import tempfile
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import time
from flask_cors import CORS
import numpy as np
from PIL import Image
import asyncio
import websockets
from websockets.server import serve

# Enhanced resource management
try:
    from .utils.enhanced_resource_manager import resource_manager
    ENHANCED_RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_RESOURCE_MANAGER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Enhanced resource manager not available, falling back to basic management")

# Secure configuration management
try:
    from .utils.secure_config import secure_config
    SECURE_CONFIG_AVAILABLE = True
except ImportError:
    SECURE_CONFIG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Secure configuration manager not available, falling back to basic configuration")

# Performance optimization imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PyBoy emulator availability
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False

# MCP server availability
try:
    from mcp.server import Server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Performance monitoring import
try:
    from utils.performance_monitor import performance_monitor
    PERFORMANCE_MONITOR_AVAILABLE = True
    print("Performance monitoring system enabled")
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    print("Performance monitoring system temporarily disabled")

# Optimization system imports
try:
    from utils.optimization_system_manager import OptimizationSystemManager, OptimizationConfig
    optimization_system_manager = OptimizationSystemManager()
    OPTIMIZATION_SYSTEM_AVAILABLE = True
    print("Optimization system enabled")
except ImportError as e:
    OPTIMIZATION_SYSTEM_AVAILABLE = False
    optimization_system_manager = None
    print(f"Optimization system temporarily disabled: {e}")
except Exception as e:
    OPTIMIZATION_SYSTEM_AVAILABLE = False
    optimization_system_manager = None
    print(f"Optimization system initialization error: {e}")

# Import state manager separately
try:
    from utils.advanced_emulator_state_manager import state_manager
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from advanced_emulator_state_manager import state_manager
        STATE_MANAGER_AVAILABLE = True
    except ImportError:
        state_manager = None
        STATE_MANAGER_AVAILABLE = False

# Configuration with secure config integration
try:
    from ...config import *
except ImportError:
    # Use secure configuration if available
    if SECURE_CONFIG_AVAILABLE:
        backend_config = secure_config.get_backend_config()
        HOST = backend_config['HOST']
        PORT = backend_config['PORT']
        # Security: Additional check for debug mode
        config_debug = backend_config.get('DEBUG', False)
        flask_env = os.environ.get('FLASK_ENV', 'production').lower()
        DEBUG = config_debug and flask_env == 'development'
        SECRET_KEY = backend_config['SECRET_KEY']
        ALLOWED_HOSTS = backend_config['ALLOWED_HOSTS']
        RATE_LIMIT = backend_config['RATE_LIMIT']
        MAX_UPLOAD_SIZE = backend_config['MAX_UPLOAD_SIZE']
        ENABLE_OPTIMIZATION = backend_config['ENABLE_OPTIMIZATION']
        OPTIMIZATION_MEMORY_MB = backend_config['OPTIMIZATION_MEMORY_MB']
        OPTIMIZATION_AUTO_SCALING = backend_config['OPTIMIZATION_AUTO_SCALING']
        OPTIMIZATION_CACHE_SIZE = backend_config['OPTIMIZATION_CACHE_SIZE']
        OPTIMIZATION_MONITORING_INTERVAL = backend_config['OPTIMIZATION_MONITORING_INTERVAL']
    else:
        # Default configuration if config.py is not found
        HOST = "0.0.0.0"
        PORT = int(os.environ.get("BACKEND_PORT", 5002))
        # Security: Debug mode should be disabled in production
        flask_env = os.environ.get('FLASK_ENV', 'production').lower()
        flask_debug = os.environ.get('FLASK_DEBUG', 'false').lower()
        # Only allow debug mode in explicit development environment
        DEBUG = (flask_env == 'development' and flask_debug in ('true', '1', 'yes', 'on'))
        SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
        ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
        RATE_LIMIT = int(os.environ.get('RATE_LIMIT', 60))
        MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 50 * 1024 * 1024))  # 50MB
        ENABLE_OPTIMIZATION = os.environ.get('ENABLE_OPTIMIZATION', 'true').lower() in ('true', '1', 'yes', 'on')
        OPTIMIZATION_MEMORY_MB = int(os.environ.get('OPTIMIZATION_MEMORY_MB', 1024))
        OPTIMIZATION_AUTO_SCALING = os.environ.get('OPTIMIZATION_AUTO_SCALING', 'true').lower() in ('true', '1', 'yes', 'on')
        OPTIMIZATION_CACHE_SIZE = int(os.environ.get('OPTIMIZATION_CACHE_SIZE', 500))
        OPTIMIZATION_MONITORING_INTERVAL = int(os.environ.get('OPTIMIZATION_MONITORING_INTERVAL', 5))

    # Default emulator settings
    DEFAULT_EMULATOR = "gb"
    SCREEN_CAPTURE_FORMAT = "jpeg"
    SCREEN_CAPTURE_QUALITY = 85
    DEFAULT_AI_API = "gemini"
    AI_REQUEST_TIMEOUT = 30
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "{asctime} - {name} - {levelname} - {message}"
    LOG_FILE = "ai_game_server.log"
    MAX_ROM_SIZE = 100 * 1024 * 1024
    ALLOWED_ROM_EXTENSIONS = [".gb", ".gbc", ".gba"]
    ACTION_HISTORY_LIMIT = 1000

    # Security settings
    # Rate limiting (requests per minute)
    RATE_LIMIT = int(os.environ.get('RATE_LIMIT', 60))
    # File upload size limit (bytes)
    MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 50 * 1024 * 1024))  # 50MB
    # Allowed hostnames
    ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    # Secret key for sessions
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

    # Optimization system settings
    ENABLE_OPTIMIZATION = os.environ.get('ENABLE_OPTIMIZATION', 'true').lower() in ('true', '1', 'yes', 'on')
    OPTIMIZATION_MEMORY_MB = int(os.environ.get('OPTIMIZATION_MEMORY_MB', 1024))
    OPTIMIZATION_AUTO_SCALING = os.environ.get('OPTIMIZATION_AUTO_SCALING', 'true').lower() in ('true', '1', 'yes', 'on')
    OPTIMIZATION_CACHE_SIZE = int(os.environ.get('OPTIMIZATION_CACHE_SIZE', 500))
    OPTIMIZATION_MONITORING_INTERVAL = int(os.environ.get('OPTIMIZATION_MONITORING_INTERVAL', 5))

# Keep backend logs in the server tree by default, but don't crash startup if the
# filesystem is read-only (common in containers and managed deploys).
SERVER_ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(SERVER_ROOT, "ai_game_server.log"))

def _build_log_handlers():
    handlers = [logging.StreamHandler()]

    if not LOG_FILE:
        return handlers

    try:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.insert(0, logging.FileHandler(LOG_FILE))
    except OSError as exc:
        print(f"Warning: failed to open log file {LOG_FILE}: {exc}. Falling back to stdout only.")

    return handlers

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    style='{',
    handlers=_build_log_handlers()
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize optimization system if enabled
if OPTIMIZATION_SYSTEM_AVAILABLE and ENABLE_OPTIMIZATION:
    try:
        # Create optimization configuration
        opt_config = OptimizationConfig(
            memory_management=True,
            max_memory_mb=OPTIMIZATION_MEMORY_MB,
            thread_pool_management=True,
            auto_scaling=OPTIMIZATION_AUTO_SCALING,
            ai_caching=True,
            max_ai_responses=OPTIMIZATION_CACHE_SIZE,
            performance_monitoring=True,
            monitoring_interval_seconds=OPTIMIZATION_MONITORING_INTERVAL,
            error_handling=True,
            state_management=True,
            enable_compression=True,
            adaptive_optimization=True
        )

        # Initialize optimization system
        if optimization_system_manager.initialize_systems():
            optimization_system_manager.start_monitoring()
            logger.info("[SUCCESS] Optimization system initialized and started")
        else:
            logger.warning("[WARNING] Optimization system initialization failed")
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize optimization system: {e}")
        OPTIMIZATION_SYSTEM_AVAILABLE = False
else:
    logger.info("[INFO] Optimization system disabled or not available")

# Configure CORS with secure settings
# Get allowed origins from environment or use development defaults
frontend_port = os.environ.get('FRONTEND_PORT', '5173')
glm_ui_port = os.environ.get('GLM_UI_PORT', '3000')
backend_port = os.environ.get('BACKEND_PORT', '5002')  # Fixed: Must match PORT default (line 132)
flask_env = os.environ.get('FLASK_ENV', 'development')

if flask_env == 'production':
    # In production, only allow specific origins
    allowed_origins = [
        f"http://localhost:{frontend_port}",
        f"http://localhost:{glm_ui_port}",
        # Add your production domain here when deployed
    ]
else:
    # In development, allow localhost origins
    allowed_origins = [
        f"http://localhost:{frontend_port}",
        f"http://localhost:{glm_ui_port}",
        f"http://127.0.0.1:{frontend_port}",
        f"http://127.0.0.1:{glm_ui_port}",
    ]

CORS(app, resources={
    r"/*": {
        "origins": allowed_origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
        "supports_credentials": True,
        "expose_headers": ["Content-Disposition", "Content-Length"]
    }
}, max_age=86400)  # Cache preflight for 24 hours

# Combined security middleware
@app.before_request
def security_middleware():
    """Combined security middleware for host validation and rate limiting"""
    # Skip validation for local development or localhost traffic
    host = request.host.split(':')[0]  # Remove port
    client_ip = get_client_ip()
    if DEBUG or host in ('localhost', '127.0.0.1', '::1') or client_ip in ('127.0.0.1', '::1', 'localhost'):
        return

    # Host validation
    if host not in [h.strip() for h in ALLOWED_HOSTS]:
        logger.warning(f"Unauthorized host access attempt: {host}")
        return jsonify({"error": "Unauthorized"}), 403

    # Rate limiting
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later.",
            "retry_after": 60
        }), 429

# Security middleware for HTTP headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    return response

# Import emulators
from backend.emulators.pyboy_emulator import PyBoyEmulator, PyBoyEmulatorMP
from backend.emulators.pygba_emulator import PyGBAEmulator
from backend.ai_apis.ai_provider_manager import ai_provider_manager
from backend.ai_apis.openclaw_model_discovery import get_model_discovery

# Configuration for emulator mode
USE_MULTI_PROCESS = os.environ.get('USE_MULTI_PROCESS', 'false').lower() == 'true'

# Initialize emulators based on configuration
if USE_MULTI_PROCESS:
    logger.info("Using multi-process emulator mode")
    emulators = {
        "pyboy": PyBoyEmulatorMP(),
        "pygba": PyGBAEmulator()  # GBA emulator doesn't have MP mode yet
    }
else:
    logger.info("Using single-process emulator mode")
    emulators = {
        "pyboy": PyBoyEmulator(),
        "pygba": PyGBAEmulator()
    }

# Thread-safe global state management
game_state_lock = threading.Lock()
action_history_lock = threading.Lock()

# Action history
action_history = []

# Game state
game_state = {
    "active_emulator": None,
    "rom_loaded": False,
    "ai_running": False,
    "current_goal": "",
    "rom_path": None,
    "fps": 60,
    "speed_multiplier": 1.0,
    "current_provider": None,
    "current_model": None
}

# In-memory save-state store keyed by emulator id
saved_states = {}

ai_runtime_state = {
    "provider": "bailian",
    "model": "bailian/qwen3.5-plus",
    "api_endpoint": "",
}


def _set_ai_runtime_state(new_state: dict) -> None:
    """Replace the in-memory AI runtime state. Used by routes/ai_runtime."""
    global ai_runtime_state
    ai_runtime_state = dict(new_state)

# Agent state for OpenClaw-style status tracking
agent_state = {
    "mode": "manual",  # "manual", "autonomous", "ai_assisted"
    "enabled": False,
    "current_goal": "",
    "current_task": "",
    "last_decision": None,
    "last_action": None,
    "last_action_time": None,
    "errors": [],  # Recent errors (last 10)
    "actions": [],  # Recent actions (last 20)
    "decisions": [],  # Recent AI decisions (last 10)
    "started_at": None,
    "stats": {
        "total_actions": 0,
        "total_decisions": 0,
        "total_errors": 0,
    }
}


def _get_agent_state() -> dict:
    """Mutable reference to the in-memory agent_state dict. Used by routes/agent."""
    return agent_state


def _mutate_agent_state(updates: dict) -> None:
    """Merge ``updates`` into ``agent_state`` atomically. Used by routes/agent."""
    agent_state.update(updates)

# Component health tracking
component_health = {
    "emulator": {
        "status": "unknown",
        "last_check": None,
        "error": None,
        "frame_count": 0,
    },
    "stream": {
        "status": "unknown",
        "last_check": None,
        "error": None,
        "clients": 0,
    },
    "runtime": {
        "status": "unknown",
        "last_check": None,
        "error": None,
        "uptime_seconds": 0,
    }
}

# Server start time for uptime tracking
SERVER_START_TIME = time.time()

if ai_runtime_state["provider"] == "bailian":
    ai_runtime_state["model"] = os.environ.get('BAILIAN_MODEL', 'bailian/qwen3.5-plus')
elif ai_runtime_state["provider"] == "lmstudio":
    ai_runtime_state["model"] = os.environ.get('LM_STUDIO_THINKING_MODEL', '')
    ai_runtime_state["api_endpoint"] = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')
elif ai_runtime_state["provider"] == "openai-compatible":
    ai_runtime_state["model"] = os.environ.get('OPENAI_MODEL', '')
    ai_runtime_state["api_endpoint"] = os.environ.get('OPENAI_ENDPOINT', '')
elif ai_runtime_state["provider"] == "gemini":
    ai_runtime_state["model"] = os.environ.get('GEMINI_MODEL', '')
elif ai_runtime_state["provider"] == "openrouter":
    ai_runtime_state["model"] = os.environ.get('OPENROUTER_MODEL', '')
elif ai_runtime_state["provider"] == "nvidia":
    ai_runtime_state["model"] = os.environ.get('NVIDIA_MODEL', '')

# Background live emulation loop state
emulation_loop_running = False
emulation_loop_thread = None

# AI provider manager is imported from ai_provider_manager

def get_game_state():
    """Thread-safe getter for game state"""
    with game_state_lock:
        return game_state.copy()

def update_game_state(updates):
    """Thread-safe updater for game state"""
    with game_state_lock:
        game_state.update(updates)

# ------------------------------------------------------------------
# Register refactored route blueprints (config, save_load, ui, ws, tetris, vision)
# ------------------------------------------------------------------
# Done at module load so test clients and IDE tooling see the full surface.
# agent_features is registered separately inside main() to avoid an
# import-time side-effect that pulls the OpenClaw provider.
try:
    from backend.routes import register_all as _register_route_blueprints
    from backend.routes.ws import WebSocketRunner as _WebSocketRunner

    def _ws_is_running():
        return bool(getattr(sys.modules[__name__], "ws_server_running", False))

    def _ws_port():
        return int(getattr(sys.modules[__name__], "WS_PORT", 5003) or 5003)

    def _ws_clients():
        ws_clients = getattr(sys.modules[__name__], "ws_clients", None)
        if ws_clients is None:
            return 0
        try:
            return len(ws_clients)
        except TypeError:
            return 0

    def _ws_start():
        fn = getattr(sys.modules[__name__], "start_websocket_server", None)
        if fn is None:
            raise RuntimeError("WebSocket server start function not available")
        return fn()

    def _ws_stop():
        fn = getattr(sys.modules[__name__], "stop_websocket_server", None)
        if fn is None:
            return None
        return fn()

    _ws_runner = _WebSocketRunner(
        is_running=_ws_is_running,
        get_port=_ws_port,
        get_clients=_ws_clients,
        start_fn=_ws_start,
        stop_fn=_ws_stop,
    )

    _route_counts = _register_route_blueprints(
        app,
        emulators_getter=lambda: emulators,
        game_state_getter=get_game_state,
        ai_provider_manager=ai_provider_manager,
        secure_config=secure_config if SECURE_CONFIG_AVAILABLE else None,
        secure_config_available=SECURE_CONFIG_AVAILABLE,
        host=HOST,
        port=PORT,
        debug=DEBUG,
        saved_states=saved_states,
        websocket_runner=_ws_runner,
        health_server_start_time_getter=lambda: SERVER_START_TIME,
        health_pyboy_available=PYBOY_AVAILABLE,
        health_mcp_available=MCP_AVAILABLE,
        health_component_health=component_health,
        ai_models_get_model_discovery=get_model_discovery,
        ai_models_openclaw_endpoint_getter=lambda: app.config.get(
            'OPENCLAW_ENDPOINT', 'http://localhost:18789'
        ),
        ai_runtime_state_getter=lambda: ai_runtime_state,
        ai_runtime_state_setter=_set_ai_runtime_state,
        ai_runtime_openclaw_endpoint_setter=lambda v: app.config.__setitem__('OPENCLAW_ENDPOINT', v),
        screen_use_multi_process=USE_MULTI_PROCESS,
        screen_optimization_system_manager=optimization_system_manager,
        screen_optimization_system_available=OPTIMIZATION_SYSTEM_AVAILABLE,
        input_ai_request_timeout=AI_REQUEST_TIMEOUT,
        # agent blueprint deps
        agent_state_getter=_get_agent_state,
        agent_state_mutate=_mutate_agent_state,
        agent_ai_apis_getter=lambda: ai_provider_manager.providers or {},
        agent_get_action_history=lambda: get_action_history(),
        # rom blueprint deps (upload pipeline helpers — defined later in
        # this module, so we resolve through module globals at call time)
        rom_validate_file_upload=lambda *a, **kw: validate_file_upload(*a, **kw),
        rom_validate_string_input=lambda *a, **kw: validate_string_input(*a, **kw),
        rom_sanitize_filename=lambda fn: sanitize_filename(fn),
        rom_max_rom_size=MAX_ROM_SIZE,
        rom_allowed_extensions=ALLOWED_ROM_EXTENSIONS,
        rom_configure_emulator_launch_ui=lambda *a, **kw: configure_emulator_launch_ui(*a, **kw),
        rom_sync_loaded_rom_state=lambda *a, **kw: sync_loaded_rom_state(*a, **kw),
        rom_ensure_emulation_loop_running=lambda: ensure_emulation_loop_running(),
    )
    logger.info(
        f"[routes] Registered — "
        f"config:{_route_counts.get('config',0)} "
        f"health:{_route_counts.get('health',0)} "
        f"ai_models:{_route_counts.get('ai_models',0)} "
        f"ai_runtime:{_route_counts.get('ai_runtime',0)} "
        f"save_load:{_route_counts.get('save_load',0)} "
        f"screen:{_route_counts.get('screen',0)} "
        f"input:{_route_counts.get('input',0)} "
        f"agent:{_route_counts.get('agent',0)} "
        f"spatial:{_route_counts.get('spatial',0)} "
        f"rom:{_route_counts.get('rom',0)} "
        f"ui:{_route_counts.get('ui',0)} "
        f"ws:{_route_counts.get('ws',0)} "
        f"tetris:{_route_counts.get('tetris',0)} "
        f"vision:{_route_counts.get('vision',0)}"
    )
except Exception as _route_exc:  # noqa: BLE001
    logger.error(f"[routes] Failed to register blueprints: {_route_exc}", exc_info=True)

def get_action_history():
    """Thread-safe getter for action history"""
    with action_history_lock:
        return action_history.copy()

def add_to_action_history(action):
    """Thread-safe method to add to action history"""
    with action_history_lock:
        action_history.append(action)
        # Keep history within limits
        if len(action_history) > ACTION_HISTORY_LIMIT:
            action_history.pop(0)

# =========================================
# Agent State Tracking Helpers
# =========================================

def record_agent_action(action: str, frames: int = 1, result: str = "success", source: str = "manual"):
    """
    Record an action to agent state for OpenClaw-style tracking.
    
    Args:
        action: The action that was executed
        frames: Number of frames the action was held
        result: "success" or "error"
        source: "manual", "ai", "autonomous"
    """
    now = datetime.now().isoformat()
    
    action_record = {
        "timestamp": now,
        "action": action,
        "frames": frames,
        "result": result,
        "source": source
    }
    
    agent_state['last_action'] = action
    agent_state['last_action_time'] = now
    agent_state['actions'].append(action_record)
    
    # Keep only last 50 actions
    if len(agent_state['actions']) > 50:
        agent_state['actions'] = agent_state['actions'][-50:]
    
    # Update stats
    agent_state['stats']['total_actions'] = agent_state['stats'].get('total_actions', 0) + 1


def record_agent_error(error_type: str, message: str, context: dict = None):
    """
    Record an error to agent state for OpenClaw-style tracking.
    
    Args:
        error_type: Type of error (e.g., "emulator_error", "ai_error", "action_error")
        message: Error message
        context: Optional context dictionary
    """
    now = datetime.now().isoformat()
    
    error_record = {
        "timestamp": now,
        "type": error_type,
        "message": message,
        "context": context
    }
    
    agent_state['errors'].append(error_record)
    
    # Keep only last 20 errors
    if len(agent_state['errors']) > 20:
        agent_state['errors'] = agent_state['errors'][-20:]
    
    # Update stats
    agent_state['stats']['total_errors'] = agent_state['stats'].get('total_errors', 0) + 1


def record_agent_decision(decision: dict, provider: str = None):
    """
    Record an AI decision to agent state for OpenClaw-style tracking.
    
    Args:
        decision: The decision made by the AI (includes action, reasoning, etc.)
        provider: The AI provider that made the decision
    """
    now = datetime.now().isoformat()
    
    decision_record = {
        "timestamp": now,
        "decision": decision,
        "provider": provider
    }
    
    agent_state['last_decision'] = decision_record
    agent_state['decisions'].append(decision_record)
    
    # Keep only last 20 decisions
    if len(agent_state['decisions']) > 20:
        agent_state['decisions'] = agent_state['decisions'][-20:]
    
    # Update stats
    agent_state['stats']['total_decisions'] = agent_state['stats'].get('total_decisions', 0) + 1


def clear_agent_state():
    """Clear agent state for a fresh start."""
    agent_state['mode'] = 'manual'
    agent_state['enabled'] = False
    agent_state['current_goal'] = ''
    agent_state['current_task'] = ''
    agent_state['last_decision'] = None
    agent_state['last_action'] = None
    agent_state['last_action_time'] = None
    agent_state['errors'] = []
    agent_state['actions'] = []
    agent_state['decisions'] = []
    agent_state['stats'] = {
        'total_actions': 0,
        'total_decisions': 0,
        'total_errors': 0,
    }

def configure_emulator_launch_ui(emulator, enabled: bool):
    """Apply UI launch preference to emulator implementations when supported."""
    if hasattr(emulator, 'set_auto_launch_ui'):
        emulator.set_auto_launch_ui(enabled)
    elif hasattr(emulator, 'auto_launch_ui'):
        emulator.auto_launch_ui = enabled

def sync_loaded_rom_state(emulator_type: str, rom_path: str, rom_name: Optional[str] = None):
    """Update shared game state after a ROM loads successfully."""
    emulator = emulators[emulator_type]
    frame_count = emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else 0

    update_game_state({
        "rom_loaded": True,
        "active_emulator": emulator_type,
        "rom_path": rom_path,
        "rom_name": rom_name or os.path.basename(rom_path),
        "frame_count": frame_count,
    })


def _live_emulation_loop():
    """Background loop that only updates frame count metadata.
    
    The SSE stream handles actual emulation ticking and rendering.
    This loop only updates game state metadata for non-streaming clients.
    """
    global emulation_loop_running
    logger.info("Starting background metadata update loop (SSE handles emulation)")
    while emulation_loop_running:
        try:
            current_state = get_game_state()
            active = current_state.get("active_emulator")
            if current_state.get("rom_loaded") and active in emulators:
                emulator = emulators[active]
                # Only update frame count metadata - DO NOT tick the emulator here
                # The SSE stream is responsible for ticking and rendering
                if hasattr(emulator, 'get_info'):
                    try:
                        info = emulator.get_info()
                        if isinstance(info, dict) and 'frame_count' in info:
                            update_game_state({"frame_count": info['frame_count']})
                    except Exception:
                        pass
                elif hasattr(emulator, 'get_frame_count'):
                    try:
                        update_game_state({"frame_count": emulator.get_frame_count()})
                    except Exception:
                        pass
            time.sleep(1/10)  # Update metadata at 10 Hz - low CPU usage

        except Exception as exc:
            logger.debug(f"Metadata update loop iteration failed: {exc}")
            time.sleep(0.5)
    logger.info("Background metadata update loop stopped")


def ensure_emulation_loop_running():
    """Start the background emulation loop if not already running."""
    global emulation_loop_running, emulation_loop_thread
    if emulation_loop_running and emulation_loop_thread and emulation_loop_thread.is_alive():
        return
    emulation_loop_running = True
    emulation_loop_thread = threading.Thread(target=_live_emulation_loop, daemon=True)
    emulation_loop_thread.start()

def read_emulator_memory_value(emulator, address: int, size: int = 1) -> Optional[int]:
    """Read emulator memory using the best available API across emulator modes."""
    try:
        if hasattr(emulator, 'get_memory'):
            raw_value = emulator.get_memory(address, size)
            if isinstance(raw_value, (bytes, bytearray)) and len(raw_value) >= size:
                return int.from_bytes(raw_value[:size], byteorder='big')

        if hasattr(emulator, 'pyboy') and emulator.pyboy:
            if size == 1:
                return emulator.pyboy.memory[address]
            return int.from_bytes(bytes(emulator.pyboy.memory[address:address + size]), byteorder='big')

        if hasattr(emulator, 'memory'):
            if size == 1:
                return emulator.memory[address]
            return int.from_bytes(bytes(emulator.memory[address:address + size]), byteorder='big')
    except Exception as exc:
        logger.debug(f"Memory read failed at {hex(address)}: {exc}")

    return None

def timeout_handler(timeout_seconds):
    """Decorator to add timeout handling to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def target():
                return func(*args, **kwargs)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(target)
                try:
                    return future.result(timeout=timeout_seconds)
                except FutureTimeoutError:
                    logger.error(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        return wrapper
    return decorator

def validate_string_input(value: str, field_name: str, min_length: int = 0, max_length: int = 1000,
                         allowed_chars: str = None, pattern: str = None) -> str:
    """Validate string input with comprehensive checks"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if len(value) < min_length:
        raise ValueError(f"{field_name} must be at least {min_length} characters long")

    if len(value) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters long")

    # Check for null bytes and other dangerous characters
    if '\x00' in value:
        raise ValueError(f"{field_name} contains null bytes")

    # Check for potential SQL injection patterns
    sql_patterns = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'exec(', 'union ', 'select ', 'insert ', 'update ', 'delete ']
    value_lower = value.lower()
    for sql_pattern in sql_patterns:
        if sql_pattern in value_lower:
            raise ValueError(f"{field_name} contains potentially dangerous characters")

    # Check for path traversal attempts
    if '../' in value or '..\\' in value:
        raise ValueError(f"{field_name} contains path traversal attempts")

    # Check for command injection patterns
    cmd_patterns = ['|', '&', ';', '`', '$(', '&&', '||', '>', '<', '>>']
    for cmd_pattern in cmd_patterns:
        if cmd_pattern in value:
            raise ValueError(f"{field_name} contains command injection patterns")

    # Custom character validation
    if allowed_chars:
        if not all(c in allowed_chars for c in value):
            raise ValueError(f"{field_name} contains invalid characters")

    # Regex pattern validation
    if pattern:
        import re
        if not re.match(pattern, value):
            raise ValueError(f"{field_name} does not match required pattern")

    return value.strip()

def validate_integer_input(value, field_name: str, min_value: int = None, max_value: int = None) -> int:
    """Validate integer input with range checks"""
    try:
        if isinstance(value, str):
            # Remove whitespace
            value = value.strip()
            # Check for octal/hex that could be dangerous
            if value.startswith(('0o', '0x', '0b')):
                raise ValueError(f"{field_name} cannot use numeric prefixes")
            int_value = int(value)
        elif isinstance(value, int):
            int_value = value
        else:
            raise ValueError(f"{field_name} must be an integer")
    except (ValueError, TypeError):
        raise ValueError(f"{field_name} must be a valid integer")

    if min_value is not None and int_value < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}")

    if max_value is not None and int_value > max_value:
        raise ValueError(f"{field_name} must be at most {max_value}")

    return int_value

def validate_file_upload(file_obj, field_name: str, allowed_extensions: list = None,
                        max_size: int = None, content_types: list = None) -> dict:
    """Validate file upload with comprehensive security checks"""
    if not hasattr(file_obj, 'filename') or not hasattr(file_obj, 'save'):
        raise ValueError(f"Invalid {field_name}: not a valid file object")

    filename = file_obj.filename
    if not filename or filename == '':
        raise ValueError(f"No {field_name} filename provided")

    # Check for dangerous filenames
    dangerous_patterns = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    for pattern in dangerous_patterns:
        if pattern in filename:
            raise ValueError(f"{field_name} filename contains dangerous characters")

    # Validate file extension
    if allowed_extensions:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in allowed_extensions:
            raise ValueError(f"Invalid {field_name} extension. Allowed: {allowed_extensions}")

    # Check file size if content length is available
    if max_size and hasattr(file_obj, 'content_length') and file_obj.content_length:
        if file_obj.content_length > max_size:
            raise ValueError(f"{field_name} too large. Maximum size: {max_size} bytes")

    # Additional security checks for file content
    if hasattr(file_obj, 'stream') and hasattr(file_obj.stream, 'read'):
        try:
            # Reset file pointer and read first few bytes for magic number detection
            file_obj.stream.seek(0)
            file_header = file_obj.stream.read(1024)  # Read first 1KB
            file_obj.stream.seek(0)  # Reset pointer

            # Check for potentially dangerous file types by magic numbers
            dangerous_magic_numbers = {
                b'\x7fELF': 'ELF executable',
                b'MZ': 'Windows executable',
                b'#!': 'Script file',
                b'<html': 'HTML file',
                b'<?xml': 'XML file',
                b'%PDF': 'PDF file',
                b'\x1f\x8b': 'GZIP file',
                b'PK\x03\x04': 'ZIP file'
            }

            for magic, desc in dangerous_magic_numbers.items():
                if file_header.startswith(magic):
                    # Only allow if explicitly permitted
                    if content_types is None or desc not in content_types:
                        raise ValueError(f"{field_name} appears to be a {desc}, which is not allowed")

        except Exception as e:
            logger.warning(f"Could not validate {field_name} content: {e}")

    return {
        'filename': filename,
        'extension': os.path.splitext(filename)[1].lower(),
        'size': getattr(file_obj, 'content_length', 0)
    }

def validate_json_data(data: dict, required_fields: list = None, optional_fields: list = None,
                     field_validators: dict = None) -> dict:
    """Validate JSON data structure and fields"""
    if not isinstance(data, dict):
        raise ValueError("Request data must be a JSON object")

    validated_data = {}

    # Check required fields
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate all fields
    for field_name, field_value in data.items():
        # Skip if field is not expected
        if required_fields and field_name not in required_fields:
            if optional_fields and field_name not in optional_fields:
                logger.warning(f"Unexpected field in request: {field_name}")
                continue

        # Apply field-specific validation
        if field_validators and field_name in field_validators:
            validator_func = field_validators[field_name]
            try:
                validated_data[field_name] = validator_func(field_value)
            except ValueError as e:
                raise ValueError(f"Invalid {field_name}: {str(e)}")
        else:
            validated_data[field_name] = field_value

    return validated_data

def sanitize_filename(filename: str) -> str:
    """Enhanced filename sanitization to prevent directory traversal and other attacks"""
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")

    # Remove path components
    filename = os.path.basename(filename)

    # Remove dangerous characters and patterns
    dangerous_patterns = [
        '..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00',
        # Windows reserved characters and patterns
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]

    # Remove dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')

    # Remove dangerous patterns (case insensitive)
    filename_lower = filename.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in filename_lower:
            filename = filename.replace(pattern, '_', 1)

    # Remove leading/trailing whitespace, dots, and control characters
    filename = ''.join(char for char in filename if ord(char) >= 32 or char.isspace())
    filename = filename.strip('. ')

    # Ensure filename is not empty and not just dots
    if not filename or filename.replace('.', '').strip() == '':
        filename = "uploaded_file"

    # Limit length properly
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_len = 255 - len(ext)
        if max_name_len > 0:
            filename = name[:max_name_len] + ext
        else:
            filename = "uploaded_file" + ext[:10]

    # Additional security: ensure no null bytes or control sequences
    filename = filename.replace('\x00', '')

    return filename

def secure_file_operation(operation_func, *args, **kwargs):
    """Execute file operations with proper error handling and security"""
    try:
        return operation_func(*args, **kwargs)
    except (OSError, IOError) as e:
        logger.error(f"File operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file operation: {e}")
        raise

def create_secure_temp_file(suffix: str = '', prefix: str = 'temp_', directory: str = None) -> str:
    """Create a secure temporary file with proper permissions"""
    import tempfile

    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)

    try:
        # Set secure permissions (read/write for owner only)
        os.chmod(temp_path, 0o600)

        # Close the file descriptor
        os.close(fd)

        return temp_path
    except Exception as e:
        # Clean up on error
        try:
            os.close(fd)
            os.unlink(temp_path)
        except:
            pass
        raise

def secure_file_copy(src_path: str, dst_path: str, chunk_size: int = 8192) -> bool:
    """Securely copy a file with proper validation and error handling"""
    try:
        # Validate source file
        if not os.path.isfile(src_path):
            raise ValueError(f"Source file does not exist: {src_path}")

        # Check file size
        file_size = os.path.getsize(src_path)
        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max: {MAX_UPLOAD_SIZE})")

        # Create destination directory if it doesn't exist
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, mode=0o700)

        # Copy file in chunks to prevent memory issues
        with open(src_path, 'rb') as src_file:
            with open(dst_path, 'wb') as dst_file:
                while True:
                    chunk = src_file.read(chunk_size)
                    if not chunk:
                        break
                    dst_file.write(chunk)

        # Set secure permissions
        os.chmod(dst_path, 0o600)

        return True

    except Exception as e:
        logger.error(f"Secure file copy failed: {e}")
        # Clean up destination file if copy failed
        try:
            if os.path.exists(dst_path):
                os.unlink(dst_path)
        except:
            pass
        return False

def secure_file_delete(file_path: str) -> bool:
    """Securely delete a file with proper error handling"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
        return True
    except Exception as e:
        logger.error(f"Secure file delete failed: {e}")
        return False

# Rate limiting implementation
class RateLimiter:
    """Simple in-memory rate limiter"""
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {client_ip: [(timestamp, request_count)]}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if client is allowed to make a request"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Clean up old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (timestamp, count) for timestamp, count in self.requests[client_ip]
                if timestamp > window_start
            ]

        # Initialize if not exists
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Count requests in the current window
        total_requests = sum(count for _, count in self.requests[client_ip])

        if total_requests >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False

        # Add current request
        self.requests[client_ip].append((current_time, 1))

        return True

# Initialize rate limiter
rate_limiter = RateLimiter(RATE_LIMIT)

def get_client_ip() -> str:
    """Get client IP address, considering proxies"""
    # Check for forwarded IP (behind proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Get the first IP in the forwarded chain
        return forwarded_for.split(',')[0].strip()

    # Check for real IP (behind proxy)
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Fall back to direct IP
    return request.remote_addr or 'unknown'

def rate_limit_middleware():
    """Rate limiting middleware"""
    # Skip rate limiting for local development
    if DEBUG:
        return

    # Get client IP
    client_ip = get_client_ip()

    # Check if request is allowed
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return jsonify({
            "error": "Rate limit exceeded. Please try again later.",
            "retry_after": 60
        }), 429

def numpy_to_base64_image(np_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded JPEG image with performance optimizations"""
    try:
        # Quick validation
        if np_array is None or np_array.size == 0:
            logger.error("Invalid numpy array: None or empty")
            return ""

        start_time = time.time()

        # Create a view instead of copy if possible to save memory
        if np_array.base is not None:
            np_array = np_array.view()
        else:
            np_array = np_array.copy()

        # Optimized format handling
        if len(np_array.shape) == 3:
            if np_array.shape[2] == 4:
                # Fast RGBA to RGB using numpy slicing (no copy)
                np_array = np_array[:, :, :3]
            elif np_array.shape[2] != 3:
                logger.error(f"Unsupported channels: {np_array.shape[2]}")
                return ""
        elif len(np_array.shape) == 2:
            # Fast grayscale to RGB using numpy stacking
            np_array = np.expand_dims(np_array, axis=-1)
            np_array = np.repeat(np_array, 3, axis=-1)
        else:
            logger.error(f"Unsupported array shape: {np_array.shape}")
            return ""

        # Optimized data type conversion
        if np_array.dtype != np.uint8:
            if np_array.dtype in [np.float32, np.float64]:
                # Fast float to uint8 conversion
                np_array = np.multiply(np_array, 255, out=np_array, casting='unsafe')
            np_array = np.clip(np_array, 0, 255, out=np_array)
            np_array = np_array.astype(np.uint8, copy=False)

        # Use the fastest available encoding method
        img_bytes = None
        encoding_method = "unknown"

        try:
            # Priority 1: OpenCV (fastest for JPEG encoding)
            if CV2_AVAILABLE:
                success, encoded_img = cv2.imencode('.jpg', np_array, [
                    cv2.IMWRITE_JPEG_QUALITY, 75,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ])
                if success:
                    img_bytes = encoded_img.tobytes()
                    encoding_method = "opencv"

            # Priority 2: PyTorch GPU acceleration
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # Convert to tensor and use GPU for processing
                    tensor = torch.from_numpy(np_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    tensor = tensor.cuda()

                    # Use torchvision for JPEG encoding (if available)
                    try:
                        import torchvision.io
                        img_bytes = torchvision.io.encode_jpeg(tensor, quality=75)
                        encoding_method = "torch_gpu"
                    except ImportError:
                        # Fallback to CPU processing
                        tensor = tensor.cpu()
                        np_array = tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
                        np_array = np_array.astype(np.uint8)
                except Exception as torch_error:
                    logger.debug(f"GPU encoding failed, falling back to CPU: {torch_error}")

            # Priority 3: Optimized PIL processing
            if img_bytes is None and PIL_AVAILABLE:
                # Use contiguous array for better PIL performance
                if not np_array.flags['C_CONTIGUOUS']:
                    np_array = np.ascontiguousarray(np_array)

                image = Image.fromarray(np_array, mode='RGB')

                # Use optimized JPEG settings
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=75, optimize=False, progressive=False)
                img_bytes = img_buffer.getvalue()
                encoding_method = "pil"

            # Fallback: raw bytes (no encoding)
            if img_bytes is None:
                img_bytes = np_array.tobytes()
                encoding_method = "raw"

            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            processing_time = time.time() - start_time
            logger.debug(f"Image conversion: {encoding_method}, {len(img_bytes)} bytes, {processing_time:.3f}s")

            return img_base64

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return ""

    except Exception as e:
        logger.error(f"Error in numpy_to_base64_image: {e}")
        return ""


# Global performance monitoring - reduced limits to save RAM
performance_monitor = {
    'frame_times': [],
    'encoding_times': [],
    'fps_history': [],
    'last_fps_update': time.time(),
    'current_fps': 0,
    'adaptive_fps_target': 60,
    'min_fps': 15,
    'max_fps': 120,
    # ccboy-inspired: decouple screen capture from emulator ticking
    # Screen is captured every N ticks (1 = every frame, 2 = every other frame, etc.)
    # This reduces CPU load while keeping emulator responsive
    'screen_capture_divider': 2,  # Capture at half the tick rate
    'last_screen_capture_frame': 0,
    'cached_screen_base64': None,  # Cache for when we skip capture
}

def update_performance_metrics(encoding_time: float, frame_time: float):
    """Update performance monitoring metrics - with memory-efficient limits"""
    current_time = time.time()

    # Track encoding performance - limit to 30 entries to save RAM
    performance_monitor['encoding_times'].append(encoding_time)
    if len(performance_monitor['encoding_times']) > 30:
        performance_monitor['encoding_times'].pop(0)

    # Track frame timing - limit to 20 entries to save RAM
    performance_monitor['frame_times'].append(frame_time)
    if len(performance_monitor['frame_times']) > 20:
        performance_monitor['frame_times'].pop(0)

    # Update FPS calculation every second
    if current_time - performance_monitor['last_fps_update'] >= 1.0:
        if len(performance_monitor['frame_times']) > 0:
            avg_frame_time = sum(performance_monitor['frame_times']) / len(performance_monitor['frame_times'])
            if avg_frame_time > 0:
                performance_monitor['current_fps'] = 1.0 / avg_frame_time

        # Adaptive FPS adjustment based on performance
        avg_encoding_time = sum(performance_monitor['encoding_times']) / len(performance_monitor['encoding_times']) if performance_monitor['encoding_times'] else 0.01

        if avg_encoding_time > 0.016:  # > 16ms encoding time (slower than 60 FPS)
            performance_monitor['adaptive_fps_target'] = max(
                performance_monitor['min_fps'],
                performance_monitor['adaptive_fps_target'] - 5
            )
        elif avg_encoding_time < 0.008 and performance_monitor['current_fps'] < 60:  # < 8ms encoding time
            performance_monitor['adaptive_fps_target'] = min(
                performance_monitor['max_fps'],
                performance_monitor['adaptive_fps_target'] + 5
            )

        performance_monitor['last_fps_update'] = current_time

def get_performance_stats() -> dict:
    """Get current performance statistics"""
    return {
        'current_fps': performance_monitor['current_fps'],
        'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
        'avg_encoding_time': sum(performance_monitor['encoding_times']) / len(performance_monitor['encoding_times']) if performance_monitor['encoding_times'] else 0,
        'avg_frame_time': sum(performance_monitor['frame_times']) / len(performance_monitor['frame_times']) if performance_monitor['frame_times'] else 0,
        'cv2_available': CV2_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'torch_cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
    }


# Global error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors"""
    logger.warning(f"Bad request: {error}")
    return jsonify({
        "error": "Bad request",
        "message": str(error),
        "status": 400
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors"""
    logger.warning(f"Endpoint not found: {request.path}")
    return jsonify({
        "error": "Endpoint not found",
        "message": f"The requested URL {request.path} was not found on this server",
        "status": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 Method Not Allowed errors"""
    logger.warning(f"Method not allowed: {request.method} {request.path}")
    return jsonify({
        "error": "Method not allowed",
        "message": f"Method {request.method} not allowed for endpoint {request.path}",
        "status": 405
    }), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle 413 Request Entity Too Large errors"""
    logger.warning(f"Request too large: {error}")
    return jsonify({
        "error": "Request too large",
        "message": f"File size exceeds maximum allowed size of {MAX_ROM_SIZE // (1024*1024)}MB",
        "status": 413
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server Error"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred on the server",
        "status": 500
    }), 500

@app.before_request
def log_request_info():
    """Log information about each request"""
    logger.debug(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    """Log information about each response"""
    logger.debug(f"Response: {response.status_code} for {request.method} {request.path}")
    return response

# /health has been extracted to backend/routes/health.py — see
# register_health_routes() in routes/health.py (registered via routes/__init__.py).

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get comprehensive status of the server"""
    status = get_game_state()
    status['ai_providers'] = ai_provider_manager.get_provider_status()

    # Add resource manager stats if available
    if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
        status['resource_manager'] = resource_manager.get_resource_stats()

    # Add secure config stats if available
    if SECURE_CONFIG_AVAILABLE:
        status['configuration'] = secure_config.get_safe_config()

    return jsonify(status), 200


# ============================================================================
# Routes below this line have been extracted to backend/routes/blueprints.
# They are registered via backend.routes.register_all() above. The legacy
# handler code has been removed; the URLs and JSON shapes are unchanged.
#
#   /api/upload-rom, /api/load_rom, /api/rom/load,
#   /api/game/state, /api/party, /api/inventory, /api/memory/watch
#     -> routes/rom.py (register_rom_routes)
#
#   /api/spatial/position, /api/spatial/minimap, /api/spatial/npcs,
#   /api/agent/strategy, /api/spatial/strategy
#     -> routes/spatial.py (register_spatial_routes)
#
#   /api/agent/state, /api/agent/status, /api/agent/goal, /api/agent/chat,
#   /api/agent/errors, /api/agent/actions, /api/agent/context,
#   /api/agent/mode (GET/POST), /api/agent/act,
#   /api/agent/dialogue, /api/agent/menu
#     -> routes/agent.py (register_agent_routes)
#
# See CHANGELOG.md "Stage 3 continued (round 2)" for details.
# ============================================================================



def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    
    # Stop WebSocket server
    stop_websocket_server()
    
    cleanup_server_resources()
    logger.info("Server shutdown complete")
    exit(0)

def main():
    """Main server function"""
    logger.info("=== STARTING AI GAME SERVER ===")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize enhanced resource manager
    if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
        try:
            resource_manager.start()
            logger.info("[OK] Enhanced resource manager initialized")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize enhanced resource manager: {e}")
    else:
        logger.info("[INFO] Enhanced resource manager not available")

    # Validate configuration
    if SECURE_CONFIG_AVAILABLE:
        try:
            validation = secure_config.validate_environment_variables()
            logger.info(f"[CONFIG] Configuration validation completed")

            if not validation['valid']:
                logger.error("[CONFIG ERROR] Critical configuration issues found:")
                for missing in validation['missing_required']:
                    logger.error(f"  - Missing required: {missing}")
            else:
                logger.info("[CONFIG] All required configuration validated successfully")

            if validation['warnings']:
                logger.warning("[CONFIG WARNINGS]:")
                for warning in validation['warnings']:
                    logger.warning(f"  - {warning}")

            logger.info(f"[CONFIG] {validation['api_keys_configured']} API key(s) configured")
            logger.info(f"[CONFIG] Service URLs: {secure_config.get_safe_config()['services']}")
        except Exception as e:
            logger.error(f"[CONFIG ERROR] Configuration validation failed: {e}")
    else:
        logger.info("[CONFIG] Secure configuration manager not available")

    # Check PyBoy availability
    try:
        from pyboy import PyBoy
        logger.info("[OK] PyBoy is available")
    except ImportError:
        logger.error("[ERROR] PyBoy is NOT available - install with 'pip install pyboy'")

    # Check SDL2 availability
    try:
        import sdl2
        logger.info("[OK] SDL2 is available")
    except ImportError:
        logger.warning("[WARN] SDL2 is not available - UI may not work")

    logger.info(f"Available AI providers: {ai_provider_manager.get_available_providers()}")

    if not ai_provider_manager.get_available_providers():
        logger.warning("[WARNING] No AI providers are available. AI features will be limited.")
        logger.info("To enable AI features, set the appropriate environment variables:")
        logger.info("  - GEMINI_API_KEY")
        logger.info("  - OPENROUTER_API_KEY")
        logger.info("  - NVIDIA_API_KEY (optional: NVIDIA_MODEL)")
        logger.info("  - OPENAI_API_KEY (optional: OPENAI_ENDPOINT for local providers)")
        logger.info("  - LM_STUDIO_URL (for local LM Studio instance)")
        logger.info("  - OLLAMA_URL (for local Ollama instance)")
        logger.info("Note: For local providers like LM Studio, you may not need an API key.")
    else:
        logger.info(f"[SUCCESS] {len(ai_provider_manager.get_available_providers())} AI provider(s) are ready for use:")
        for provider_name in ai_provider_manager.get_available_providers():
            logger.info(f"  - {provider_name}")

    # Start WebSocket server for streaming
    logger.info("[WS] Starting WebSocket streaming server...")
    start_websocket_server()
    logger.info(f"[WS] WebSocket server started on ws://localhost:{WS_PORT}/api/ws/stream")

    # ------------------------------------------------------------------
    # Wire agent_features (sessions, memory, events, telemetry, collision)
    # ------------------------------------------------------------------
    try:
        from backend.agent_features import register_all as register_agent_features

        def _active_emulator_ref():
            try:
                _state = get_game_state()
                _active = _state.get("active_emulator")
                if _active and _active in emulators:
                    return emulators[_active]
            except Exception:  # noqa: BLE001
                return None
            return None

        counts = register_agent_features(
            app,
            emulator_ref_getter=_active_emulator_ref,
        )
        logger.info(
            f"[agent_features] Registered routes — sessions:{counts.get('sessions',0)} "
            f"memory:{counts.get('memory',0)} events:{counts.get('events',0)} "
            f"telemetry:{counts.get('telemetry',0)} collision:{counts.get('collision',0)}"
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[agent_features] Failed to register: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Wire refactored route blueprints (config, save_load, ui, ws, tetris, vision)
    # ------------------------------------------------------------------
    try:
        from backend.routes import register_all as register_route_blueprints
        from backend.routes.ws import WebSocketRunner

        # Build a WebSocketRunner from the (currently-undefined) ws globals.
        # If they're not defined (most common case), the blueprint uses a stub.
        def _ws_is_running():
            return bool(getattr(sys.modules[__name__], "ws_server_running", False))

        def _ws_port():
            return int(getattr(sys.modules[__name__], "WS_PORT", 5003) or 5003)

        def _ws_clients():
            ws_clients = getattr(sys.modules[__name__], "ws_clients", None)
            if ws_clients is None:
                return 0
            try:
                return len(ws_clients)
            except TypeError:
                return 0

        def _ws_start():
            fn = getattr(sys.modules[__name__], "start_websocket_server", None)
            if fn is None:
                raise RuntimeError(
                    "WebSocket server start function not available in this build"
                )
            return fn()

        def _ws_stop():
            fn = getattr(sys.modules[__name__], "stop_websocket_server", None)
            if fn is None:
                return None
            return fn()

        ws_runner = WebSocketRunner(
            is_running=_ws_is_running,
            get_port=_ws_port,
            get_clients=_ws_clients,
            start_fn=_ws_start,
            stop_fn=_ws_stop,
        )

        counts = register_route_blueprints(
            app,
            emulators_getter=lambda: emulators,
            game_state_getter=get_game_state,
            ai_provider_manager=ai_provider_manager,
            secure_config=secure_config if SECURE_CONFIG_AVAILABLE else None,
            secure_config_available=SECURE_CONFIG_AVAILABLE,
            host=HOST,
            port=PORT,
            debug=DEBUG,
            saved_states=saved_states,
            websocket_runner=ws_runner,
            health_server_start_time_getter=lambda: SERVER_START_TIME,
            health_pyboy_available=PYBOY_AVAILABLE,
            health_mcp_available=MCP_AVAILABLE,
            health_get_memory_usage=_get_memory_usage,
            health_component_health=component_health,
            health_ws_status_getter=lambda: (
                bool(ws_server_running) if 'ws_server_running' in globals() else False,
                int(WS_PORT) if 'WS_PORT' in globals() else 5003,
                len(ws_clients) if 'ws_clients' in globals() else 0,
            ),
            health_get_performance_stats=get_performance_stats,
            health_agent_state_getter=lambda: agent_state,
            ai_models_get_model_discovery=get_model_discovery,
            ai_models_openclaw_endpoint_getter=lambda: app.config.get(
                'OPENCLAW_ENDPOINT', 'http://localhost:18789'
            ),
            ai_runtime_state_getter=lambda: ai_runtime_state,
            ai_runtime_state_setter=_set_ai_runtime_state,
            ai_runtime_openclaw_endpoint_setter=lambda v: app.config.__setitem__('OPENCLAW_ENDPOINT', v),
            screen_numpy_to_base64=numpy_to_base64_image,
            screen_update_performance_metrics=update_performance_metrics,
            screen_performance_monitor=performance_monitor,
            screen_get_memory_usage=_get_memory_usage,
            screen_use_multi_process=USE_MULTI_PROCESS,
            screen_optimization_system_manager=optimization_system_manager,
            screen_optimization_system_available=OPTIMIZATION_SYSTEM_AVAILABLE,
            input_update_game_state=update_game_state,
            input_get_action_history=get_action_history,
            input_add_to_action_history=add_to_action_history,
            input_record_agent_action=record_agent_action,
            input_record_agent_error=record_agent_error,
            input_record_agent_decision=record_agent_decision,
            input_validate_string_input=validate_string_input,
            input_validate_integer_input=validate_integer_input,
            input_validate_json_data=validate_json_data,
            input_timeout_handler=timeout_handler,
            input_ai_request_timeout=AI_REQUEST_TIMEOUT,
        )
        logger.info(
            f"[routes] Registered routes — "
            f"config:{counts.get('config',0)} health:{counts.get('health',0)} "
            f"ai_models:{counts.get('ai_models',0)} "
            f"ai_runtime:{counts.get('ai_runtime',0)} "
            f"save_load:{counts.get('save_load',0)} "
            f"screen:{counts.get('screen',0)} "
            f"input:{counts.get('input',0)} "
            f"ui:{counts.get('ui',0)} ws:{counts.get('ws',0)} "
            f"tetris:{counts.get('tetris',0)} vision:{counts.get('vision',0)}"
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[routes] Failed to register: {e}", exc_info=True)

    try:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
    finally:
        # Stop WebSocket server on exit
        stop_websocket_server()
        cleanup_server_resources()

if __name__ == "__main__":
    main()
