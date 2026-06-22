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

@app.route('/api/upload-rom', methods=['POST'])
def upload_rom():
    """Upload a ROM file and load it into the specified emulator with enhanced security"""
    logger.info("=== ROM UPLOAD REQUEST RECEIVED ===")

    try:
        # Check for ROM file in both possible field names for compatibility
        file = None
        if 'rom_file' in request.files:
            file = request.files['rom_file']
        elif 'rom' in request.files:
            file = request.files['rom']
        else:
            return jsonify({"error": "No ROM file provided"}), 400
        emulator_type = request.form.get('emulator_type', 'gb')
        if emulator_type == 'gb':
            emulator_type = 'pyboy'
        launch_ui = request.form.get('launch_ui', 'false')

        logger.info(f"File received: {file.filename}")
        logger.info(f"Emulator type: {emulator_type}")
        logger.info(f"Launch UI: {launch_ui}")
        logger.info(f"Available emulators: {list(emulators.keys())}")

        # Additional ROM validation
        if file.filename == '' or file.filename is None:
            return jsonify({"error": "No filename provided"}), 400

        # Check file size before processing
        if hasattr(file, 'content_length') and file.content_length > MAX_ROM_SIZE:
            return jsonify({
                "error": f"File size exceeds maximum allowed size of {MAX_ROM_SIZE // (1024*1024)}MB"
            }), 400

        # Validate file upload with comprehensive security checks
        try:
            file_info = validate_file_upload(
                file,
                "ROM file",
                allowed_extensions=ALLOWED_ROM_EXTENSIONS,
                max_size=MAX_ROM_SIZE
            )
            logger.info(f"File validation passed: {file_info}")
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return jsonify({"error": str(e)}), 400

        # Validate emulator type
        try:
            emulator_type = validate_string_input(
                emulator_type,
                "emulator_type",
                min_length=2,
                max_length=20,
                allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789-"
            )
        except ValueError as e:
            return jsonify({"error": f"Invalid emulator type: {str(e)}"}), 400

        # Validate launch_ui parameter
        try:
            launch_ui = validate_string_input(
                launch_ui,
                "launch_ui",
                min_length=2,
                max_length=10,
                pattern=r'^(true|false)$'
            )
            launch_ui = launch_ui.lower() == 'true'
        except ValueError as e:
            return jsonify({"error": f"Invalid launch_ui parameter: {str(e)}"}), 400

        # Sanitize filename and create secure temporary file
        safe_filename = sanitize_filename(file_info['filename'])

        # Use secure temporary file creation.
        # Important: create the temp path, close the handle, then save into it.
        # Saving into a still-open NamedTemporaryFile can result in empty files on some setups.
        fd, temp_rom_path = tempfile.mkstemp(suffix=file_info['extension'])
        os.close(fd)
        file.stream.seek(0)
        with open(temp_rom_path, 'wb') as rom_out:
            shutil.copyfileobj(file.stream, rom_out)
        file.stream.seek(0)
        os.chmod(temp_rom_path, 0o600)

        logger.info(f"ROM saved to temporary path: {temp_rom_path}")

        # Validate emulator type with fallback
        emulator_type_mapping = {
            'gb': 'pyboy',
            'gba': 'pygba',
            'pyboy': 'pyboy',
            'pygba': 'pygba'
        }

        # Map the emulator type with validation
        mapped_emulator_type = emulator_type_mapping.get(emulator_type.lower())
        if not mapped_emulator_type or mapped_emulator_type not in emulators:
            logger.error(f"Invalid emulator type: {emulator_type}")
            return jsonify({
                "error": f"Invalid emulator type. Available: {list(emulator_type_mapping.keys())}"
            }), 400

        # Use the mapped emulator type
        emulator_type = mapped_emulator_type
        emulator_instance = emulators[emulator_type]
        configure_emulator_launch_ui(emulator_instance, launch_ui)

        logger.info(f"Loading ROM into {emulator_type} emulator...")

        # Verify ROM file integrity
        try:
            temp_file_size = os.path.getsize(temp_rom_path)
            if temp_file_size < 0x150:
                logger.error(f"Uploaded ROM temp file is too small: {temp_file_size} bytes")
                return jsonify({"error": f"File too small to be a valid ROM ({temp_file_size} bytes)"}), 400

            with open(temp_rom_path, 'rb') as rom_check:
                file_header = rom_check.read(512)

            # Basic ROM validation - check for Game Boy header pattern
            if len(file_header) < 0x150:  # Minimum ROM size
                logger.error(f"Uploaded ROM header read was too short: {len(file_header)} bytes")
                return jsonify({"error": f"Could not read enough ROM header bytes ({len(file_header)})"}), 400

            # Check for Nintendo logo pattern (bytes 0x104-0x133)
            nintendo_logo = [
                0xCE, 0xED, 0x66, 0x66, 0xCC, 0x0D, 0x00, 0x0B,
                0x03, 0x73, 0x00, 0x83, 0x00, 0x0C, 0x00, 0x0D,
                0x00, 0x08, 0x11, 0x1F, 0x88, 0x89, 0x00, 0x0E,
                0xDC, 0xCC, 0x6E, 0xE6, 0xDD, 0xDD, 0xD9, 0x99,
                0xBB, 0xBB, 0x67, 0x63, 0x6E, 0x0E, 0xEC, 0xCC,
                0xDD, 0xDC, 0x99, 0x9F, 0xBB, 0xB9, 0x33, 0x3E
            ]

            rom_logo = list(file_header[0x104:0x134])
            logo_matches = sum(1 for i in range(len(nintendo_logo)) if i < len(rom_logo) and rom_logo[i] == nintendo_logo[i])

            if logo_matches < 20:  # Allow some variation for homebrew/modified ROMs
                logger.warning(f"ROM logo validation weak ({logo_matches}/{len(nintendo_logo)} matches), but continuing")

        except Exception as validation_error:
            logger.warning(f"ROM header validation failed: {validation_error}")
            # Continue anyway for maximum compatibility

        # Load ROM into emulator with enhanced error handling
        logger.info(f"Loading ROM into {emulator_type} emulator...")

        try:
            success = emulator_instance.load_rom(temp_rom_path)

            if success:
                # Initialize emulator and run a few frames to ensure it's working
                emulator = emulator_instance

            # Test emulator functionality with proper error handling
                test_success = False
                test_error = None

                if hasattr(emulator, 'pyboy') and emulator.pyboy:
                    try:
                        # Run a short warm-up with a rendered final frame so the
                        # live WebUI has a meaningful framebuffer immediately.
                        for i in range(120):
                            render_this_frame = (i == 119)
                            result = emulator.pyboy.tick(1, render_this_frame)
                            if result is False and i > 0:
                                logger.warning(f"Warm-up frame {i} returned False")
                        test_success = True
                        logger.info("Emulator warm-up frames completed successfully")
                    except Exception as tick_error:
                        test_error = str(tick_error)
                        logger.error(f"Emulator warm-up frames failed: {tick_error}")
                else:
                    test_error = "Emulator has no pyboy attribute"
                    logger.warning("Emulator has no pyboy attribute")

                if not test_success:
                    logger.warning("Emulator functionality test failed, but continuing")

                # Get UI status
                ui_status = emulator.get_ui_status() if hasattr(emulator, 'get_ui_status') else {"running": False}

                logger.info(f"UI status: {ui_status}")
                rom_name = file.filename or safe_filename
                sync_loaded_rom_state(emulator_type, temp_rom_path, rom_name=rom_name)
                ensure_emulation_loop_running()

                # Prepare comprehensive response
                response_data = {
                    "message": "ROM loaded successfully",
                    "rom_name": rom_name,
                    "original_filename": file.filename,
                    "emulator_type": emulator_type,
                    "rom_size": os.path.getsize(temp_rom_path),
                    "ui_launched": ui_status.get("running", False),
                    "ui_status": ui_status,
                    "test_success": test_success,
                    "test_error": test_error,
                    "frame_count": emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else 0,
                    "temp_path": temp_rom_path
                }

                # Check if UI auto-launch was attempted but failed
                auto_launch_enabled = ui_status.get("auto_launch_enabled", True)
                ui_process_running = ui_status.get("running", False)

                if not ui_process_running and auto_launch_enabled and launch_ui:
                    logger.warning("UI process failed to launch automatically")
                    # Add helpful information for manual UI launch
                    response_data["ui_help"] = {
                        "message": "Automatic UI launch failed. You can:",
                        "actions": [
                            "Try launching UI manually using the UI control panel",
                            "Check if PyBoy is properly installed: pip install pyboy",
                            "Verify SDL2 libraries are available on your system",
                            "Check the emulator logs for more details"
                        ]
                    }

                return jsonify(response_data), 200

            else:
                logger.error(f"Failed to load ROM: {temp_rom_path}")
                if os.path.exists(temp_rom_path):
                    os.unlink(temp_rom_path)
                return jsonify({
                    "error": "Failed to load ROM into emulator",
                    "details": "The emulator could not load the ROM file. This could be due to: - Corrupted ROM file - Incompatible ROM format - Emulator initialization failure",
                    "emulator_type": emulator_type
                }), 500

        except Exception as load_error:
            logger.error(f"Exception during ROM loading: {load_error}")
            if 'temp_rom_path' in locals() and os.path.exists(temp_rom_path):
                try:
                    os.unlink(temp_rom_path)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up temp file: {cleanup_error}")
            return jsonify({
                "error": "Exception during ROM loading",
                "details": str(load_error),
                "emulator_type": emulator_type
            }), 500

    except Exception as e:
        logger.error(f"Error uploading ROM: {e}", exc_info=True)
        # Clean up temp file if it exists
        if 'temp_rom_path' in locals() and os.path.exists(temp_rom_path):
            try:
                os.unlink(temp_rom_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temp file: {cleanup_error}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/api/load_rom', methods=['POST'])
def load_rom():
    """
    Legacy endpoint for backward compatibility with frontend services.
    Redirects to the main upload-rom endpoint.
    """
    logger.info("=== LEGACY LOAD_ROM REQUEST RECEIVED ===")

    try:
        # Check for ROM file in both possible field names
        if 'rom' in request.files or 'rom_file' in request.files:
            # Forward multipart uploads to the main upload endpoint.
            return upload_rom()

        data = request.get_json(silent=True) or {}
        if data:
            rom_path = data.get('path') or data.get('rom_path')
            emulator_type = data.get('emulator_type', 'gb')
            launch_ui = data.get('launch_ui', False)
            if emulator_type == 'gb':
                emulator_type = 'pyboy'
            if not rom_path:
                return jsonify({"error": "No ROM path provided"}), 400
            if emulator_type not in emulators:
                return jsonify({"error": f"Emulator {emulator_type} not found"}), 404
            emulator = emulators[emulator_type]
            if hasattr(emulator, 'set_auto_launch_ui'):
                emulator.set_auto_launch_ui(bool(launch_ui))
            result = emulator.load_rom(rom_path)
            if result:
                sync_loaded_rom_state(emulator_type, rom_path, os.path.basename(rom_path))
                return jsonify({"status": "success", "rom_loaded": True, "rom_name": os.path.basename(rom_path), "emulator": emulator_type}), 200
            return jsonify({"error": "Failed to load ROM"}), 500

        return jsonify({"error": "No ROM file provided"}), 400

    except Exception as e:
        logger.error(f"Error in legacy load_rom endpoint: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/api/game/state', methods=['GET'])
def api_game_state():
    state = get_game_state()
    return jsonify(state), 200

@app.route('/api/party', methods=['GET'])
def api_party():
    state = get_game_state()
    active = state.get('active_emulator')
    empty = {'party_count': 0, 'party': [], 'timestamp': datetime.now().isoformat()}
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty), 200
    try:
        emulator = emulators[active]
        party = []
        if hasattr(emulator, 'get_party_info'):
            try:
                raw_party = emulator.get_party_info() or []
            except Exception:
                raw_party = []
            for idx, mon in enumerate(raw_party, 1):
                mon = mon or {}
                party.append({
                    'slot': mon.get('slot', idx),
                    'species_id': mon.get('species_id'),
                    'species_name': mon.get('species_name'),
                    'level': mon.get('level'),
                    'hp': mon.get('hp'),
                    'max_hp': mon.get('max_hp'),
                    'status': mon.get('status'),
                    'status_text': mon.get('status_text'),
                    'type1': mon.get('type1'),
                    'type2': mon.get('type2'),
                    'moves': mon.get('moves') or [],
                    'ot_id': mon.get('ot_id'),
                    'hp_percent': mon.get('hp_percent')
                })
        return jsonify({'party_count': len(party), 'party': party, 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        return jsonify({**empty, 'error': str(e)}), 200

@app.route('/api/inventory', methods=['GET'])
def api_inventory():
    state = get_game_state()
    active = state.get('active_emulator')
    empty = {'money': 0, 'money_formatted': '¥0', 'item_count': 0, 'items': [], 'timestamp': datetime.now().isoformat()}
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty), 200
    try:
        emulator = emulators[active]
        items = []
        money = 0
        if hasattr(emulator, 'get_inventory_info'):
            try:
                inv = emulator.get_inventory_info() or {}
            except Exception:
                inv = {}
            money = inv.get('money', 0) or 0
            for idx, item in enumerate(inv.get('items') or [], 1):
                item = item or {}
                items.append({
                    'slot': item.get('slot', idx),
                    'id': item.get('id', 0),
                    'name': item.get('name', 'Unknown'),
                    'quantity': item.get('quantity', 0)
                })
        return jsonify({'money': money, 'money_formatted': f'¥{money:,}', 'item_count': len(items), 'items': items, 'timestamp': datetime.now().isoformat()}), 200
    except Exception as e:
        return jsonify({**empty, 'error': str(e)}), 200

@app.route('/api/memory/watch', methods=['GET'])
def api_memory_watch():
    state = get_game_state()
    active = state.get('active_emulator')
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify({'addresses': [], 'values': [], 'timestamp': datetime.now().isoformat()}), 200
    try:
        emulator = emulators[active]
        # minimal compatibility payload for frontend
        payload = {
            'addresses': [],
            'values': [],
            'timestamp': datetime.now().isoformat(),
            'frame_count': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else 0
        }
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({'addresses': [], 'values': [], 'timestamp': datetime.now().isoformat(), 'error': str(e)}), 200


# =========================================
# Server-Side Vision Analysis Endpoints
# =========================================
# These endpoints capture the screen and return STRUCTURED TEXT ANALYSIS,
# not just raw image data. This allows LM Studio / MCP agents to understand
# the screen even when their interface does not truly consume attached images.
#
# KEY DIFFERENCE:
# - get_screen/screenshot: Returns raw image bytes (for humans/visual display)
# - analyze_screen/describe_screen/ocr_screen: Returns text analysis (for AI agents)
# =========================================

# /api/vision/* endpoints have been extracted to backend/routes/vision.py
# and are registered via the routes blueprint at server startup.


# =========================================
# Spatial Endpoints (MCP/UI Contract)
# =========================================

@app.route('/api/spatial/position', methods=['GET'])
def api_spatial_position():
    """
    Get player position and map information.
    
    Response shape:
    {
        "x": number,          // Player X coordinate (0-255)
        "y": number,          // Player Y coordinate (0-255)
        "map_id": number,     // Current map ID (0-255)
        "map_name": string,   // Human-readable map name
        "timestamp": string,  // ISO timestamp
        "loaded": boolean     // True if ROM is loaded
    }
    
    Empty/loading response (no ROM):
    {
        "x": 0, "y": 0, "map_id": 0, "map_name": "none",
        "timestamp": "...", "loaded": false
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'x': 0, 'y': 0, 'map_id': 0, 'map_name': 'none',
        'timestamp': datetime.now().isoformat(),
        'loaded': False
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        if hasattr(emulator, 'get_position'):
            pos = emulator.get_position()
            pos['timestamp'] = datetime.now().isoformat()
            pos['loaded'] = True
            return jsonify(pos), 200
        
        # Fallback: try to read memory directly for Pokemon
        if hasattr(emulator, '_read_byte'):
            x = emulator._read_byte(0xD062)
            y = emulator._read_byte(0xD063)
            map_id = emulator._read_byte(0xD35E)
            return jsonify({
                'x': x, 'y': y, 'map_id': map_id, 'map_name': f'Map {map_id}',
                'timestamp': datetime.now().isoformat(),
                'loaded': True
            }), 200
        
        return jsonify(empty_response), 200
        
    except Exception as e:
        logger.debug(f"Error getting position: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/spatial/minimap', methods=['GET'])
def api_spatial_minimap():
    """
    Get minimap data for the current area.
    
    Response shape:
    {
        "width": number,      // Map width in tiles
        "height": number,     // Map height in tiles
        "tiles": number[][],  // 2D array of tile IDs (sparse)
        "player": {           // Player position on minimap
            "x": number,
            "y": number
        },
        "timestamp": string,
        "loaded": boolean
    }
    
    Empty/loading response:
    {
        "width": 0, "height": 0, "tiles": [],
        "player": {"x": 0, "y": 0},
        "timestamp": "...", "loaded": false
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'width': 0, 'height': 0, 'tiles': [],
        'player': {'x': 0, 'y': 0},
        'timestamp': datetime.now().isoformat(),
        'loaded': False
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Get player position
        player_pos = {'x': 0, 'y': 0}
        if hasattr(emulator, 'get_position'):
            pos = emulator.get_position()
            player_pos = {'x': pos.get('x', 0), 'y': pos.get('y', 0)}
        
        # For now, return sparse minimap data
        # Full tilemap reading would require more complex memory parsing
        # This provides a stable contract for the UI
        
        return jsonify({
            'width': 20,   # Default Game Boy screen width in tiles
            'height': 18,  # Default Game Boy screen height in tiles
            'tiles': [],   # Sparse - would need tilemap memory reading
            'player': player_pos,
            'timestamp': datetime.now().isoformat(),
            'loaded': True,
            'note': 'Sparse minimap - tile data not implemented'
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting minimap: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/spatial/npcs', methods=['GET'])
def api_spatial_npcs():
    """
    Get nearby NPC information.
    
    Response shape:
    {
        "npcs": [
            {
                "id": number,         // NPC sprite ID
                "name": string,       // NPC type name (if known)
                "x": number,          // X position
                "y": number,          // Y position
                "type": string,       // "npc", "trainer", "pokemon", etc.
            }
        ],
        "count": number,
        "timestamp": string,
        "loaded": boolean
    }
    
    Empty/loading response:
    {
        "npcs": [], "count": 0,
        "timestamp": "...", "loaded": false
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'npcs': [], 'count': 0,
        'timestamp': datetime.now().isoformat(),
        'loaded': False
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        npcs = []
        
        # Check for enemy Pokemon in battle
        if hasattr(emulator, 'get_battle_info'):
            battle = emulator.get_battle_info()
            if battle.get('in_battle') and battle.get('enemy'):
                enemy = battle['enemy']
                npcs.append({
                    'id': enemy.get('species_id', 0),
                    'name': enemy.get('species_name', 'Unknown'),
                    'x': -1,  # Battle position not applicable
                    'y': -1,
                    'type': 'enemy_pokemon',
                    'level': enemy.get('level', 0),
                    'hp_percent': enemy.get('hp_percent', 0),
                })
        
        # Note: Full NPC reading would require parsing sprite data
        # from memory addresses 0xC000-0xCFFF (OAM/sprite memory)
        
        return jsonify({
            'npcs': npcs,
            'count': len(npcs),
            'timestamp': datetime.now().isoformat(),
            'loaded': True,
            'note': 'Battle NPCs only - sprite memory reading not implemented'
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting NPCs: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/agent/strategy', methods=['GET'])
@app.route('/api/spatial/strategy', methods=['GET'])
def api_spatial_strategy():
    """
    Get strategic analysis and recommendations.
    
    Response shape:
    {
        "status": string,         // Current game status summary
        "health": {
            "party_healthy": boolean,
            "lowest_hp_percent": number,
            "needs_healing": boolean
        },
        "battle": {
            "in_battle": boolean,
            "recommendation": string  // "attack", "run", "catch", "heal"
        },
        "recommendations": string[],  // List of recommended actions
        "timestamp": string,
        "loaded": boolean
    }
    
    Empty/loading response:
    {
        "status": "no_rom", "health": {...}, "battle": {...},
        "recommendations": [], "timestamp": "...", "loaded": false
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'status': 'no_rom',
        'health': {
            'party_healthy': True,
            'lowest_hp_percent': 100,
            'needs_healing': False
        },
        'battle': {
            'in_battle': False,
            'recommendation': 'none'
        },
        'recommendations': [],
        'timestamp': datetime.now().isoformat(),
        'loaded': False
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Get party info
        party = []
        if hasattr(emulator, 'get_party_info'):
            party = emulator.get_party_info() or []
        
        # Calculate health metrics
        lowest_hp = 100
        needs_healing = False
        for mon in party:
            hp_pct = mon.get('hp_percent', 100)
            if hp_pct < lowest_hp:
                lowest_hp = hp_pct
            if hp_pct < 30:
                needs_healing = True
        
        party_healthy = not needs_healing
        
        # Get battle info
        battle_info = {'in_battle': False, 'recommendation': 'none'}
        battle_rec = 'none'
        
        if hasattr(emulator, 'get_battle_info'):
            battle = emulator.get_battle_info()
            battle_info['in_battle'] = battle.get('in_battle', False)
            
            if battle.get('in_battle'):
                enemy = battle.get('enemy', {})
                enemy_hp_pct = enemy.get('hp_percent', 100)
                
                # Simple strategy logic
                if lowest_hp < 20:
                    battle_rec = 'run'
                elif enemy_hp_pct < 30 and lowest_hp > 50:
                    battle_rec = 'catch'
                else:
                    battle_rec = 'attack'
                
                battle_info['recommendation'] = battle_rec
                battle_info['enemy'] = enemy
        
        # Build recommendations list
        recommendations = []
        
        if needs_healing:
            recommendations.append('Heal party at Pokemon Center')
        if battle_info['in_battle']:
            recommendations.append(f'Battle: {battle_rec}')
        if not party:
            recommendations.append('Get first Pokemon')
        
        # Get inventory for ball count
        inv = {}
        if hasattr(emulator, 'get_inventory_info'):
            inv = emulator.get_inventory_info() or {}
        
        money = inv.get('money', 0)
        if money > 0:
            recommendations.insert(0, f'Money: ¥{money:,}')
        
        # Build status summary
        if battle_info['in_battle']:
            status = f"In battle vs {battle_info.get('enemy', {}).get('species_name', 'unknown')}"
        elif needs_healing:
            status = f"Party needs healing ({lowest_hp:.0f}% HP)"
        elif party:
            status = f"Exploring with {len(party)} Pokemon"
        else:
            status = "Ready"
        
        return jsonify({
            'status': status,
            'health': {
                'party_healthy': party_healthy,
                'lowest_hp_percent': round(lowest_hp, 1),
                'needs_healing': needs_healing
            },
            'battle': battle_info,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'loaded': True
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting strategy: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/agent/mode', methods=['POST'])
def api_agent_mode_set():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'manual')
    if 'agent_state' in globals():
        agent_state['mode'] = mode
    return jsonify({'ok': True, 'mode': mode}), 200



# =========================================
# OpenClaw-Style Agent/Health/Status Endpoints
# =========================================

@app.route('/api/agent/state', methods=['GET'])
def api_agent_state():
    """
    Get comprehensive agent state (OpenClaw-style).
    
    Response shape:
    {
        "mode": "manual" | "autonomous" | "ai_assisted",
        "enabled": boolean,
        "current_goal": string,
        "current_task": string,
        "last_decision": {...} | null,
        "last_action": string | null,
        "last_action_time": string | null,
        "recent_errors": [...],  // Last 10 errors
        "recent_actions": [...],  // Last 20 actions
        "stats": {
            "total_actions": number,
            "total_decisions": number,
            "total_errors": number
        },
        "started_at": string | null,
        "timestamp": string
    }
    """
    now = datetime.now().isoformat()
    
    return jsonify({
        'mode': agent_state.get('mode', 'manual'),
        'enabled': agent_state.get('enabled', False),
        'current_goal': agent_state.get('current_goal', ''),
        'current_task': agent_state.get('current_task', ''),
        'last_decision': agent_state.get('last_decision'),
        'last_action': agent_state.get('last_action'),
        'last_action_time': agent_state.get('last_action_time'),
        'recent_errors': agent_state.get('errors', [])[-10:],
        'recent_actions': agent_state.get('actions', [])[-20:],
        'stats': agent_state.get('stats', {
            'total_actions': 0,
            'total_decisions': 0,
            'total_errors': 0,
        }),
        'started_at': agent_state.get('started_at'),
        'timestamp': now
    }), 200


@app.route('/api/agent/status', methods=['GET'])
def api_agent_status():
    """
    Get agent status summary (OpenClaw-style).
    
    Response shape:
    {
        "mode": "manual" | "autonomous" | "ai_assisted",
        "enabled": boolean,
        "goal": string,
        "last_action": string | null,
        "last_error": string | null,
        "action_count": number,
        "error_count": number,
        "timestamp": string
    }
    """
    errors = agent_state.get('errors', [])
    last_error = errors[-1] if errors else None
    
    return jsonify({
        'mode': agent_state.get('mode', 'manual'),
        'enabled': agent_state.get('enabled', False),
        'goal': agent_state.get('current_goal', ''),
        'last_action': agent_state.get('last_action'),
        'last_error': last_error,
        'action_count': agent_state.get('stats', {}).get('total_actions', 0),
        'error_count': agent_state.get('stats', {}).get('total_errors', 0),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/agent/goal', methods=['POST', 'GET'])
def api_agent_goal():
    """
    Set or get the current agent goal.
    
    POST Body: {"goal": "string", "task": "string (optional)"}
    
    GET Response: {"goal": string, "task": string}
    """
    if request.method == 'GET':
        return jsonify({
            'goal': agent_state.get('current_goal', ''),
            'task': agent_state.get('current_task', ''),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    data = request.get_json(silent=True) or {}
    goal = data.get('goal', '')
    task = data.get('task', '')
    
    agent_state['current_goal'] = goal
    agent_state['current_task'] = task
    
    logger.info(f"Agent goal set: {goal}, task: {task}")
    
    return jsonify({
        'ok': True,
        'goal': goal,
        'task': task,
        'timestamp': datetime.now().isoformat()
    }), 200


# Intent patterns for floating chat instruction parsing
INTENT_PATTERNS = {
    'goal': ['goal:', 'objective:', 'my goal is ', 'i want to ', 'go for '],
    'task': ['task:', 'do: ', 'please ', 'can you '],
    'query': ['?status', '?state', 'status', 'state:']
}

def parse_instruction(message: str) -> dict:
    """
    Parse message to detect instruction intent.
    
    Returns:
        {'intent': 'goal'|'task'|'query'|'chat', 'value': string}
    """
    if not message:
        return {'intent': 'chat', 'value': ''}
    
    lower = message.lower().strip()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if lower.startswith(pattern):
                return {
                    'intent': intent,
                    'value': message[len(pattern):].strip()
                }
    
    return {'intent': 'chat', 'value': message}


def format_agent_status_response() -> str:
    """Format agent status as a readable string for chat response"""
    mode = agent_state.get('mode', 'manual')
    goal = agent_state.get('current_goal', '(none)')
    task = agent_state.get('current_task', '(none)')
    last_action = agent_state.get('last_action', 'None')
    last_time = agent_state.get('last_action_time', 'Never')
    stats = agent_state.get('stats', {})
    total_actions = stats.get('total_actions', 0)
    
    response = f"Agent Status:\n"
    response += f"- Mode: {mode}\n"
    response += f"- Goal: {goal}\n"
    response += f"- Task: {task}\n"
    response += f"- Last action: {last_action}"
    if last_time != 'Never':
        response += f" ({last_time})"
    response += f"\n- Total actions: {total_actions}"
    
    return response


@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    """
    Agent-aware chat endpoint that can set goals/tasks via chat.
    
    This endpoint integrates the floating chat with the agent state system.
    Use prefixes to control agent behavior:
    - "goal: <text>" - Set the agent's current goal
    - "task: <text>" - Set the agent's current task
    - "?status" or "status" - Query agent status
    - Plain text - Regular chat with AI
    
    Request Body:
    {
        "message": "string (required)",
        "api_name": "string (optional, default: openclaw)",
        "api_key": "string (optional)",
        "api_endpoint": "string (optional)",
        "model": "string (optional)"
    }
    
    Response:
    {
        "ok": boolean,
        "intent_detected": "goal"|"task"|"query"|"chat",
        "goal_updated": string|null,
        "task_updated": string|null,
        "chat_response": "string",
        "agent_state": {...},
        "timestamp": "string"
    }
    """
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({
                "ok": False,
                "error": "No ROM loaded",
                "chat_response": None,
                "timestamp": datetime.now().isoformat()
            }), 400
        
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({
                "ok": False,
                "error": "Message is required",
                "chat_response": None,
                "timestamp": datetime.now().isoformat()
            }), 400
        
        # Parse intent from message
        parsed = parse_instruction(user_message)
        intent = parsed['intent']
        value = parsed['value']
        
        goal_updated = None
        task_updated = None
        chat_response = None
        
        # Handle intent
        if intent == 'goal' and value:
            # Set agent goal
            agent_state['current_goal'] = value
            logger.info(f"Agent goal set via chat: {value}")
            goal_updated = value
            chat_response = f"I've updated your goal to: {value}. What would you like me to help you with?"
        
        elif intent == 'task' and value:
            # Set agent task
            agent_state['current_task'] = value
            logger.info(f"Agent task set via chat: {value}")
            task_updated = value
            chat_response = f"I've set your current task to: {value}. I'll focus on this now."
        
        elif intent == 'query':
            # Return agent status
            chat_response = format_agent_status_response()
        
        else:
            # Regular chat - get AI response
            api_name = data.get('api_name', 'openclaw')
            api_key = data.get('api_key')
            api_endpoint = data.get('api_endpoint')
            model = data.get('model')
            
            # Get screen bytes
            current_state = get_game_state()
            emulator = emulators[current_state["active_emulator"]]
            img_bytes = emulator.get_screen_bytes()
            
            if len(img_bytes) == 0:
                return jsonify({
                    "ok": False,
                    "error": "Failed to capture screen",
                    "chat_response": None,
                    "timestamp": datetime.now().isoformat()
                }), 500
            
            # Build context with agent state
            context = {
                "current_goal": agent_state.get('current_goal', ''),
                "current_task": agent_state.get('current_task', ''),
                "action_history": get_action_history()[-10:],
                "game_type": current_state["active_emulator"].upper()
            }
            
            # Get AI response using the provider
            try:
                if api_name in ai_apis:
                    ai_connector = ai_apis[api_name]
                    chat_response = ai_connector.chat_with_ai(user_message, img_bytes, context)
                elif api_name == 'openclaw' or not api_name:
                    # Use OpenClaw provider
                    from backend.ai_apis.openclaw_ai_provider import OpenClawAIProvider
                    oc_provider = OpenClawAIProvider(
                        endpoint=api_endpoint or os.environ.get('OPENCLAW_ENDPOINT', 'http://localhost:18789'),
                        api_key=api_key or os.environ.get('OPENCLAW_API_KEY', '')
                    )
                    chat_response = oc_provider.chat_with_ai(user_message, img_bytes, context)
                else:
                    chat_response = f"Provider '{api_name}' not available. Use: {', '.join(ai_apis.keys())}"
            except Exception as e:
                logger.error(f"AI chat error: {e}")
                chat_response = f"I couldn't process that request. Error: {str(e)}"
        
        # Get current agent state for response
        errors = agent_state.get('errors', [])
        last_error = errors[-1] if errors else None
        
        return jsonify({
            "ok": True,
            "intent_detected": intent,
            "goal_updated": goal_updated,
            "task_updated": task_updated,
            "chat_response": chat_response,
            "agent_state": {
                "mode": agent_state.get('mode', 'manual'),
                "enabled": agent_state.get('enabled', False),
                "current_goal": agent_state.get('current_goal', ''),
                "current_task": agent_state.get('current_task', ''),
                "last_action": agent_state.get('last_action'),
                "last_action_time": agent_state.get('last_action_time'),
                "last_error": last_error,
                "action_count": agent_state.get('stats', {}).get('total_actions', 0)
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        return jsonify({
            "ok": False,
            "error": str(e),
            "chat_response": None,
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/api/agent/errors', methods=['GET'])
def api_agent_errors():
    """
    Get recent agent errors.
    
    Query params:
        limit: number (default: 10, max: 50)
        
    Response shape:
    {
        "errors": [
            {
                "timestamp": string,
                "type": string,
                "message": string,
                "context": {...} | null
            }
        ],
        "count": number,
        "total_errors": number
    }
    """
    try:
        limit = min(int(request.args.get('limit', 10)), 50)
    except (ValueError, TypeError):
        limit = 10
    
    errors = agent_state.get('errors', [])[-limit:]
    
    return jsonify({
        'errors': errors,
        'count': len(errors),
        'total_errors': agent_state.get('stats', {}).get('total_errors', 0),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/agent/actions', methods=['GET'])
def api_agent_actions():
    """
    Get recent agent actions.
    
    Query params:
        limit: number (default: 20, max: 100)
        
    Response shape:
    {
        "actions": [
            {
                "timestamp": string,
                "action": string,
                "frames": number,
                "result": string
            }
        ],
        "count": number,
        "total_actions": number
    }
    """
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
    except (ValueError, TypeError):
        limit = 20
    
    actions = agent_state.get('actions', [])[-limit:]
    
    return jsonify({
        'actions': actions,
        'count': len(actions),
        'total_actions': agent_state.get('stats', {}).get('total_actions', 0),
        'timestamp': datetime.now().isoformat()
    }), 200


# /api/health, /api/health/runtime, /api/health/emulator and /api/health/stream
# have been extracted to backend/routes/health.py — see register_health_routes()
# (registered via routes/__init__.py). The _format_uptime / _get_*_health_dict
# helpers were moved inline inside the blueprint.


# =========================================
# Agent Tools API (MCP/LM Studio Ready)
# =========================================

@app.route('/api/agent/context', methods=['GET'])
def api_agent_context():
    """
    Get comprehensive agent context for AI decision making.
    
    Response shape:
    {
        "loaded": boolean,
        "rom_name": string,
        "frame": number,
        "game_mode": "exploration" | "battle" | "menu" | "dialogue" | "title" | "unknown",
        "position": {"x": number, "y": number, "map_id": number, "map_name": string},
        "party": {"count": number, "pokemon": [...]},
        "inventory": {"money": number, "items": [...]},
        "battle": {"in_battle": boolean, "enemy": {...}},
        "health_summary": {"party_healthy": boolean, "lowest_hp_percent": number, "needs_healing": boolean},
        "recommendations": string[],
        "timestamp": string
    }
    
    Empty/loading response (no ROM):
    {
        "loaded": false,
        "game_mode": "none",
        "position": {...defaults...},
        "party": {...defaults...},
        ...
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'loaded': False,
        'rom_name': None,
        'frame': 0,
        'game_mode': 'none',
        'position': {'x': 0, 'y': 0, 'map_id': 0, 'map_name': 'none'},
        'party': {'count': 0, 'pokemon': []},
        'inventory': {'money': 0, 'items': []},
        'battle': {'in_battle': False, 'enemy': None},
        'health_summary': {
            'party_healthy': True,
            'lowest_hp_percent': 100,
            'needs_healing': False
        },
        'recommendations': [],
        'timestamp': datetime.now().isoformat()
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Get all context in one call
        position = {'x': 0, 'y': 0, 'map_id': 0, 'map_name': 'unknown'}
        party = {'count': 0, 'pokemon': []}
        inventory = {'money': 0, 'items': []}
        battle = {'in_battle': False, 'enemy': None}
        
        # Position
        if hasattr(emulator, 'get_position'):
            position = emulator.get_position()
        
        # Party
        if hasattr(emulator, 'get_party_info'):
            pokemon_list = emulator.get_party_info() or []
            party = {
                'count': len(pokemon_list),
                'pokemon': pokemon_list
            }
        
        # Inventory
        if hasattr(emulator, 'get_inventory_info'):
            inv = emulator.get_inventory_info() or {}
            inventory = {
                'money': inv.get('money', 0),
                'items': inv.get('items', [])
            }
        
        # Battle
        if hasattr(emulator, 'get_battle_info'):
            battle = emulator.get_battle_info()
        
        # Determine game mode
        game_mode = 'exploration'
        if battle.get('in_battle'):
            game_mode = 'battle'
        elif position.get('map_id', 255) == 255:
            game_mode = 'title'
        
        # Health summary
        lowest_hp = 100
        needs_healing = False
        for mon in party.get('pokemon', []):
            hp_pct = mon.get('hp_percent', 100)
            if hp_pct < lowest_hp:
                lowest_hp = hp_pct
            if hp_pct < 30:
                needs_healing = True
        
        # Recommendations
        recommendations = []
        if needs_healing:
            recommendations.append('Heal party at Pokemon Center')
        if battle.get('in_battle'):
            enemy = battle.get('enemy', {})
            if enemy.get('hp_percent', 100) < 30 and lowest_hp > 50:
                recommendations.append('Consider catching this Pokemon')
            else:
                recommendations.append('Battle in progress')
        if inventory.get('money', 0) > 10000:
            recommendations.append(f'Consider spending money: ¥{inventory["money"]:,}')
        
        return jsonify({
            'loaded': True,
            'rom_name': state.get('rom_name', state.get('rom_path', 'Unknown')),
            'frame': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else 0,
            'game_mode': game_mode,
            'position': position,
            'party': party,
            'inventory': inventory,
            'battle': battle,
            'health_summary': {
                'party_healthy': not needs_healing,
                'lowest_hp_percent': round(lowest_hp, 1),
                'needs_healing': needs_healing
            },
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting agent context: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/agent/mode', methods=['GET'])
def api_get_game_mode():
    """
    Get the current game mode/state.
    
    Response shape:
    {
        "mode": "exploration" | "battle" | "menu" | "dialogue" | "title" | "unknown",
        "in_battle": boolean,
        "in_menu": boolean,
        "in_dialogue": boolean,
        "details": {
            "battle_type": "wild" | "trainer" | "none",
            "menu_type": "main" | "pokemon" | "bag" | "none",
            "dialogue_active": boolean
        },
        "loaded": boolean,
        "timestamp": string
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'mode': 'none',
        'in_battle': False,
        'in_menu': False,
        'in_dialogue': False,
        'details': {
            'battle_type': 'none',
            'menu_type': 'none',
            'dialogue_active': False
        },
        'loaded': False,
        'timestamp': datetime.now().isoformat()
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Get battle info
        battle_info = {}
        if hasattr(emulator, 'get_battle_info'):
            battle_info = emulator.get_battle_info()
        
        in_battle = battle_info.get('in_battle', False)
        
        # Get position for title screen detection
        position = {}
        if hasattr(emulator, 'get_position'):
            position = emulator.get_position()
        
        # Determine mode
        if in_battle:
            mode = 'battle'
        elif position.get('map_id', 0) == 255 or position.get('map_id', 0) == 0 and position.get('x', 0) == 0:
            # Likely title screen or loading
            mode = 'title'
        else:
            mode = 'exploration'
        
        return jsonify({
            'mode': mode,
            'in_battle': in_battle,
            'in_menu': False,  # Would need memory analysis for this
            'in_dialogue': False,  # Would need memory analysis for this
            'details': {
                'battle_type': battle_info.get('battle_type', 'none'),
                'menu_type': 'none',  # Placeholder
                'dialogue_active': False  # Placeholder
            },
            'loaded': True,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting game mode: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/agent/act', methods=['POST'])
def api_act_and_observe():
    """
    Execute an action and return the new state observation.
    
    POST Body:
    {
        "action": "UP" | "DOWN" | "LEFT" | "RIGHT" | "A" | "B" | "START" | "SELECT",
        "frames": number (default: 1)
    }
    
    Response shape:
    {
        "success": boolean,
        "action": string,
        "frames": number,
        "observation": {
            "game_mode": string,
            "position": {...},
            "battle": {...},
            "health_summary": {...}
        },
        "changes": {
            "position_changed": boolean,
            "battle_started": boolean,
            "battle_ended": boolean
        },
        "timestamp": string
    }
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify({
            'success': False,
            'error': 'No ROM loaded',
            'timestamp': datetime.now().isoformat()
        }), 400
    
    try:
        data = request.get_json(silent=True) or {}
        action = data.get('action', 'NOOP').upper()
        frames = data.get('frames', 1)
        
        # Validate action
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'}
        if action not in valid_actions:
            return jsonify({
                'success': False,
                'error': f'Invalid action: {action}',
                'valid_actions': list(valid_actions),
                'timestamp': datetime.now().isoformat()
            }), 400
        
        emulator = emulators[active]
        
        # Get state before action
        position_before = {}
        battle_before = {}
        if hasattr(emulator, 'get_position'):
            position_before = emulator.get_position()
        if hasattr(emulator, 'get_battle_info'):
            battle_before = emulator.get_battle_info()
        
        # Execute action
        success = False
        if hasattr(emulator, 'step'):
            success = emulator.step(action, frames)
        
        # Get state after action
        position_after = {}
        battle_after = {}
        if hasattr(emulator, 'get_position'):
            position_after = emulator.get_position()
        if hasattr(emulator, 'get_battle_info'):
            battle_after = emulator.get_battle_info()
        
        # Determine changes
        position_changed = (
            position_before.get('x') != position_after.get('x') or
            position_before.get('y') != position_after.get('y') or
            position_before.get('map_id') != position_after.get('map_id')
        )
        
        battle_started = not battle_before.get('in_battle', False) and battle_after.get('in_battle', False)
        battle_ended = battle_before.get('in_battle', False) and not battle_after.get('in_battle', False)
        
        # Determine game mode
        game_mode = 'exploration'
        if battle_after.get('in_battle'):
            game_mode = 'battle'
        
        # Health summary
        lowest_hp = 100
        needs_healing = False
        if hasattr(emulator, 'get_party_info'):
            party = emulator.get_party_info() or []
            for mon in party:
                hp_pct = mon.get('hp_percent', 100)
                if hp_pct < lowest_hp:
                    lowest_hp = hp_pct
                if hp_pct < 30:
                    needs_healing = True
        
        return jsonify({
            'success': success,
            'action': action,
            'frames': frames,
            'observation': {
                'game_mode': game_mode,
                'position': position_after,
                'battle': battle_after,
                'health_summary': {
                    'party_healthy': not needs_healing,
                    'lowest_hp_percent': round(lowest_hp, 1),
                    'needs_healing': needs_healing
                }
            },
            'changes': {
                'position_changed': position_changed,
                'battle_started': battle_started,
                'battle_ended': battle_ended
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error in act_and_observe: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/agent/dialogue', methods=['GET'])
def api_dialogue_state():
    """
    Get the current dialogue/text box state.
    
    Response shape:
    {
        "active": boolean,
        "text": string or null,
        "has_options": boolean,
        "options": string[],
        "selected_option": number,
        "can_advance": boolean,
        "loaded": boolean,
        "timestamp": string
    }
    
    Note: Full dialogue detection requires memory analysis.
    This returns a safe default structure.
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'active': False,
        'text': None,
        'has_options': False,
        'options': [],
        'selected_option': 0,
        'can_advance': True,
        'loaded': False,
        'timestamp': datetime.now().isoformat()
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Dialogue detection would require:
        # 1. Reading tilemap for text box detection
        # 2. Reading dialogue text from memory
        # 3. Reading menu selection state
        
        # For now, return safe defaults
        # This can be enhanced with memory scanning later
        
        # Basic detection: check if in battle (battle has dialogue-like menus)
        in_battle = False
        if hasattr(emulator, 'get_battle_info'):
            battle = emulator.get_battle_info()
            in_battle = battle.get('in_battle', False)
        
        return jsonify({
            'active': False,
            'text': None,
            'has_options': in_battle,  # Battle has move/item selection
            'options': [],
            'selected_option': 0,
            'can_advance': True,
            'loaded': True,
            'note': 'Dialogue detection requires memory scanning - safe defaults returned',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting dialogue state: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/agent/menu', methods=['GET'])
def api_menu_state():
    """
    Get the current menu state.
    
    Response shape:
    {
        "active": boolean,
        "type": "main" | "pokemon" | "bag" | "battle" | "save" | "none",
        "selection": number,
        "options": string[],
        "can_close": boolean,
        "loaded": boolean,
        "timestamp": string
    }
    
    Note: Full menu detection requires memory analysis.
    This returns a safe default structure.
    """
    state = get_game_state()
    active = state.get('active_emulator')
    
    empty_response = {
        'active': False,
        'type': 'none',
        'selection': 0,
        'options': [],
        'can_close': True,
        'loaded': False,
        'timestamp': datetime.now().isoformat()
    }
    
    if not state.get('rom_loaded') or not active or active not in emulators:
        return jsonify(empty_response), 200
    
    try:
        emulator = emulators[active]
        
        # Menu detection would require:
        # 1. Reading menu state byte
        # 2. Reading menu selection index
        # 3. Parsing menu items
        
        # Check battle state (battle has menus)
        in_battle = False
        battle_type = 'none'
        if hasattr(emulator, 'get_battle_info'):
            battle = emulator.get_battle_info()
            in_battle = battle.get('in_battle', False)
            battle_type = battle.get('battle_type', 'none')
        
        if in_battle:
            return jsonify({
                'active': True,
                'type': 'battle',
                'selection': 0,
                'options': ['FIGHT', 'BAG', 'POKEMON', 'RUN'],
                'can_close': False,
                'loaded': True,
                'timestamp': datetime.now().isoformat()
            }), 200
        
        return jsonify({
            'active': False,
            'type': 'none',
            'selection': 0,
            'options': [],
            'can_close': True,
            'loaded': True,
            'note': 'Menu detection requires memory scanning - safe defaults returned',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.debug(f"Error getting menu state: {e}")
        empty_response['error'] = str(e)
        return jsonify(empty_response), 200


@app.route('/api/rom/load', methods=['POST'])
def api_rom_load_alias():
    return load_rom()


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
