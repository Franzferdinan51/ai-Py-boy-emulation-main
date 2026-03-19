"""
Enhanced AI Game Boy Server with Stream Stability Fixes
"""
import os
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

ai_runtime_state = {
    "provider": ai_provider_manager.default_provider,
    "model": "",
    "api_endpoint": "",
}

if ai_runtime_state["provider"] == "lmstudio":
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

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint for monitoring"""
    import platform
    import sys
    
    health_data = {
        "status": "healthy",
        "service": "ai-game-server",
        "version": "3.0.0",
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "flask": "ok",
            "pyboy": "ok" if PYBOY_AVAILABLE else "not_available",
            "mcp": "ok" if MCP_AVAILABLE else "not_available"
        }
    }
    return jsonify(health_data), 200

@app.route('/api/config/validate', methods=['GET'])
def validate_configuration():
    """Validate system configuration and environment variables"""
    if SECURE_CONFIG_AVAILABLE:
        validation = secure_config.validate_environment_variables()
        safe_config = secure_config.get_safe_config()

        response_data = {
            "validation": validation,
            "configuration": safe_config,
            "timestamp": time.time()
        }

        if not validation['valid']:
            return jsonify(response_data), 400
        elif validation['warnings']:
            return jsonify(response_data), 200  # OK but with warnings
        else:
            return jsonify(response_data), 200
    else:
        return jsonify({
            "error": "Configuration validation not available",
            "fallback": os.environ.get('FLASK_ENV', 'development')
        }), 503

@app.route('/api/config', methods=['GET'])
def get_configuration():
    """Get safe configuration information"""
    if SECURE_CONFIG_AVAILABLE:
        return jsonify(secure_config.get_safe_config()), 200
    else:
        return jsonify({
            "error": "Configuration manager not available",
            "basic_config": {
                "host": HOST,
                "port": PORT,
                "debug": DEBUG,
                "environment": os.environ.get('FLASK_ENV', 'development')
            }
        }), 503

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

@app.route('/api/providers/status', methods=['GET'])
def get_providers_status():
    """Get detailed status of all AI providers"""
    return jsonify(ai_provider_manager.get_provider_status()), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get a list of available models for a given provider"""
    provider_name = request.args.get('provider')
    if not provider_name:
        return jsonify({"error": "Provider name is required"}), 400

    models = ai_provider_manager.get_models(provider_name)
    return jsonify({"models": models}), 200

@app.route('/api/openclaw/models', methods=['GET'])
def get_openclaw_models():
    """Get all available models from OpenClaw Gateway"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        discovery = get_model_discovery(app.config.get('OPENCLAW_ENDPOINT', 'http://localhost:18789'))
        models = discovery.get_available_models(force_refresh=force_refresh)
        
        return jsonify({
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "capabilities": model.capabilities,
                    "context_window": model.context_window,
                    "is_vision_capable": model.is_vision_capable,
                    "is_free": model.is_free,
                    "description": model.description,
                    "priority": model.priority
                }
                for model in models
            ],
            "timestamp": datetime.now().isoformat(),
            "cached": not force_refresh and discovery._is_cache_valid()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get OpenClaw models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/openclaw/models/vision', methods=['GET'])
def get_vision_models():
    """Get only vision-capable models from OpenClaw"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        discovery = get_model_discovery(app.config.get('OPENCLAW_ENDPOINT', 'http://localhost:18789'))
        models = discovery.get_vision_models()
        
        return jsonify({
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "capabilities": model.capabilities,
                    "context_window": model.context_window,
                    "is_free": model.is_free,
                    "description": model.description,
                    "priority": model.priority
                }
                for model in models
            ],
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get vision models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/openclaw/models/planning', methods=['GET'])
def get_planning_models():
    """Get models suitable for planning/decision making"""
    try:
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        discovery = get_model_discovery(app.config.get('OPENCLAW_ENDPOINT', 'http://localhost:18789'))
        models = discovery.get_planning_models()
        
        return jsonify({
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "capabilities": model.capabilities,
                    "context_window": model.context_window,
                    "is_free": model.is_free,
                    "description": model.description,
                    "priority": model.priority
                }
                for model in models
            ],
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get planning models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/openclaw/models/recommend', methods=['GET'])
def recommend_model():
    """Get model recommendation for a specific use case"""
    try:
        use_case = request.args.get('use_case', 'planning')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        discovery = get_model_discovery(app.config.get('OPENCLAW_ENDPOINT', 'http://localhost:18789'))
        model = discovery.recommend_model(use_case)
        
        if model:
            return jsonify({
                "model": {
                    "id": model.id,
                    "name": model.name,
                    "provider": model.provider,
                    "capabilities": model.capabilities,
                    "context_window": model.context_window,
                    "is_vision_capable": model.is_vision_capable,
                    "is_free": model.is_free,
                    "description": model.description,
                    "priority": model.priority
                },
                "use_case": use_case,
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({"error": "No model found for this use case"}), 404
            
    except Exception as e:
        logger.error(f"Failed to recommend model: {e}")
        return jsonify({"error": str(e)}), 500

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
        launch_ui = request.form.get('launch_ui', 'true')

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
            if not rom_path:
                return jsonify({"error": "No ROM path provided"}), 400
            if emulator_type not in emulators:
                return jsonify({"error": f"Emulator {emulator_type} not found"}), 404
            emulator = emulators[emulator_type]
            result = emulator.load_rom(rom_path, launch_ui=launch_ui)
            if result:
                game_states[emulator_type].update({
                    'rom_loaded': True,
                    'rom_name': Path(rom_path).name,
                    'window_open': True,
                    'last_updated': time.time()
                })
                return jsonify({"status": "success", "rom_loaded": True, "rom_name": Path(rom_path).name, "emulator": emulator_type}), 200
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

@app.route('/api/agent/status', methods=['GET'])
def api_agent_status():
    return jsonify({
        'mode': agent_state.get('mode', 'manual') if 'agent_state' in globals() else 'manual',
        'enabled': agent_state.get('enabled', False) if 'agent_state' in globals() else False,
        'last_decision': agent_state.get('last_decision') if 'agent_state' in globals() else None,
        'last_action': agent_state.get('last_action') if 'agent_state' in globals() else None,
    }), 200

@app.route('/api/openclaw/health', methods=['GET'])
def api_openclaw_health():
    return jsonify({'ok': True, 'reachable': True}), 200

@app.route('/api/rom/load', methods=['POST'])
def api_rom_load_alias():
    return load_rom()

@app.route('/api/action', methods=['POST', 'OPTIONS'])
def execute_action():
    """Execute an action in the active emulator"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    # Track request for logging
    request_id = f"action_{time.time_ns()}"
    logger.info(f"[{request_id}] Action request received")

    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning(f"[{request_id}] Action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        # Validate and parse JSON data
        try:
            data = request.get_json(force=True)
            if data is None:
                return jsonify({"error": "Invalid JSON data"}), 400
        except Exception as e:
            return jsonify({"error": f"JSON parse error: {str(e)}"}), 400

        action = data.get('action', data.get('button', 'SELECT'))
        frames = data.get('frames', 1)

        # Validate action using comprehensive input validation
        try:
            action = validate_string_input(
                action,
                "action",
                min_length=1,
                max_length=10,
                allowed_chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
            )
        except ValueError as e:
            logger.warning(f"[{request_id}] Invalid action format: {action} - {e}")
            return jsonify({"error": str(e)}), 400

        # Validate action is in allowed set
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP'}
        if action not in valid_actions:
            logger.warning(f"[{request_id}] Invalid action requested: {action}")
            return jsonify({
                "error": f"Invalid action: {action}",
                "valid_actions": list(valid_actions)
            }), 400

        # Validate frames using comprehensive input validation
        try:
            frames = validate_integer_input(
                frames,
                "frames",
                min_value=1,
                max_value=100
            )
        except ValueError as e:
            logger.warning(f"[{request_id}] Invalid frames value: {frames} - {e}")
            return jsonify({"error": str(e)}), 400

        logger.info(f"[{request_id}] Executing action: {action} for {frames} frame(s)")
        emulator = emulators[current_state["active_emulator"]]

        # Add timeout protection for emulator operations
        success = timeout_handler(5.0)(emulator.step)(action, frames)

        if success:
            add_to_action_history(action)
            logger.debug(f"[{request_id}] Action {action} executed successfully")
            return jsonify({
                "message": "Action executed successfully",
                "action": action,
                "frames": frames,
                "history_length": len(get_action_history())
            }), 200
        else:
            logger.error(f"[{request_id}] Failed to execute action: {action}")
            return jsonify({"error": "Failed to execute action"}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Error executing action: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/ai-action', methods=['POST', 'OPTIONS'])
def get_ai_action():
    """Get the next action from the specified AI API with optimization integration"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    # Track request for logging
    request_id = f"ai_action_{time.time_ns()}"
    start_time = time.time()
    logger.info(f"[{request_id}] AI action request received")

    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning(f"[{request_id}] AI action requested but no ROM loaded")
            return jsonify({"error": "No ROM loaded"}), 400

        # Validate and parse JSON data
        try:
            data = validate_json_data(request.get_data(as_text=True), "AI action request")
        except ValueError as e:
            logger.error(f"[{request_id}] Invalid JSON data: {e}")
            return jsonify({"error": f"Invalid request data: {str(e)}"}), 400

        # Extract and validate parameters
        api_name = data.get('api_name')
        requested_api_name = api_name
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')
        goal = data.get('goal', '')

        # Validate optional API parameters
        if api_name is not None:
            try:
                api_name = validate_string_input(
                    api_name,
                    "api_name",
                    min_length=1,
                    max_length=50,
                    pattern=r'^[a-zA-Z0-9_-]+$'
                )
            except ValueError as e:
                logger.warning(f"[{request_id}] Invalid API name: {e}")
                return jsonify({"error": f"Invalid API name: {str(e)}"}), 400

        if api_key is not None:
            try:
                api_key = validate_string_input(
                    api_key,
                    "api_key",
                    min_length=1,
                    max_length=500,
                    pattern=r'^[a-zA-Z0-9._-]+$'
                )
            except ValueError as e:
                logger.warning(f"[{request_id}] Invalid API key format")
                return jsonify({"error": "Invalid API key format"}), 400

        if api_endpoint is not None:
            try:
                api_endpoint = validate_string_input(
                    api_endpoint,
                    "api_endpoint",
                    min_length=1,
                    max_length=500,
                    pattern=r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?$'
                )
            except ValueError as e:
                logger.warning(f"[{request_id}] Invalid API endpoint: {e}")
                return jsonify({"error": f"Invalid API endpoint: {str(e)}"}), 400

        if model is not None:
            try:
                model = validate_string_input(
                    model,
                    "model",
                    min_length=1,
                    max_length=100,
                    pattern=r'^[a-zA-Z0-9._-]+$'
                )
            except ValueError as e:
                logger.warning(f"[{request_id}] Invalid model name: {e}")
                return jsonify({"error": f"Invalid model name: {str(e)}"}), 400

        # Validate goal
        try:
            goal = validate_string_input(
                goal,
                "goal",
                min_length=0,
                max_length=500,
                pattern=r'^[a-zA-Z0-9\s\.,\?\!@#\$%\^&\*\(\)_\-\+=\{\}\[\]:;"\'<>\?/~`|\\]+$'
            )
        except ValueError as e:
            logger.warning(f"[{request_id}] Invalid goal: {e}")
            return jsonify({"error": f"Invalid goal: {str(e)}"}), 400

        configured_provider = ai_runtime_state.get("provider")
        configured_model = ai_runtime_state.get("model")
        configured_endpoint = ai_runtime_state.get("api_endpoint")
        api_name = api_name or configured_provider
        if not model and configured_model:
            model = configured_model
        if not api_endpoint and configured_endpoint:
            api_endpoint = configured_endpoint

        logger.info(f"[{request_id}] AI action request: api={api_name or 'auto'}, model={model or 'default'}, goal='{goal[:50]}...'")
        update_game_state({
            "current_goal": goal,
            "current_provider": api_name,
            "current_model": model,
        })

        current_state = get_game_state()
        emulator = emulators[current_state["active_emulator"]]

        # Get screen using optimized timeout protection
        try:
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'thread_pool_manager'):
                # Use thread pool for screen capture
                def capture_screen():
                    return timeout_handler(3.0)(emulator.get_screen)()

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(capture_screen)
                    screen_array = future.result(timeout=4.0)  # Slightly longer timeout
            else:
                screen_array = timeout_handler(3.0)(emulator.get_screen)()
        except Exception as screen_error:
            logger.error(f"[{request_id}] Timeout getting screen: {screen_error}")
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'error_handler'):
                optimization_system_manager.error_handler.log_error(
                    error_type="screen_capture_timeout",
                    error_message=str(screen_error),
                    context={"request_id": request_id}
                )
            return jsonify({"error": "Screen capture timeout"}), 500

        if screen_array is None or screen_array.size == 0:
            logger.error(f"[{request_id}] Failed to get screen from PyBoy emulator")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert screen to bytes with optimization
        try:
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'ai_cache_manager'):
                # Create game context for caching
                game_context = {
                    'frame_count': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else 0,
                    'active_emulator': current_state["active_emulator"],
                    'goal': goal,
                    'timestamp': time.time()
                }

                # Check for cached AI response
                screen_hash = optimization_system_manager.ai_cache_manager.generate_screen_hash(
                    screen_array.tobytes(), game_context
                )
                cached_response = optimization_system_manager.ai_cache_manager.get_ai_response(
                    screen_hash, goal, api_name, model
                )

                if cached_response:
                    # Use cached AI response
                    logger.info(f"[{request_id}] AI action cache hit - using cached response")
                    action = cached_response.response
                    actual_provider = cached_response.provider
                    cache_hit = True
                    total_time = time.time() - start_time

                    # Update action history
                    add_to_action_history(action)

                    logger.info(f"[{request_id}] AI ({actual_provider or 'cached'}) suggested action: {action}")
                    return jsonify({
                        "action": action,
                        "provider_used": actual_provider,
                        "history": get_action_history()[-10:],
                        "optimization": {
                            "cache_hit": True,
                            "response_time_ms": round(total_time * 1000, 2),
                            "optimization_enabled": True
                        }
                    }), 200

            # Get fresh screen bytes
            img_bytes = timeout_handler(3.0)(emulator.get_screen_bytes)()
            cache_hit = False

        except Exception as convert_error:
            logger.error(f"[{request_id}] Timeout converting screen: {convert_error}")
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'error_handler'):
                optimization_system_manager.error_handler.log_error(
                    error_type="screen_conversion_timeout",
                    error_message=str(convert_error),
                    context={"request_id": request_id}
                )
            return jsonify({"error": "Screen processing timeout"}), 500

        if len(img_bytes) == 0:
            logger.error(f"[{request_id}] Failed to convert PyBoy screen to bytes")
            return jsonify({"error": "Failed to process screen image"}), 500

        logger.debug(f"[{request_id}] PyBoy screen captured for AI: {len(img_bytes)} bytes")

        # Set environment variables for this request if provided (with validation)
        if api_key:
            if len(api_key) < 10:
                logger.warning(f"[{request_id}] API key seems too short for provider: {api_name}")
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key

        if model:
            if len(model) < 1:
                logger.warning(f"[{request_id}] Model name is empty for provider: {api_name}")
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Check if the requested provider is available
        if api_name and api_name not in ai_provider_manager.get_available_providers():
            available_providers = ai_provider_manager.get_available_providers()
            logger.warning(f"[{request_id}] Requested provider '{api_name}' is not available. Available providers: {available_providers}")

            if requested_api_name:
                return jsonify({
                    "error": f"Provider '{api_name}' is not available",
                    "available_providers": available_providers,
                    "suggestion": f"Please use one of the available providers: {', '.join(available_providers)}"
                }), 400

            logger.warning(f"[{request_id}] Falling back because configured provider '{api_name}' is unavailable")
            api_name = None

        # Use provider manager with optimization and timeout protection
        logger.debug(f"[{request_id}] Calling AI API: {api_name or 'auto'} with model: {model or 'default'}")
        current_history = get_action_history()

        try:
            # Use optimized thread pool if available
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'thread_pool_manager'):
                def call_ai_api():
                    return timeout_handler(AI_REQUEST_TIMEOUT)(ai_provider_manager.get_next_action)(
                        img_bytes, goal, current_history, api_name, model
                    )

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_ai_api)
                    action, actual_provider = future.result(timeout=AI_REQUEST_TIMEOUT + 5.0)
            else:
                action, actual_provider = timeout_handler(AI_REQUEST_TIMEOUT)(ai_provider_manager.get_next_action)(
                    img_bytes, goal, current_history, api_name, model
                )

        except Exception as ai_timeout_error:
            logger.error(f"[{request_id}] AI request timeout: {ai_timeout_error}")
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'error_handler'):
                optimization_system_manager.error_handler.log_error(
                    error_type="ai_request_timeout",
                    error_message=str(ai_timeout_error),
                    context={"request_id": request_id, "api_name": api_name, "model": model}
                )
            return jsonify({"error": "AI request timeout"}), 500

        if not action or not isinstance(action, str):
            logger.error(f"[{request_id}] Invalid AI response: action={action}")
            if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'error_handler'):
                optimization_system_manager.error_handler.log_error(
                    error_type="invalid_ai_response",
                    error_message=f"Invalid AI response: {action}",
                    context={"request_id": request_id, "provider": actual_provider}
                )
            return jsonify({"error": "AI returned invalid action"}), 500

        add_to_action_history(action)
        update_game_state({
            "current_provider": actual_provider or api_name,
            "current_model": model,
        })

        # Cache the AI response if optimization is available
        if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'ai_cache_manager'):
            try:
                optimization_system_manager.ai_cache_manager.cache_ai_response(
                    screen_hash=screen_hash,
                    user_goal=goal,
                    ai_response=action,
                    provider_name=actual_provider or api_name,
                    model_name=model,
                    response_time=time.time() - start_time,
                    confidence_score=0.8  # Default confidence
                )
                logger.debug(f"[{request_id}] AI response cached for future use")
            except Exception as cache_error:
                logger.debug(f"[{request_id}] Failed to cache AI response: {cache_error}")

        total_time = time.time() - start_time
        logger.info(f"[{request_id}] AI ({actual_provider or 'fallback'}) suggested action: {action}")

        # Prepare response with optimization info
        response_data = {
            "action": action,
            "provider_used": actual_provider,
            "history": get_action_history()[-10:]
        }

        # Add optimization information if available
        if OPTIMIZATION_SYSTEM_AVAILABLE:
            response_data["optimization"] = {
                "cache_hit": cache_hit,
                "response_time_ms": round(total_time * 1000, 2),
                "memory_pressure": optimization_system_manager.get_memory_pressure(),
                "optimization_enabled": True
            }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"[{request_id}] Error getting AI action: {e}", exc_info=True)
        api_name = data.get('api_name') if 'data' in locals() else 'unknown'
        goal = data.get('goal', '') if 'data' in locals() else 'unknown'
        logger.error(f"[{request_id}] AI action failed - API: {api_name}, Goal: '{goal}', Screen size: {len(img_bytes) if 'img_bytes' in locals() else 'unknown'}")

        # Log error with optimization system if available
        if OPTIMIZATION_SYSTEM_AVAILABLE and hasattr(optimization_system_manager, 'error_handler'):
            optimization_system_manager.error_handler.log_error(
                error_type="ai_action_failed",
                error_message=str(e),
                context={
                    "request_id": request_id,
                    "api_name": api_name,
                    "goal": goal,
                    "screen_size": len(img_bytes) if 'img_bytes' in locals() else 0
                }
            )

        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def ai_chat():
    """Send a message to the AI and get a response with automatic fallback"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "chat request")

        user_message = data.get('message', '')
        api_name = data.get('api_name')
        api_key = data.get('api_key')
        api_endpoint = data.get('api_endpoint')
        model = data.get('model')

        # Validate user message input
        try:
            user_message = validate_string_input(
                user_message,
                "message",
                min_length=1,
                max_length=2000,
                pattern=r'^[a-zA-Z0-9\s\.,\?\!@#\$%\^&\*\(\)_\-\+=\{\}\[\]:;"\'<>\?/~`|\\]+$'
            )
        except ValueError as e:
            return jsonify({"error": f"Invalid message: {str(e)}"}), 400

        # Validate optional API parameters
        if api_name is not None:
            try:
                api_name = validate_string_input(
                    api_name,
                    "api_name",
                    min_length=1,
                    max_length=50,
                    pattern=r'^[a-zA-Z0-9_-]+$'
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid API name: {str(e)}"}), 400

        if api_key is not None:
            try:
                api_key = validate_string_input(
                    api_key,
                    "api_key",
                    min_length=1,
                    max_length=500,
                    pattern=r'^[a-zA-Z0-9._-]+$'
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid API key format"}), 400

        if api_endpoint is not None:
            try:
                api_endpoint = validate_string_input(
                    api_endpoint,
                    "api_endpoint",
                    min_length=1,
                    max_length=500,
                    pattern=r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?$'
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid API endpoint: {str(e)}"}), 400

        if model is not None:
            try:
                model = validate_string_input(
                    model,
                    "model",
                    min_length=1,
                    max_length=100,
                    pattern=r'^[a-zA-Z0-9._-]+$'
                )
            except ValueError as e:
                return jsonify({"error": f"Invalid model name: {str(e)}"}), 400

        current_state = get_game_state()
        emulator = emulators[current_state["active_emulator"]]

        # Get screen bytes using proper PyBoy connectors
        img_bytes = emulator.get_screen_bytes()

        if len(img_bytes) == 0:
            logger.error("Failed to get screen bytes from PyBoy for chat")
            return jsonify({"error": "Failed to capture screen"}), 500

        logger.debug(f"PyBoy screen captured for chat: {len(img_bytes)} bytes")

        context = {
            "current_goal": current_state["current_goal"],
            "action_history": get_action_history()[-20:],
            "game_type": current_state["active_emulator"].upper()
        }

        # Get model/provider with priority: game_state -> request params -> defaults
        current_provider = current_state.get('current_provider')
        current_model = current_state.get('current_model')
        configured_provider = ai_runtime_state.get('provider')
        configured_model = ai_runtime_state.get('model')
        configured_endpoint = ai_runtime_state.get('api_endpoint')

        # Use request parameters first, then runtime config, then stored values.
        api_name = api_name or configured_provider or current_provider
        model = model or configured_model or current_model
        api_endpoint = api_endpoint or configured_endpoint

        logger.debug(f"Chat using provider: {api_name or 'auto'}, model: {model or 'default'}")
        logger.debug(f"Stored provider: {current_provider}, stored model: {current_model}")

        # Set environment variables for this request if provided
        if api_key:
            if api_name == 'gemini':
                os.environ['GEMINI_API_KEY'] = api_key
            elif api_name == 'openrouter':
                os.environ['OPENROUTER_API_KEY'] = api_key
            elif api_name == 'openai-compatible':
                os.environ['OPENAI_API_KEY'] = api_key
                if api_endpoint:
                    os.environ['OPENAI_ENDPOINT'] = api_endpoint
            elif api_name == 'nvidia':
                os.environ['NVIDIA_API_KEY'] = api_key
        if model:
            if api_name == 'openai-compatible':
                os.environ['OPENAI_MODEL'] = model
            elif api_name == 'nvidia':
                os.environ['NVIDIA_MODEL'] = model

        # Use provider manager with automatic fallback
        response_text, actual_provider = ai_provider_manager.chat_with_ai(
            user_message, img_bytes, context, api_name, model
        )
        update_game_state({
            "current_provider": actual_provider or api_name,
            "current_model": model,
        })

        logger.info(f"AI chat message from user: {user_message} (provider: {actual_provider or 'fallback'})")
        return jsonify({
            "response": response_text,
            "provider_used": actual_provider
        }), 200

    except Exception as e:
        logger.error(f"Error in AI chat: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/launch', methods=['POST'])
def launch_ui():
    """Launch UI process for the current ROM"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        if not hasattr(emulator, 'launch_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.launch_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process launched successfully")
            return jsonify({
                "message": "UI launched successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to launch UI process")
            return jsonify({"error": "Failed to launch UI"}), 500

    except Exception as e:
        logger.error(f"Error launching UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/stop', methods=['POST'])
def stop_ui():
    """Stop the UI process"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        if not hasattr(emulator, 'stop_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.stop_ui()

        if success:
            logger.info("UI process stopped successfully")
            return jsonify({"message": "UI stopped successfully"}), 200
        else:
            logger.error("Failed to stop UI process")
            return jsonify({"error": "Failed to stop UI"}), 500

    except Exception as e:
        logger.error(f"Error stopping UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/restart', methods=['POST'])
def restart_ui():
    """Restart the UI process"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        if not hasattr(emulator, 'restart_ui'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        success = emulator.restart_ui()

        if success:
            ui_status = emulator.get_ui_status()
            logger.info("UI process restarted successfully")
            return jsonify({
                "message": "UI restarted successfully",
                "ui_status": ui_status
            }), 200
        else:
            logger.error("Failed to restart UI process")
            return jsonify({"error": "Failed to restart UI"}), 500

    except Exception as e:
        logger.error(f"Error restarting UI: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/ui/status', methods=['GET'])
def get_ui_status():
    """Get UI process status"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        if not hasattr(emulator, 'get_ui_status'):
            return jsonify({"error": "UI control not supported by this emulator"}), 400

        ui_status = emulator.get_ui_status()

        return jsonify({
            "ui_status": ui_status,
            "rom_loaded": current_state["rom_loaded"],
            "active_emulator": current_state["active_emulator"]
        }), 200

    except Exception as e:
        logger.error(f"Error getting UI status: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/tetris/train', methods=['POST'])
def train_tetris_ai():
    """Train the Tetris genetic AI"""
    try:
        data = request.get_json()
        population_size = data.get('population_size', 20)
        generations = data.get('generations', 5)

        # Get Tetris AI provider
        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        # Check if ROM is loaded and emulator is available
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({
                'success': False,
                'error': 'No ROM loaded'
            }), 400

        emulator = emulators[current_state["active_emulator"]]

        # Start training
        logger.info(f"Starting Tetris AI training: population_size={population_size}, generations={generations}")

        # Train in a separate thread to avoid blocking
        def train_async():
            try:
                results = tetris_ai.train_generation(emulator, population_size, generations)
                logger.info(f"Tetris AI training completed: {results}")
            except Exception as e:
                logger.error(f"Tetris AI training failed: {e}")

        import threading
        training_thread = threading.Thread(target=train_async)
        training_thread.daemon = True
        training_thread.start()

        return jsonify({
            'success': True,
            'message': 'Tetris AI training started',
            'population_size': population_size,
            'generations': generations,
            'provider_status': tetris_ai.get_status()
        })

    except Exception as e:
        logger.error(f"Error starting Tetris AI training: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/status', methods=['GET'])
def get_tetris_status():
    """Get Tetris genetic AI status"""
    try:
        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'available': False,
                'error': 'Tetris genetic AI not available'
            })

        return jsonify({
            'success': True,
            'status': tetris_ai.get_status()
        })

    except Exception as e:
        logger.error(f"Error getting Tetris AI status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/save', methods=['POST'])
def save_tetris_model():
    """Save Tetris AI model"""
    try:
        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "tetris save request")

        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        # Validate filepath for security
        try:
            filepath = validate_string_input(
                filepath,
                "filepath",
                min_length=1,
                max_length=500
            )

            # Additional path validation
            filepath = sanitize_filename(filepath)

            # Ensure filepath has proper extension
            if not filepath.endswith(('.pkl', '.model', '.dat')):
                raise ValueError("Filepath must have a valid model extension (.pkl, .model, .dat)")

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        success = tetris_ai.save_training_state(filepath)

        return jsonify({
            'success': success,
            'message': 'Model saved successfully' if success else 'Failed to save model'
        })

    except Exception as e:
        logger.error(f"Error saving Tetris model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/tetris/load', methods=['POST'])
def load_tetris_model():
    """Load Tetris AI model"""
    try:
        # Validate and parse JSON data
        data = validate_json_data(request.get_data(as_text=True), "tetris load request")

        filepath = data.get('filepath')

        if not filepath:
            return jsonify({
                'success': False,
                'error': 'Filepath required'
            }), 400

        # Validate filepath for security
        try:
            filepath = validate_string_input(
                filepath,
                "filepath",
                min_length=1,
                max_length=500
            )

            # Additional path validation
            filepath = sanitize_filename(filepath)

            # Ensure filepath has proper extension
            if not filepath.endswith(('.pkl', '.model', '.dat')):
                raise ValueError("Filepath must have a valid model extension (.pkl, .model, .dat)")

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

        tetris_ai = ai_provider_manager.providers.get('tetris-genetic', {}).get('connector')

        if not tetris_ai:
            return jsonify({
                'success': False,
                'error': 'Tetris genetic AI not available'
            }), 400

        success = tetris_ai.load_training_state(filepath)

        return jsonify({
            'success': success,
            'message': 'Model loaded successfully' if success else 'Failed to load model'
        })

    except Exception as e:
        logger.error(f"Error loading Tetris model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/screen', methods=['GET'])
def get_screen():
    """Get the current screen from the active emulator with optimization integration"""
    try:
        start_time = time.time()
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Get screen array directly from the emulator.
        # For live gameplay view, correctness is more important than async screen-capture
        # optimization/caching layers, which can freeze the displayed frame.
        screen_array = emulator.get_screen()

        # Validate screen data - don't use placeholders
        if screen_array is None or screen_array.size == 0:
            logger.error("Screen data is None or empty")
            return jsonify({"error": "Failed to capture screen"}), 500

        # Convert to base64 with optimization
        conversion_start = time.time()

        # Use direct conversion without screen caching for live gameplay view.
        img_base64 = numpy_to_base64_image(screen_array)
        conversion_time = time.time() - conversion_start
        cache_hit = False

        if not img_base64:
            logger.error("Failed to convert screen to base64")
            return jsonify({"error": "Failed to process screen image"}), 500

        total_time = time.time() - start_time

        # Update performance metrics
        update_performance_metrics(conversion_time, total_time)

        # Prepare response with optimization info
        response_data = {
            "image": img_base64,
            "shape": screen_array.shape,
            "timestamp": time.time(),
            "pyboy_frame": emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else None,
            "performance": {
                "total_time_ms": round(total_time * 1000, 2),
                "conversion_time_ms": round(conversion_time * 1000, 2),
                "current_fps": round(performance_monitor['current_fps'], 1),
                "adaptive_fps_target": performance_monitor['adaptive_fps_target']
            }
        }

        # Add optimization information if available
        if OPTIMIZATION_SYSTEM_AVAILABLE:
            response_data["optimization"] = {
                "cache_hit": cache_hit,
                "memory_pressure": optimization_system_manager.get_memory_pressure(),
                "optimization_enabled": True
            }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error getting screen: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance monitoring statistics"""
    try:
        stats = get_performance_stats()

        # Get emulator-specific stats if available
        emulator_stats = {}
        current_state = get_game_state()
        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]
            if hasattr(emulator, 'get_performance_stats'):
                emulator_stats = emulator.get_performance_stats()

        return jsonify({
            "server_performance": stats,
            "emulator_performance": emulator_stats,
            "system_info": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_usage_mb": _get_memory_usage(),
                "multi_process_mode": USE_MULTI_PROCESS,
                "timestamp": time.time()
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/emulator/mode', methods=['GET'])
def get_emulator_mode():
    """Get current emulator mode"""
    return jsonify({
        "multi_process_mode": USE_MULTI_PROCESS,
        "available_modes": ["single-process", "multi-process"],
        "current_mode": "multi-process" if USE_MULTI_PROCESS else "single-process"
    }), 200


@app.route('/api/emulator/clear-cache', methods=['POST'])
def clear_emulator_cache():
    """Clear emulator caches for performance optimization"""
    try:
        current_state = get_game_state()
        cleared = []

        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]

            # Clear screen cache if available
            if hasattr(emulator, 'clear_screen_cache'):
                emulator.clear_screen_cache()
                cleared.append("screen_cache")

            # Clear server performance cache
            performance_monitor['frame_times'].clear()
            performance_monitor['encoding_times'].clear()
            cleared.append("performance_cache")

            logger.info(f"Cleared emulator caches: {cleared}")
            return jsonify({
                "message": "Caches cleared successfully",
                "cleared_caches": cleared,
                "cache_size_after": {
                    "performance_monitor": {
                        "frame_times": len(performance_monitor['frame_times']),
                        "encoding_times": len(performance_monitor['encoding_times'])
                    }
                }
            }), 200
        else:
            return jsonify({"error": "No emulator running"}), 400

    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        return jsonify({"error": str(e)}), 500


def _get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0.0

# Missing endpoints for GLM4.5-UI compatibility
@app.route('/characters', methods=['GET'])
def get_characters():
    """Get character information (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"characters": []}), 200

        # Placeholder data - in a real implementation, this would extract character data from the game
        characters = [
            {"id": 1, "name": "Player", "level": 1, "hp": 100, "max_hp": 100},
            {"id": 2, "name": "Rival", "level": 5, "hp": 150, "max_hp": 150}
        ]

        return jsonify({"characters": characters}), 200
    except Exception as e:
        logger.error(f"Error getting characters: {e}")
        return jsonify({"characters": []}), 200

@app.route('/memory', methods=['POST'])
def get_memory():
    """Get memory data (placeholder for GLM4.5-UI compatibility)"""
    try:
        data = request.get_json()
        address = data.get('address', 0)
        size = data.get('size', 1)

        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"values": []}), 200

        emulator = emulators[current_state["active_emulator"]]

        # Placeholder implementation - in real PyBoy, this would read actual memory
        memory_values = [0x00] * size

        return jsonify({"values": memory_values}), 200
    except Exception as e:
        logger.error(f"Error getting memory: {e}")
        return jsonify({"values": []}), 200

@app.route('/save_state', methods=['POST'])
def save_state():
    """Save game state (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Placeholder implementation
        return jsonify({"success": True, "message": "State saved"}), 200
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return jsonify({"error": "Failed to save state"}), 500

@app.route('/load_state', methods=['POST'])
def load_state():
    """Load game state (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Placeholder implementation
        return jsonify({"success": True, "message": "State loaded"}), 200
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return jsonify({"error": "Failed to load state"}), 500

@app.route('/api/save_state', methods=['POST'])
def api_save_state():
    """Save game state API endpoint for frontend compatibility"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Use the emulator's save state method if available
        if hasattr(emulator, 'save_state'):
            state_data = emulator.save_state()
            if state_data:
                return jsonify({"success": True, "message": "State saved successfully"}), 200
            else:
                return jsonify({"error": "Failed to save state data"}), 500
        else:
            # Fallback for emulators without save state support
            return jsonify({"success": True, "message": "State saved (placeholder)"}), 200

    except Exception as e:
        logger.error(f"Error saving state: {e}")
        return jsonify({"error": "Failed to save state"}), 500

@app.route('/api/load_state', methods=['POST'])
def api_load_state():
    """Load game state API endpoint for frontend compatibility"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Use the emulator's load state method if available
        if hasattr(emulator, 'load_state'):
            # Note: In a real implementation, you'd get the state data from the request
            # For now, we'll return a success response
            return jsonify({"success": True, "message": "State loaded successfully"}), 200
        else:
            # Fallback for emulators without load state support
            return jsonify({"success": True, "message": "State loaded (placeholder)"}), 200

    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return jsonify({"error": "Failed to load state"}), 500

@app.route('/tilemap', methods=['GET'])
def get_tilemap():
    """Get tilemap data (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"width": 20, "height": 18, "tiles": []}), 200

        # Placeholder tilemap data
        tiles = [[0 for _ in range(20)] for _ in range(18)]

        return jsonify({
            "width": 20,
            "height": 18,
            "tiles": tiles
        }), 200
    except Exception as e:
        logger.error(f"Error getting tilemap: {e}")
        return jsonify({"width": 20, "height": 18, "tiles": []}), 200

@app.route('/sprites', methods=['GET'])
def get_sprites():
    """Get sprite information (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"sprites": []}), 200

        # Placeholder sprite data
        sprites = [
            {"x": 100, "y": 100, "tile": 1, "attributes": 0},
            {"x": 120, "y": 100, "tile": 2, "attributes": 0}
        ]

        return jsonify({"sprites": sprites}), 200
    except Exception as e:
        logger.error(f"Error getting sprites: {e}")
        return jsonify({"sprites": []}), 200

@app.route('/start', methods=['POST'])
def start_emulator():
    """Start emulator (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if current_state["rom_loaded"] and current_state["active_emulator"]:
            return jsonify({"success": True, "message": "Emulator already running"}), 200

        return jsonify({"error": "No ROM loaded"}), 400
    except Exception as e:
        logger.error(f"Error starting emulator: {e}")
        return jsonify({"error": "Failed to start emulator"}), 500

@app.route('/stop', methods=['POST'])
def stop_emulator():
    """Stop emulator (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"success": True, "message": "Emulator already stopped"}), 200

        # Stop emulator logic here
        return jsonify({"success": True, "message": "Emulator stopped"}), 200
    except Exception as e:
        logger.error(f"Error stopping emulator: {e}")
        return jsonify({"error": "Failed to stop emulator"}), 500

@app.route('/pause', methods=['POST'])
def pause_emulator():
    """Pause emulator (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        return jsonify({"success": True, "message": "Emulator paused"}), 200
    except Exception as e:
        logger.error(f"Error pausing emulator: {e}")
        return jsonify({"error": "Failed to pause emulator"}), 500

@app.route('/resume', methods=['POST'])
def resume_emulator():
    """Resume emulator (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        return jsonify({"success": True, "message": "Emulator resumed"}), 200
    except Exception as e:
        logger.error(f"Error resuming emulator: {e}")
        return jsonify({"error": "Failed to resume emulator"}), 500

@app.route('/reset', methods=['POST'])
def reset_emulator():
    """Reset emulator (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]
        if hasattr(emulator, 'reset'):
            emulator.reset()

        return jsonify({"success": True, "message": "Emulator reset"}), 200
    except Exception as e:
        logger.error(f"Error resetting emulator: {e}")
        return jsonify({"error": "Failed to reset emulator"}), 500

@app.route('/status', methods=['GET'])
def get_emulator_status():
    """Get emulator status (placeholder for GLM4.5-UI compatibility)"""
    try:
        current_state = get_game_state()

        status = {
            "isRunning": current_state["rom_loaded"] and current_state["active_emulator"] is not None,
            "isPaused": False,
            "romLoaded": current_state["rom_loaded"],
            "fps": 60,
            "frameCount": 0,
            "gameTitle": current_state.get("rom_path", "Unknown") if current_state["rom_loaded"] else "No ROM"
        }

        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"isRunning": False, "isPaused": False, "romLoaded": False, "fps": 0, "frameCount": 0, "gameTitle": "Unknown"}), 200

@app.route('/speed', methods=['POST'])
def set_emulator_speed():
    """Set emulator speed (placeholder for GLM4.5-UI compatibility)"""
    try:
        data = request.get_json()
        speed = data.get('speed', 1.0)

        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        # Update speed in game state
        update_game_state({"speed_multiplier": speed})

        return jsonify({"success": True, "message": f"Speed set to {speed}x"}), 200
    except Exception as e:
        logger.error(f"Error setting speed: {e}")
        return jsonify({"error": "Failed to set speed"}), 500

@app.route('/sound', methods=['POST'])
def enable_sound():
    """Enable/disable sound (placeholder for GLM4.5-UI compatibility)"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)

        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        return jsonify({"success": True, "message": f"Sound {'enabled' if enabled else 'disabled'}"}), 200
    except Exception as e:
        logger.error(f"Error setting sound: {e}")
        return jsonify({"error": "Failed to set sound"}), 500

@app.route('/input', methods=['POST'])
def send_input():
    """Send input to emulator (for GLM4.5-UI compatibility)"""
    try:
        data = request.get_json()
        button = data.get('button', 'A').upper()
        press = data.get('press', True)

        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400

        emulator = emulators[current_state["active_emulator"]]

        # Map GLM4.5-UI button names to emulator actions
        action_mapping = {
            'A': 'A',
            'B': 'B',
            'UP': 'UP',
            'DOWN': 'DOWN',
            'LEFT': 'LEFT',
            'RIGHT': 'RIGHT',
            'START': 'START',
            'SELECT': 'SELECT'
        }

        action = action_mapping.get(button, 'NOOP')
        frames = 1 if press else 0

        success = emulator.step(action, frames)

        if success:
            return jsonify({"success": True, "message": f"Input {button} sent"}), 200
        else:
            return jsonify({"error": "Failed to send input"}), 500
    except Exception as e:
        logger.error(f"Error sending input: {e}")
        return jsonify({"error": "Failed to send input"}), 500

@app.route('/api/screen/debug', methods=['GET'])
def get_screen_debug():
    """Debug endpoint to test screen capture functionality"""
    try:
        current_state = get_game_state()
        
        # Always return debug info, even without ROM
        debug_info = {
            "rom_loaded": current_state["rom_loaded"],
            "active_emulator": current_state.get("active_emulator"),
            "rom_path": current_state.get("rom_path"),
            "timestamp": time.time(),
        }
        
        # If ROM is loaded, get detailed debug info
        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]

            # Get emulator info
            info = emulator.get_info() if hasattr(emulator, 'get_info') else {}
            debug_info["emulator_info"] = info

            # Get screen array
            screen_array = emulator.get_screen()

            debug_info.update({
                "screen_shape": screen_array.shape if screen_array is not None else None,
                "screen_dtype": str(screen_array.dtype) if screen_array is not None else None,
                "screen_min": int(screen_array.min()) if screen_array is not None else None,
                "screen_max": int(screen_array.max()) if screen_array is not None else None,
                "screen_size": screen_array.size if screen_array is not None else None,
            })

            # Try base64 conversion
            if screen_array is not None and screen_array.size > 0:
                img_base64 = numpy_to_base64_image(screen_array)
                debug_info["base64_success"] = img_base64 is not None and len(img_base64) > 0
                debug_info["base64_length"] = len(img_base64) if img_base64 else 0
                debug_info["base64_preview"] = img_base64[:100] + "..." if img_base64 and len(img_base64) > 100 else img_base64
            else:
                debug_info["base64_success"] = False
                debug_info["base64_length"] = 0
                debug_info["base64_preview"] = None
        else:
            # No ROM loaded - return partial debug info
            debug_info["error"] = "No ROM loaded"
            debug_info["screen_shape"] = None
            debug_info["screen_dtype"] = None
            debug_info["base64_success"] = False
            debug_info["base64_length"] = 0
            debug_info["base64_preview"] = None

        return jsonify(debug_info), 200 if current_state["rom_loaded"] else 503

    except Exception as e:
        logger.error(f"Error in debug screen endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Debug endpoint error: {str(e)}"}), 500

@app.route('/api/stream', methods=['GET'])
def stream_screen():
    """Simple, stable SSE endpoint for live screen streaming."""
    def generate():
        logger.info("SSE stream requested")

        # verify ROM/emulator exists before opening loop
        state = get_game_state()
        if not state.get("rom_loaded") or not state.get("active_emulator"):
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            return

        emulator = emulators[state["active_emulator"]]
        target_fps = 30
        frame_interval = 1.0 / target_fps
        consecutive_errors = 0
        max_consecutive_errors = 20

        # initial status event
        yield f"data: {json.dumps({'status': 'stream_started', 'fps': target_fps})}\n\n"

        while True:
            loop_start = time.time()
            try:
                # re-check state in case ROM was unloaded
                state = get_game_state()
                if not state.get("rom_loaded") or not state.get("active_emulator"):
                    yield f"data: {json.dumps({'error': 'ROM unloaded'})}\n\n"
                    break

                # advance emulator and capture current frame
                emulator.step('NOOP', 1)
                screen_array = emulator.get_screen()
                if screen_array is None or getattr(screen_array, 'size', 0) == 0:
                    raise RuntimeError('Empty screen buffer')

                img_base64 = numpy_to_base64_image(screen_array)
                if not img_base64:
                    raise RuntimeError('Failed to encode frame')

                payload = {
                    'image': img_base64,
                    'timestamp': time.time(),
                    'frame': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else None,
                    'fps': target_fps,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                consecutive_errors = 0

            except GeneratorExit:
                logger.info("SSE client disconnected")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"SSE stream frame error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                # do not immediately kill stream; send heartbeat error and continue
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
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/ws/status', methods=['GET'])
def get_websocket_status():
    """Get WebSocket server status."""
    return jsonify({
        "running": ws_server_running,
        "port": WS_PORT,
        "url": f"ws://localhost:{WS_PORT}/api/ws/stream",
        "clients": len(ws_clients),
        "timestamp": datetime.now().isoformat()
    }), 200


# WebSocket control endpoint
@app.route('/api/ws/start', methods=['POST'])
def start_websocket_endpoint():
    """Start the WebSocket server."""
    try:
        start_websocket_server()
        return jsonify({
            "success": True,
            "message": "WebSocket server started",
            "url": f"ws://localhost:{WS_PORT}/api/ws/stream",
            "port": WS_PORT,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/ws/stop', methods=['POST'])
def stop_websocket_endpoint():
    """Stop the WebSocket server."""
    try:
        stop_websocket_server()
        return jsonify({
            "success": True,
            "message": "WebSocket server stopped",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Failed to stop WebSocket server: {e}")
        return jsonify({"error": str(e)}), 500


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
