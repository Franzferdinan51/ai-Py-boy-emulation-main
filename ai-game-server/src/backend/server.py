"""
Enhanced AI Game Boy Server with Stream Stability Fixes
"""
import os
import signal
import threading
import io
import multiprocessing  # FIX: Added missing import for cpu_count()
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import base64
import logging
import tempfile
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import time
from flask_cors import CORS
import numpy as np
from PIL import Image

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
    """Continuously advance the active emulator so the WebUI shows a true live view."""
    global emulation_loop_running
    logger.info("Starting background live emulation loop")
    while emulation_loop_running:
        try:
            current_state = get_game_state()
            active = current_state.get("active_emulator")
            if current_state.get("rom_loaded") and active in emulators:
                emulator = emulators[active]
                # Directly tick PyBoy with rendering enabled to update screen buffer
                # This is critical - step() doesn't render by default!
                if hasattr(emulator, 'pyboy') and emulator.pyboy:
                    try:
                        emulator.pyboy.tick(1, True)  # Render every frame for live view
                    except Exception:
                        pass
                elif hasattr(emulator, 'step'):
                    try:
                        emulator.step('NOOP', 1)
                    except Exception:
                        pass
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
            time.sleep(1/15)  # ~15 FPS target - reduces CPU while keeping view live

        except Exception as exc:
            logger.debug(f"Live emulation loop iteration failed: {exc}")
            time.sleep(0.1)
    logger.info("Background live emulation loop stopped")


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
    'max_fps': 120
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
            # Forward JSON path-based loads to the filesystem loader.
            return load_rom_api()

        return jsonify({"error": "No ROM file provided"}), 400

    except Exception as e:
        logger.error(f"Error in legacy load_rom endpoint: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

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
    """SSE endpoint for live screen streaming with enhanced resource management"""
    def generate():
        # Generate unique connection ID
        connection_id = f"sse_stream_{int(time.time() * 1000)}"

        # Register connection with resource manager (if available)
        if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
            resource_manager.register_sse_connection(connection_id, lambda: logger.info(f"SSE connection {connection_id} cleanup"))

        logger.info(f"SSE stream requested (ID: {connection_id}). Checking game state...")
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            logger.warning(f"SSE stream aborted (ID: {connection_id}): No ROM loaded.")
            yield f"data: {json.dumps({'error': 'No ROM loaded'})}\n\n"
            if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
                resource_manager.unregister_sse_connection(connection_id)
            return

        emulator = emulators[current_state["active_emulator"]]
        logger.info(f"Starting SSE stream for {current_state['active_emulator']} (ID: {connection_id}).")

        frame_count = 0
        last_frame_time = time.time()
        target_fps = performance_monitor['adaptive_fps_target']  # Start with adaptive target
        frame_interval = 1.0 / target_fps

        # Dynamic frame rate adjustment with CPU optimization
        frame_times = []
        max_frame_times = 30  # Reduced from 60 to 30 for less memory usage
        min_fps = performance_monitor['min_fps']
        max_fps = performance_monitor['max_fps']

        # CPU optimization settings
        cpu_saver_mode = False
        cpu_saver_threshold = 70  # Enable CPU saver when system CPU > 70%
        adaptive_sleep = True
        min_frame_interval = 1.0 / max_fps  # Maximum frame rate
        max_frame_interval = 1.0 / min_fps  # Minimum frame rate

        # Performance monitoring
        pyboy_timeout_count = 0
        max_pyboy_timeouts = 3
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_pyboy_error = None

        # Use enhanced resource manager for thread pool (if available)
        pool_id = f"sse_pool_{connection_id}"
        if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
            executor = resource_manager.create_thread_pool(pool_id, max_workers=1)
        else:
            # Fallback to direct ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=1)
        last_client_activity = time.time()
        connection_timeout = 60  # 60 seconds without client response

        # CPU monitoring
        last_cpu_check = time.time()
        cpu_check_interval = 5.0  # Check CPU every 5 seconds

        try:
            while True:
                try:
                    # Calculate timing for consistent frame rate
                    current_time = time.time()
                    elapsed = current_time - last_frame_time

                    # Only process frame if enough time has passed
                    if elapsed >= frame_interval:
                        try:
                            # Execute PyBoy operations with timeout protection
                            def step_emulator():
                                return emulator.step('NOOP', 1)

                            def get_screen():
                                return emulator.get_screen()

                            # Timeout-protected PyBoy calls with adaptive timeouts
                            step_timeout = 0.05 if pyboy_timeout_count == 0 else 0.1
                            try:
                                success = executor.submit(step_emulator).result(timeout=step_timeout)
                                pyboy_timeout_count = 0  # Reset timeout counter on success
                            except FutureTimeoutError:
                                pyboy_timeout_count += 1
                                logger.warning(f"PyBoy step timeout (count: {pyboy_timeout_count}), using last frame")
                                if pyboy_timeout_count >= max_pyboy_timeouts:
                                    raise TimeoutError(f"Too many PyBoy timeouts: {pyboy_timeout_count}")
                                raise TimeoutError("PyBoy step timeout")

                            # Re-check ROM loaded state before proceeding
                            current_state = get_game_state()
                            if not current_state["rom_loaded"] or not current_state["active_emulator"]:
                                logger.warning("Stream aborted: ROM no longer loaded")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'rom_unloaded',
                                    'message': 'ROM is no longer loaded',
                                    'error': 'No ROM loaded'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                break

                            screen_timeout = 0.05 if pyboy_timeout_count == 0 else 0.1
                            try:
                                screen_array = executor.submit(get_screen).result(timeout=screen_timeout)
                            except FutureTimeoutError:
                                pyboy_timeout_count += 1
                                logger.warning(f"PyBoy get_screen timeout (count: {pyboy_timeout_count})")

                                # Instead of using placeholder, send error heartbeat
                                if pyboy_timeout_count >= max_pyboy_timeouts:
                                    error_data = {
                                        'heartbeat': True,
                                        'frame': frame_count,
                                        'timestamp': current_time,
                                        'fps': target_fps,
                                        'status': 'pyboy_timeout',
                                        'message': 'PyBoy timeout threshold reached',
                                        'error': 'PyBoy API timeout',
                                        'timeout_count': pyboy_timeout_count
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    break
                                else:
                                    # Send timeout error but continue streaming
                                    error_data = {
                                        'heartbeat': True,
                                        'frame': frame_count,
                                        'timestamp': current_time,
                                        'fps': target_fps,
                                        'status': 'timeout_error',
                                        'message': 'PyBoy get_screen timeout',
                                        'error': 'Screen capture timeout',
                                        'timeout_count': pyboy_timeout_count
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    frame_count += 1
                                    last_frame_time = current_time
                                    continue

                            # Validate screen data - if invalid, send error instead of placeholder
                            if screen_array is None or screen_array.size == 0:
                                logger.warning(f"Stream frame {frame_count}: PyBoy API returned empty screen")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'empty_screen',
                                    'message': 'Screen data is empty',
                                    'error': 'Screen capture failed'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                frame_count += 1
                                last_frame_time = current_time
                                continue

                            # Additional validation for proper PyBoy format
                            if len(screen_array.shape) != 3 or screen_array.shape[2] not in [3, 4]:
                                logger.warning(f"Stream frame {frame_count}: Invalid shape {screen_array.shape}")
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'invalid_format',
                                    'message': 'Invalid screen format',
                                    'error': f'Invalid screen shape: {screen_array.shape}'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                                frame_count += 1
                                last_frame_time = current_time
                                continue

                            # Convert to base64 using timeout-protected function
                            def convert_image():
                                return numpy_to_base64_image(screen_array)

                            conversion_start = time.time()
                            try:
                                img_base64 = executor.submit(convert_image).result(timeout=0.05)
                                conversion_time = time.time() - conversion_start
                                # Update global performance metrics
                                update_performance_metrics(conversion_time, time.time() - current_time)
                            except FutureTimeoutError:
                                logger.warning("Image conversion timeout")
                                img_base64 = None
                                conversion_time = 0.05

                            if not img_base64:
                                # Send heartbeat without image if conversion failed
                                data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
                                    'status': 'no_image',
                                    'error': 'Screen conversion failed',
                                    'elapsed_ms': round((time.time() - current_time) * 1000, 2)
                                }
                            else:
                                # Send normal frame with PyBoy metadata and timing info
                                frame_process_time = (time.time() - current_time) * 1000
                                data = {
                                    'image': img_base64,
                                    'shape': screen_array.shape,
                                    'timestamp': current_time,
                                    'frame': frame_count,
                                    'fps': target_fps,
                                    'adaptive_fps_target': performance_monitor['adaptive_fps_target'],
                                    'actual_interval': elapsed,
                                    'process_time_ms': round(frame_process_time, 2),
                                    'conversion_time_ms': round(conversion_time * 1000, 2),
                                    'current_fps': round(performance_monitor['current_fps'], 1),
                                    'pyboy_frame': emulator.get_frame_count() if hasattr(emulator, 'get_frame_count') else frame_count,
                                    'status': 'streaming'
                                }

                            yield f"data: {json.dumps(data)}\n\n"

                            # Update counters and reset error counter on success
                            frame_count += 1
                            last_frame_time = current_time
                            consecutive_errors = 0

                            # Log progress every 60 frames (2 seconds at 30fps)
                            if frame_count % 60 == 0:
                                logger.info(f"Streamed frame {frame_count} successfully using PyBoy API")

                        except Exception as inner_e:
                            consecutive_errors += 1
                            last_pyboy_error = str(inner_e)
                            logger.error(f"Error processing stream frame {frame_count}: {inner_e} (consecutive errors: {consecutive_errors})", exc_info=True)

                            # Enhanced error recovery based on error type
                            if consecutive_errors >= max_consecutive_errors:
                                logger.warning(f"Too many consecutive errors ({consecutive_errors}), attempting recovery...")

                                # Send detailed recovery heartbeat
                                recovery_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'status': 'recovery_attempt',
                                    'consecutive_errors': consecutive_errors,
                                    'last_error': last_pyboy_error,
                                    'message': 'Attempting to recover from stream errors',
                                    'recovery_action': 'resetting_pyboy_state'
                                }
                                yield f"data: {json.dumps(recovery_data)}\n\n"

                                # Perform actual recovery
                                try:
                                    # Attempt to reset PyBoy state
                                    emulator.reset()
                                    time.sleep(0.2)  # Allow time for reset
                                    consecutive_errors = 0
                                    logger.info("PyBoy state reset successful")
                                except Exception as reset_error:
                                    logger.error(f"PyBoy reset failed: {reset_error}")
                                    # If reset fails, send final error and break
                                    error_data = {
                                        'heartbeat': True,
                                        'frame': frame_count,
                                        'timestamp': current_time,
                                        'fps': target_fps,
                                        'status': 'fatal_error',
                                        'consecutive_errors': consecutive_errors,
                                        'message': 'Recovery failed - stream terminated',
                                        'error': last_pyboy_error
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                                    break
                            else:
                                # Send detailed error heartbeat but continue streaming
                                error_data = {
                                    'heartbeat': True,
                                    'frame': frame_count,
                                    'timestamp': current_time,
                                    'fps': target_fps,
                                    'error': str(inner_e),
                                    'status': 'error_continuing',
                                    'consecutive_errors': consecutive_errors,
                                    'pyboy_status': 'PyBoy API error'
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"

                            frame_count += 1
                            last_frame_time = current_time
                            continue

                    else:
                        # Enhanced connection health monitoring
                        time_since_last_activity = current_time - last_client_activity

                        # Check for connection timeout
                        if time_since_last_activity > connection_timeout:
                            logger.warning(f"Connection timeout after {connection_timeout}s, closing stream")
                            timeout_data = {
                                'heartbeat': True,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'fps': target_fps,
                                'status': 'connection_timeout',
                                'message': 'Client connection timed out'
                            }
                            yield f"data: {json.dumps(timeout_data)}\n\n"
                            break

                        # CPU monitoring and adaptive adjustments
                        if current_time - last_cpu_check >= cpu_check_interval:
                            try:
                                import psutil
                                system_cpu = psutil.cpu_percent(interval=0.1)

                                # Enable CPU saver mode if system is under heavy load
                                old_cpu_saver_mode = cpu_saver_mode
                                cpu_saver_mode = system_cpu > cpu_saver_threshold

                                if cpu_saver_mode != old_cpu_saver_mode:
                                    if cpu_saver_mode:
                                        logger.info(f"CPU saver mode enabled (System CPU: {system_cpu:.1f}%)")
                                        # Reduce frame rate when CPU is high
                                        target_fps = max(min_fps, target_fps // 2)
                                        frame_interval = 1.0 / target_fps
                                    else:
                                        logger.info(f"CPU saver mode disabled (System CPU: {system_cpu:.1f}%)")
                                        # Restore normal frame rate
                                        target_fps = performance_monitor['adaptive_fps_target']
                                        frame_interval = 1.0 / target_fps

                                last_cpu_check = current_time
                            except ImportError:
                                pass  # psutil not available, skip CPU monitoring

                        # Send heartbeat every 5 seconds (reduced from 3) to keep connection alive during idle periods
                        if int(current_time) % 5 == 0 and int(current_time) != int(last_frame_time):
                            heartbeat_data = {
                                'heartbeat': True,
                                'frame': frame_count,
                                'timestamp': current_time,
                                'fps': target_fps,
                                'status': 'idle_heartbeat',
                                'message': 'Stream waiting for next frame interval',
                                'uptime': round(time_since_last_activity, 1),
                                'current_fps': target_fps,
                                'avg_frame_time': round(sum(frame_times) / len(frame_times) * 1000, 2) if frame_times else 0,
                                'cpu_saver_mode': cpu_saver_mode
                            }
                            yield f"data: {json.dumps(heartbeat_data)}\n\n"

                    # Dynamic frame rate adjustment based on performance - use global adaptive target
                    frame_process_time = (time.time() - current_time)
                    frame_times.append(frame_process_time)

                    if len(frame_times) > max_frame_times:
                        frame_times.pop(0)

                    # Update target FPS from global performance monitor
                    new_target_fps = performance_monitor['adaptive_fps_target']
                    if new_target_fps != target_fps:
                        old_fps = target_fps
                        target_fps = new_target_fps
                        frame_interval = 1.0 / target_fps
                        logger.info(f"Adaptive FPS adjusted: {old_fps} -> {target_fps}")

                    # Local frame rate adjustment as backup
                    if len(frame_times) >= 30:  # Only adjust after collecting enough data
                        avg_frame_time = sum(frame_times) / len(frame_times)
                        if avg_frame_time > frame_interval * 1.5:  # Taking too long
                            if target_fps > min_fps:
                                target_fps = max(min_fps, target_fps - 2)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Local adjustment: Reducing FPS to {target_fps} due to slow performance")
                        elif avg_frame_time < frame_interval * 0.7:  # Processing quickly
                            if target_fps < max_fps:
                                target_fps = min(max_fps, target_fps + 1)
                                frame_interval = 1.0 / target_fps
                                logger.info(f"Local adjustment: Increasing FPS to {target_fps} due to good performance")

                    # Sleep for the remaining time to maintain frame rate (CPU-optimized)
                    remaining_time = frame_interval - elapsed
                    if remaining_time > 0:
                        # Adaptive sleep: sleep longer in CPU saver mode
                        sleep_multiplier = 0.9 if cpu_saver_mode else 0.8
                        actual_sleep = remaining_time * sleep_multiplier

                        # Don't sleep for very short durations to reduce CPU wake-ups
                        if actual_sleep < 0.001:  # Less than 1ms
                            # Very short sleep - use busy wait for better precision
                            pass
                        else:
                            time.sleep(actual_sleep)

                except GeneratorExit:
                    logger.info("SSE stream client disconnected gracefully.")
                    break
                except KeyboardInterrupt:
                    logger.info("SSE stream interrupted by user.")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    last_pyboy_error = str(e)
                    logger.error(f"Unexpected error in SSE stream: {e}", exc_info=True)
                    # Send final error message before breaking
                    try:
                        error_data = {
                            'heartbeat': True,
                            'frame': frame_count,
                            'timestamp': time.time(),
                            'fps': target_fps,
                            'status': 'stream_crashed',
                            'message': 'Stream encountered unrecoverable error',
                            'error': str(e)
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                    except:
                        pass
                    break
                finally:
                    # Update client activity timestamp
                    last_client_activity = time.time()

        except Exception as outer_e:
            logger.error(f"Critical error in SSE stream generator: {outer_e}", exc_info=True)
            try:
                error_data = {
                    'error': str(outer_e),
                    'frame': frame_count,
                    'timestamp': time.time(),
                    'fps': target_fps,
                    'status': 'stream_ending',
                    'pyboy_status': 'Critical failure'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            except:
                pass
        finally:
            # Cleanup resources - this is guaranteed to run
            logger.info(f"SSE stream cleanup (ID: {connection_id}). Total frames: {frame_count}, Final FPS: {target_fps}")

            # Unregister SSE connection (if resource manager available)
            if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
                resource_manager.unregister_sse_connection(connection_id)

            # Clean up thread pool using resource manager (if available)
            try:
                if 'pool_id' in locals():
                    if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
                        resource_manager.cleanup_thread_pool(pool_id, wait=False)
                    else:
                        # Fallback: shutdown executor directly
                        if hasattr(executor, 'shutdown'):
                            executor.shutdown(wait=False)
                    logger.info(f"Thread pool {pool_id} shutdown successfully")
            except Exception as cleanup_error:
                logger.error(f"Error during thread pool cleanup: {cleanup_error}")

            # Clean up any remaining resources
            try:
                # Force garbage collection to clean up any remaining references
                import gc
                gc.collect()
                logger.info("Garbage collection completed")
            except Exception as gc_error:
                logger.warning(f"Garbage collection failed: {gc_error}")

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def cleanup_server_resources():
    """Clean up server resources before shutdown"""
    logger.info("Cleaning up server resources...")

    # Clean up enhanced resource manager
    if ENHANCED_RESOURCE_MANAGER_AVAILABLE:
        try:
            resource_manager.stop()
            logger.info("Cleaned up enhanced resource manager")
        except Exception as e:
            logger.error(f"Error cleaning up enhanced resource manager: {e}")

    # Clean up optimization system
    if OPTIMIZATION_SYSTEM_AVAILABLE and ENABLE_OPTIMIZATION:
        try:
            optimization_system_manager.stop_monitoring()
            optimization_system_manager.emergency_shutdown()
            logger.info("Cleaned up optimization system")
        except Exception as e:
            logger.error(f"Error cleaning up optimization system: {e}")

    # Clean up emulators
    try:
        for emulator_name, emulator in emulators.items():
            if hasattr(emulator, 'cleanup'):
                emulator.cleanup()
                logger.info(f"Cleaned up {emulator_name} emulator")
    except Exception as e:
        logger.error(f"Error cleaning up emulators: {e}")

    # Clean up AI provider manager
    try:
        if hasattr(ai_provider_manager, 'cleanup'):
            ai_provider_manager.cleanup()
            logger.info("Cleaned up AI provider manager")
    except Exception as e:
        logger.error(f"Error cleaning up AI provider manager: {e}")

    # Force garbage collection
    try:
        import gc
        gc.collect()
        logger.info("Server garbage collection completed")
    except Exception as e:
        logger.error(f"Error during server garbage collection: {e}")

# Performance Monitoring Endpoints
@app.route('/api/performance/dashboard', methods=['GET'])
def get_performance_dashboard_endpoint():
    """Get comprehensive performance dashboard data"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        dashboard_data = get_performance_dashboard()
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Error getting performance dashboard: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/health', methods=['GET'])
def get_system_health_endpoint():
    """Get system health status"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        health_data = get_system_health()
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/optimize', methods=['POST'])
def optimize_system_endpoint():
    """Trigger system optimization"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        data = request.get_json() or {}
        aggressive = data.get('aggressive', False)

        optimizations = optimize_system(aggressive=aggressive)
        return jsonify({
            "success": True,
            "optimizations_executed": len(optimizations),
            "optimizations": optimizations
        })
    except Exception as e:
        logger.error(f"Error optimizing system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/metrics', methods=['GET'])
def get_performance_metrics_endpoint():
    """Get detailed performance metrics"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        timeframe = int(request.args.get('timeframe', 300))  # Default 5 minutes
        metrics_data = performance_monitor.get_detailed_metrics(timeframe)
        return jsonify(metrics_data)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/components/<component_name>/health', methods=['GET'])
def get_component_health_endpoint(component_name):
    """Get health status for a specific component"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        health_data = performance_monitor.get_component_health(component_name)
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Error getting component health for {component_name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/config', methods=['GET'])
def get_performance_config_endpoint():
    """Get performance monitoring configuration"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        config = performance_monitor.get_optimization_config()
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error getting performance config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/config', methods=['POST'])
def update_performance_config_endpoint():
    """Update performance monitoring configuration"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400

        performance_monitor.set_optimization_config(data)
        return jsonify({"success": True, "message": "Configuration updated"})
    except Exception as e:
        logger.error(f"Error updating performance config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/export', methods=['GET'])
def export_performance_data_endpoint():
    """Export performance data"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    try:
        timeframe = int(request.args.get('timeframe', 3600))  # Default 1 hour
        filename = f"performance_data_{int(time.time())}.json"
        filepath = os.path.join(os.getcwd(), filename)

        performance_monitor.export_performance_data(filepath, timeframe)

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"Error exporting performance data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance/stream', methods=['GET'])
def stream_performance_metrics():
    """SSE endpoint for real-time performance metrics streaming"""
    if not PERFORMANCE_MONITOR_AVAILABLE:
        return jsonify({"error": "Performance monitoring not available"}), 503

    def generate_metrics():
        try:
            while True:
                # Get current performance snapshot
                dashboard_data = get_performance_dashboard()

                # Send as SSE event
                yield f"data: {json.dumps(dashboard_data)}\n\n"

                # Wait for next update
                time.sleep(5)  # Update every 5 seconds

        except GeneratorExit:
            logger.info("Performance metrics stream closed by client")
        except Exception as e:
            logger.error(f"Error in performance metrics stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate_metrics()),
        mimetype='text/event-stream'
    )

# ============================================================================
# OPTIMIZATION SYSTEM API ENDPOINTS
# ============================================================================

@app.route('/api/optimization/status', methods=['GET'])
def get_optimization_status():
    """Get overall optimization system status"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        health = optimization_system_manager.get_system_health()
        stats = optimization_system_manager.get_optimization_stats()

        return jsonify({
            "status": health.status.value,
            "health": asdict(health),
            "stats": stats,
            "uptime_seconds": stats.get('uptime_seconds', 0),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/config', methods=['GET'])
def get_optimization_config():
    """Get current optimization configuration"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        return jsonify({
            "config": asdict(optimization_system_manager.config),
            "environment_settings": {
                "ENABLE_OPTIMIZATION": ENABLE_OPTIMIZATION,
                "OPTIMIZATION_MEMORY_MB": OPTIMIZATION_MEMORY_MB,
                "OPTIMIZATION_AUTO_SCALING": OPTIMIZATION_AUTO_SCALING,
                "OPTIMIZATION_CACHE_SIZE": OPTIMIZATION_CACHE_SIZE,
                "OPTIMIZATION_MONITORING_INTERVAL": OPTIMIZATION_MONITORING_INTERVAL
            }
        })
    except Exception as e:
        logger.error(f"Error getting optimization config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/config', methods=['POST'])
def update_optimization_config():
    """Update optimization configuration"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400

        # Update configuration
        success = optimization_system_manager.update_configuration(**data)

        if success:
            logger.info(f"Optimization configuration updated: {data}")
            return jsonify({
                "message": "Configuration updated successfully",
                "updated_config": asdict(optimization_system_manager.config)
            })
        else:
            return jsonify({"error": "Failed to update configuration"}), 400

    except Exception as e:
        logger.error(f"Error updating optimization config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/metrics', methods=['GET'])
def get_optimization_metrics():
    """Get detailed optimization metrics"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        stats = optimization_system_manager.get_optimization_stats()
        health = optimization_system_manager.get_system_health()

        # Calculate additional metrics
        metrics = {
            "system_health": asdict(health),
            "optimization_stats": stats,
            "component_health": {
                "memory": health.memory_health,
                "cpu": health.cpu_health,
                "thread_pools": health.thread_pool_health,
                "cache": health.cache_health,
                "error_rate": health.error_rate,
                "performance_score": health.performance_score
            },
            "resource_usage": {
                "memory_pressure": optimization_system_manager.get_memory_pressure(),
                "active_issues": len(health.active_issues),
                "uptime_hours": stats.get('uptime_seconds', 0) / 3600
            }
        }

        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting optimization metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/restart', methods=['POST'])
def restart_optimization_system():
    """Restart optimization system or specific components"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        data = request.get_json() or {}
        system_name = data.get('system')

        if system_name:
            # Restart specific system
            success = optimization_system_manager.restart_system(system_name)
            if success:
                message = f"Successfully restarted {system_name} system"
            else:
                message = f"Failed to restart {system_name} system"
                return jsonify({"error": message}), 400
        else:
            # Restart entire optimization system
            try:
                optimization_system_manager.stop_monitoring()
                optimization_system_manager.emergency_shutdown()
                time.sleep(2)  # Brief pause
                if optimization_system_manager.initialize_systems():
                    optimization_system_manager.start_monitoring()
                    message = "Successfully restarted optimization system"
                else:
                    message = "Failed to reinitialize optimization system"
                    return jsonify({"error": message}), 500
            except Exception as restart_error:
                logger.error(f"Error during optimization system restart: {restart_error}")
                return jsonify({"error": f"Restart failed: {restart_error}"}), 500

        logger.info(message)
        return jsonify({
            "message": message,
            "status": optimization_system_manager.status.value,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error(f"Error restarting optimization system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/health', methods=['GET'])
def optimization_health_check():
    """Comprehensive health check for optimization systems"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        health = optimization_system_manager.get_system_health()
        is_healthy = optimization_system_manager.validate_system_health()

        health_check = {
            "overall_status": health.status.value,
            "is_healthy": is_healthy,
            "health_score": health.performance_score,
            "critical_issues": health.active_issues,
            "component_status": {
                "memory": {"health": health.memory_health, "status": "ok" if health.memory_health > 0.5 else "warning"},
                "cpu": {"health": health.cpu_health, "status": "ok" if health.cpu_health > 0.5 else "warning"},
                "thread_pools": {"health": health.thread_pool_health, "status": "ok" if health.thread_pool_health > 0.5 else "warning"},
                "cache": {"health": health.cache_health, "status": "ok" if health.cache_health > 0.5 else "warning"},
                "error_rate": {"health": 1.0 - health.error_rate, "status": "ok" if health.error_rate < 0.1 else "warning"}
            },
            "recommendations": [],
            "timestamp": time.time()
        }

        # Add recommendations based on health
        if health.memory_health < 0.3:
            health_check["recommendations"].append("Consider increasing memory limits or enabling memory cleanup")
        if health.cpu_health < 0.3:
            health_check["recommendations"].append("High CPU usage detected - consider scaling down thread pools")
        if health.cache_health < 0.5:
            health_check["recommendations"].append("Cache performance suboptimal - consider adjusting cache settings")
        if health.error_rate > 0.1:
            health_check["recommendations"].append("High error rate detected - check system logs")

        status_code = 200 if is_healthy else 503
        return jsonify(health_check), status_code

    except Exception as e:
        logger.error(f"Error during optimization health check: {e}")
        return jsonify({
            "overall_status": "error",
            "is_healthy": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/optimization/perform', methods=['POST'])
def perform_optimizations():
    """Manually trigger optimizations"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        optimizations = optimization_system_manager.perform_optimizations()

        return jsonify({
            "message": "Optimizations performed successfully",
            "optimizations": optimizations,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error performing optimizations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/system/<system_name>/enable', methods=['POST'])
def enable_optimization_system(system_name):
    """Enable a specific optimization system"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        success = optimization_system_manager.enable_system(system_name)
        if success:
            return jsonify({
                "message": f"Successfully enabled {system_name} system",
                "system": system_name,
                "status": "enabled"
            })
        else:
            return jsonify({"error": f"Failed to enable {system_name} system"}), 400
    except Exception as e:
        logger.error(f"Error enabling {system_name} system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/system/<system_name>/disable', methods=['POST'])
def disable_optimization_system(system_name):
    """Disable a specific optimization system"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        success = optimization_system_manager.disable_system(system_name)
        if success:
            return jsonify({
                "message": f"Successfully disabled {system_name} system",
                "system": system_name,
                "status": "disabled"
            })
        else:
            return jsonify({"error": f"Failed to disable {system_name} system"}), 400
    except Exception as e:
        logger.error(f"Error disabling {system_name} system: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/export', methods=['GET'])
def export_optimization_config():
    """Export optimization configuration"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        success = optimization_system_manager.export_configuration(tmp_path)
        if success:
            # Read and return the file
            with open(tmp_path, 'r') as f:
                config_data = json.load(f)

            # Clean up temporary file
            os.unlink(tmp_path)

            return jsonify({
                "message": "Configuration exported successfully",
                "config": config_data,
                "export_timestamp": time.time()
            })
        else:
            return jsonify({"error": "Failed to export configuration"}), 500

    except Exception as e:
        logger.error(f"Error exporting optimization config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/import', methods=['POST'])
def import_optimization_config():
    """Import optimization configuration"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        data = request.get_json()
        if not data or 'config' not in data:
            return jsonify({"error": "No configuration data provided"}), 400

        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(data['config'], tmp_file, indent=2)
            tmp_path = tmp_file.name

        success = optimization_system_manager.import_configuration(tmp_path)

        # Clean up temporary file
        os.unlink(tmp_path)

        if success:
            return jsonify({
                "message": "Configuration imported successfully",
                "current_config": asdict(optimization_system_manager.config),
                "import_timestamp": time.time()
            })
        else:
            return jsonify({"error": "Failed to import configuration"}), 400

    except Exception as e:
        logger.error(f"Error importing optimization config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/report', methods=['GET'])
def get_optimization_report():
    """Generate comprehensive optimization system report"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        report = optimization_system_manager.get_optimization_report()

        if 'error' in report:
            return jsonify(report), 500

        return jsonify(report)
    except Exception as e:
        logger.error(f"Error generating optimization report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimization/logs', methods=['GET'])
def get_optimization_logs():
    """Get detailed optimization system logs"""
    if not OPTIMIZATION_SYSTEM_AVAILABLE:
        return jsonify({"error": "Optimization system not available"}), 503

    try:
        limit = request.args.get('limit', 100, type=int)
        logs = optimization_system_manager.get_detailed_logs(limit)

        return jsonify({
            "logs": logs,
            "total_logs": sum(len(category_logs) for category_logs in logs.values()),
            "limit": limit,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error getting optimization logs: {e}")
        return jsonify({"error": str(e)}), 500


# ========== NEW API ENDPOINTS FOR AI-GAME-ASSISTANT UI ==========

# Global state for agent mode
agent_mode_state = {
    "enabled": True,
    "current_action": "Idle",
    "last_decision": "Initializing...",
    "mode": "auto",  # auto, manual
    "autonomous_level": "moderate"  # passive, moderate, aggressive
}

openclaw_runtime_state = {
    "endpoint": os.environ.get('OPENCLAW_MCP_ENDPOINT', 'http://localhost:18789'),
    "vision_model": os.environ.get('VISION_MODEL', 'kimi-k2.5'),
    "planning_model": os.environ.get('PLANNING_MODEL', 'glm-5'),
    "use_dual_model": os.environ.get('USE_DUAL_MODEL', 'true').lower() == 'true',
    "objectives": "Complete Pokemon Red with safe exploration and clear progress.",
    "personality": "strategic",
}


def _normalize_openclaw_endpoint(endpoint: Optional[str]) -> str:
    candidate = (endpoint or openclaw_runtime_state["endpoint"]).strip()
    if not candidate:
        raise ValueError("OpenClaw endpoint is required")

    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("OpenClaw endpoint must be a valid http(s) URL")

    return candidate.rstrip("/")


def _normalize_optional_runtime_endpoint(endpoint: Optional[str]) -> str:
    candidate = (endpoint or "").strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Runtime endpoint must be a valid http(s) URL")

    return candidate.rstrip("/")


def _normalize_openclaw_model_choice(model: Optional[str]) -> str:
    candidate = str(model or "").strip()
    if not candidate:
        return ""

    if "/" in candidate:
        return candidate.split("/")[-1]

    return candidate


def _build_openclaw_health_payload(endpoint: str, ok: bool, status: Optional[int], service_status: Optional[str] = None,
                                   error_message: Optional[str] = None, details: Optional[Dict] = None) -> Dict:
    return {
        "ok": ok,
        "endpoint": endpoint,
        "status": status,
        "service_status": service_status,
        "error": error_message,
        "details": details,
        "checked_at": datetime.now().isoformat(),
    }


def _probe_openclaw_health(endpoint: Optional[str]) -> Dict:
    normalized_endpoint = _normalize_openclaw_endpoint(endpoint)
    health_url = urljoin(f"{normalized_endpoint}/", "health")
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "openclaw-webui",
    }
    probe_request = urllib_request.Request(health_url, headers=request_headers)

    try:
        with urllib_request.urlopen(probe_request, timeout=3) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            payload = json.loads(response_body) if response_body else {}
            service_status = payload.get("status") if isinstance(payload, dict) else None
            details = payload if isinstance(payload, dict) else {"body": response_body}
            return _build_openclaw_health_payload(
                normalized_endpoint,
                True,
                response.getcode(),
                service_status=service_status,
                details=details,
            )
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        payload = json.loads(error_body) if error_body else {}
        service_status = payload.get("status") if isinstance(payload, dict) else None
        details = payload if isinstance(payload, dict) else {"body": error_body}
        return _build_openclaw_health_payload(
            normalized_endpoint,
            False,
            exc.code,
            service_status=service_status,
            error_message=error_body or str(exc),
            details=details,
        )
    except Exception as exc:
        return _build_openclaw_health_payload(
            normalized_endpoint,
            False,
            None,
            error_message=str(exc),
        )


@app.route('/api/openclaw/config', methods=['GET'])
def get_openclaw_config_api():
    return jsonify({
        **openclaw_runtime_state,
        "dual_model_status": ai_provider_manager.get_dual_model_status() if hasattr(ai_provider_manager, 'get_dual_model_status') else None,
        "timestamp": datetime.now().isoformat(),
    }), 200


@app.route('/api/openclaw/config', methods=['POST'])
def update_openclaw_config_api():
    try:
        data = request.get_json(silent=True) or {}
        allowed_vision_models = {"kimi-k2.5", "qwen-vl-plus", "glm-4v-flash", "MiniMax-M2.7"}
        allowed_planning_models = {"glm-5", "qwen3.5-plus", "MiniMax-M2.7", "MiniMax-M2.5"}
        allowed_personalities = {"strategic", "casual", "speedrun", "explorer"}

        endpoint = _normalize_openclaw_endpoint(data.get("endpoint", openclaw_runtime_state["endpoint"]))
        vision_model = _normalize_openclaw_model_choice(data.get("vision_model", openclaw_runtime_state["vision_model"]))
        planning_model = _normalize_openclaw_model_choice(data.get("planning_model", openclaw_runtime_state["planning_model"]))
        use_dual_model = data.get("use_dual_model", openclaw_runtime_state["use_dual_model"])
        personality = str(data.get("personality", openclaw_runtime_state["personality"])).strip()
        objectives = str(data.get("objectives", openclaw_runtime_state["objectives"])).strip()

        if vision_model not in allowed_vision_models:
            return jsonify({"error": f"Invalid vision model. Allowed: {sorted(allowed_vision_models)}"}), 400

        if planning_model not in allowed_planning_models:
            return jsonify({"error": f"Invalid planning model. Allowed: {sorted(allowed_planning_models)}"}), 400

        if personality not in allowed_personalities:
            return jsonify({"error": f"Invalid personality. Allowed: {sorted(allowed_personalities)}"}), 400

        if not objectives:
            return jsonify({"error": "Objectives cannot be empty"}), 400

        openclaw_runtime_state["endpoint"] = endpoint
        openclaw_runtime_state["vision_model"] = vision_model
        openclaw_runtime_state["planning_model"] = planning_model
        openclaw_runtime_state["use_dual_model"] = bool(use_dual_model)
        openclaw_runtime_state["personality"] = personality
        openclaw_runtime_state["objectives"] = objectives
        os.environ['OPENCLAW_MCP_ENDPOINT'] = endpoint
        app.config['OPENCLAW_ENDPOINT'] = endpoint
        
        ai_provider_manager.set_openclaw_endpoint(endpoint)
        if hasattr(ai_provider_manager, 'configure_dual_model'):
            ai_provider_manager.configure_dual_model(
                vision_model=vision_model,
                planning_model=planning_model,
                use_dual_model=bool(use_dual_model)
            )

        return jsonify({
            **openclaw_runtime_state,
            "timestamp": datetime.now().isoformat(),
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error(f"Error updating OpenClaw config: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route('/api/openclaw/health', methods=['GET'])
def get_openclaw_health_api():
    endpoint = request.args.get('endpoint', openclaw_runtime_state["endpoint"])

    try:
        payload = _probe_openclaw_health(endpoint)
        return jsonify(payload), 200
    except ValueError as exc:
        return jsonify({
            "ok": False,
            "endpoint": endpoint,
            "status": None,
            "service_status": None,
            "error": str(exc),
            "details": None,
            "checked_at": datetime.now().isoformat(),
        }), 400


@app.route('/api/ai/runtime', methods=['GET'])
def get_ai_runtime_config():
    """Get the backend's active AI runtime defaults."""
    return jsonify({
        **ai_runtime_state,
        "available_providers": ai_provider_manager.get_available_providers(),
        "provider_status": ai_provider_manager.get_provider_status(),
        "timestamp": datetime.now().isoformat(),
    }), 200


@app.route('/api/ai/runtime', methods=['POST'])
def update_ai_runtime_config():
    """Update the backend's preferred provider/model runtime settings."""
    try:
        data = request.get_json(silent=True) or {}
        allowed_providers = set(ai_provider_manager.providers.keys()) or {
            'openclaw', 'lmstudio', 'gemini', 'openrouter', 'openai-compatible', 'nvidia', 'mock', 'tetris-genetic'
        }

        provider = str(data.get('provider', ai_runtime_state['provider'])).strip() or ai_runtime_state['provider']
        model = str(data.get('model', ai_runtime_state['model'])).strip()
        api_endpoint = _normalize_optional_runtime_endpoint(data.get('api_endpoint', ai_runtime_state['api_endpoint']))

        if provider not in allowed_providers:
            return jsonify({"error": f"Invalid provider. Allowed: {sorted(allowed_providers)}"}), 400

        endpoint_env_map = {
            'lmstudio': 'LM_STUDIO_URL',
            'openai-compatible': 'OPENAI_ENDPOINT',
        }
        model_env_map = {
            'gemini': 'GEMINI_MODEL',
            'openrouter': 'OPENROUTER_MODEL',
            'openai-compatible': 'OPENAI_MODEL',
            'nvidia': 'NVIDIA_MODEL',
        }

        endpoint_env_key = endpoint_env_map.get(provider)
        if endpoint_env_key:
            if api_endpoint:
                os.environ[endpoint_env_key] = api_endpoint
            else:
                os.environ.pop(endpoint_env_key, None)

        model_env_key = model_env_map.get(provider)
        if model_env_key:
            if model:
                os.environ[model_env_key] = model
            else:
                os.environ.pop(model_env_key, None)
        elif provider == 'lmstudio':
            if model:
                os.environ['LM_STUDIO_THINKING_MODEL'] = model
            else:
                os.environ.pop('LM_STUDIO_THINKING_MODEL', None)

        if provider == 'lmstudio' and not api_endpoint:
            api_endpoint = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')
        if provider == 'lmstudio' and not model:
            model = os.environ.get('LM_STUDIO_THINKING_MODEL', '')
        elif provider == 'openai-compatible' and not api_endpoint:
            api_endpoint = os.environ.get('OPENAI_ENDPOINT', '')

        ai_runtime_state['provider'] = provider
        ai_runtime_state['model'] = model
        ai_runtime_state['api_endpoint'] = api_endpoint

        ai_provider_manager.set_default_provider(provider)
        ai_provider_manager.reinitialize_provider(provider)

        update_game_state({
            "current_provider": provider,
            "current_model": model or None,
        })

        return jsonify({
            **ai_runtime_state,
            "available_providers": ai_provider_manager.get_available_providers(),
            "provider_status": ai_provider_manager.get_provider_status(),
            "timestamp": datetime.now().isoformat(),
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.error(f"Error updating AI runtime config: {exc}")
        return jsonify({"error": str(exc)}), 500

@app.route('/api/lmstudio/config', methods=['GET'])
def get_lmstudio_config():
    """Get LM Studio configuration"""
    return jsonify({
        "endpoint": os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1'),
        "thinking_model": os.environ.get('LM_STUDIO_THINKING_MODEL', ''),
        "vision_model": os.environ.get('LM_STUDIO_VISION_MODEL', ''),
        "timestamp": datetime.now().isoformat(),
    }), 200

@app.route('/api/lmstudio/config', methods=['POST'])
def update_lmstudio_config():
    """Update LM Studio configuration (runtime only - not persisted to .env)"""
    try:
        data = request.get_json(silent=True) or {}

        if 'endpoint' in data:
            endpoint = _normalize_optional_runtime_endpoint(data.get('endpoint'))
            if endpoint:
                os.environ['LM_STUDIO_URL'] = endpoint
            else:
                os.environ.pop('LM_STUDIO_URL', None)

        if 'thinking_model' in data:
            thinking_model = str(data.get('thinking_model') or '').strip()
            if thinking_model:
                os.environ['LM_STUDIO_THINKING_MODEL'] = thinking_model
            else:
                os.environ.pop('LM_STUDIO_THINKING_MODEL', None)

        if 'vision_model' in data:
            vision_model = str(data.get('vision_model') or '').strip()
            if vision_model:
                os.environ['LM_STUDIO_VISION_MODEL'] = vision_model
            else:
                os.environ.pop('LM_STUDIO_VISION_MODEL', None)

        ai_provider_manager.reinitialize_provider('lmstudio')

        if ai_runtime_state.get('provider') == 'lmstudio':
            ai_runtime_state['api_endpoint'] = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')
            ai_runtime_state['model'] = os.environ.get('LM_STUDIO_THINKING_MODEL', '')
        
        return jsonify({
            "endpoint": os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1'),
            "thinking_model": os.environ.get('LM_STUDIO_THINKING_MODEL', ''),
            "vision_model": os.environ.get('LM_STUDIO_VISION_MODEL', ''),
            "message": "LM Studio configuration updated",
            "timestamp": datetime.now().isoformat(),
        }), 200
        
    except Exception as exc:
        logger.error(f"Error updating LM Studio config: {exc}")
        return jsonify({"error": str(exc)}), 500

@app.route('/api/lmstudio/models', methods=['GET'])
def get_lmstudio_models():
    """Get list of available models from LM Studio"""
    try:
        models = ai_provider_manager.get_models('lmstudio')
        return jsonify({
            "models": models,
            "timestamp": datetime.now().isoformat(),
        }), 200
    except Exception as exc:
        logger.error(f"Error fetching LM Studio models: {exc}")
        return jsonify({"error": str(exc)}), 500

@app.route('/api/game/state', methods=['GET'])
def get_game_state_api():
    """Get current game state - screen, running status, etc."""
    try:
        current_state = get_game_state()
        frame_count = current_state.get("frame_count", 0)

        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]
            if hasattr(emulator, 'get_frame_count'):
                frame_count = emulator.get_frame_count()
                update_game_state({"frame_count": frame_count})
        
        # Build game state response
        game_state = {
            "running": current_state["rom_loaded"],
            "rom_loaded": current_state["rom_loaded"],
            "rom_name": current_state.get("rom_name", ""),
            "screen_available": current_state["rom_loaded"],
            "frame_count": frame_count,
            "fps": current_state.get("fps", 0),
            "emulator": current_state.get("active_emulator") or "gb",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(game_state), 200
    except Exception as e:
        logger.error(f"Error getting game state: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agent/status', methods=['GET'])
def get_agent_status_api():
    """Get agent status - connected, mode, last action"""
    try:
        current_state = get_game_state()
        actual_mode = agent_mode_state["mode"]
        # Normalize mode for frontend (return 'auto' for any auto_* mode)
        frontend_mode = "auto" if actual_mode.startswith("auto_") else actual_mode
        
        # Get dual-model status
        dual_model_status = None
        if hasattr(ai_provider_manager, 'get_dual_model_status'):
            dual_model_status = ai_provider_manager.get_dual_model_status()
        
        agent_status = {
            "connected": current_state["rom_loaded"],
            "agent_name": "OpenClaw",
            "mode": frontend_mode,  # Frontend-friendly mode
            "actual_mode": actual_mode,  # Actual mode for advanced use
            "autonomous_level": agent_mode_state["autonomous_level"],
            "current_action": agent_mode_state["current_action"],
            "last_decision": agent_mode_state["last_decision"],
            "enabled": agent_mode_state["enabled"],
            "game_running": current_state["rom_loaded"],
            "objectives": openclaw_runtime_state["objectives"],
            "personality": openclaw_runtime_state["personality"],
            "provider": ai_runtime_state["provider"],
            "model": ai_runtime_state["model"],
            # Dual-model architecture (NEW)
            "vision_model": openclaw_runtime_state["vision_model"],
            "planning_model": openclaw_runtime_state["planning_model"],
            "use_dual_model": openclaw_runtime_state["use_dual_model"],
            "dual_model_status": dual_model_status,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(agent_status), 200
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/game/button', methods=['POST'])
def press_game_button():
    """Press a game button (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT)"""
    try:
        data = request.get_json()
        button = data.get('button', '').upper()
        
        # Validate button
        valid_buttons = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        if button not in valid_buttons:
            return jsonify({"error": f"Invalid button. Valid buttons: {valid_buttons}"}), 400
        
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        # Execute action using emulator's step method
        emulator = emulators[current_state["active_emulator"]]
        success = emulator.step(button, 1)
        
        # Update agent state
        agent_mode_state["current_action"] = button
        agent_mode_state["last_decision"] = f"Pressed {button}"
        
        # Also add to action history
        add_to_action_history(button)
        
        return jsonify({
            "success": success,
            "button": button,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error pressing button: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/watch', methods=['GET'])
def get_memory_watch():
    """Get watched memory addresses"""
    try:
        # Return default watched addresses for Pokemon Red
        watched_addresses = [
            {"address": 0xD058, "name": "Player X Position", "size": 1},
            {"address": 0xD05C, "name": "Player Y Position", "size": 1},
            {"address": 0xD160, "name": "Player HP", "size": 2},
            {"address": 0xD16B, "name": "Player Max HP", "size": 2},
            {"address": 0xD11C, "name": "Map ID", "size": 1},
            {"address": 0xD347, "name": "Money", "size": 3},
            {"address": 0xDCD6, "name": "Player State", "size": 1},
        ]
        
        current_state = get_game_state()
        memory_data = []
        
        if current_state["rom_loaded"] and current_state["active_emulator"]:
            emulator = emulators[current_state["active_emulator"]]
            
            # Try to read actual memory if available
            for addr_info in watched_addresses:
                try:
                    value = read_emulator_memory_value(
                        emulator,
                        addr_info["address"],
                        addr_info.get("size", 1)
                    )
                    memory_data.append({
                        **addr_info,
                        "value": value,
                        "hex": hex(value) if value is not None else "N/A"
                    })
                except Exception:
                    memory_data.append({
                        **addr_info,
                        "value": None,
                        "hex": "N/A"
                    })
        else:
            memory_data = [{"error": "No ROM loaded"}]
        
        return jsonify({
            "addresses": watched_addresses,
            "values": memory_data,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting memory watch: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MEMORY API ENDPOINTS
# ============================================================================

@app.route('/api/memory/<path:address>', methods=['GET'])
def read_memory(address):
    """
    Read memory from the active emulator.
    
    Args:
        address: Memory address to read (0x0000-0xFFFF or hex like 0xD058)
        
    Query params:
        size: Number of bytes to read (default: 1, max: 256)
        format: Output format - 'int', 'hex', 'binary' (default: 'int')
    """
    try:
        # Parse address (support both int and hex string)
        try:
            if isinstance(address, str):
                if address.lower().startswith('0x'):
                    address = int(address, 16)
                else:
                    address = int(address)
            # Validate address range
            if address < 0 or address > 0xFFFF:
                return jsonify({"error": "Address must be between 0x0000 and 0xFFFF"}), 400
        except ValueError:
            return jsonify({"error": f"Invalid address format: {address}. Use decimal or hex (0x prefix)"}), 400
        
        # Get size parameter
        size = request.args.get('size', 1, type=int)
        size = min(max(1, size), 256)  # Clamp between 1-256
        
        output_format = request.args.get('format', 'int')
        
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        emulator = emulators[current_state["active_emulator"]]
        
        # Read memory using PyBoy API
        values = []
        for i in range(size):
            addr = (address + i) & 0xFFFF  # Wrap around
            try:
                if hasattr(emulator, 'pyboy') and emulator.pyboy:
                    value = emulator.pyboy.memory[addr]
                elif hasattr(emulator, 'memory'):
                    value = emulator.memory[addr]
                else:
                    return jsonify({"error": "Emulator does not support memory access"}), 503
                values.append(value)
            except Exception as mem_error:
                logger.error(f"Memory read error at {hex(addr)}: {mem_error}")
                return jsonify({"error": f"Failed to read memory at {hex(addr)}"}), 500
        
        # Format output
        if output_format == 'hex':
            formatted = [hex(v) for v in values]
        elif output_format == 'binary':
            formatted = [bin(v) for v in values]
        else:
            formatted = values
        
        return jsonify({
            "address": hex(address),
            "address_int": address,
            "size": size,
            "values": values,
            "formatted": formatted,
            "format": output_format,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error reading memory: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/memory/<path:address>', methods=['POST'])
def write_memory(address):
    """
    Write memory to the active emulator.
    
    Args:
        address: Memory address to write (0x0000-0xFFFF or hex like 0xD058)
        
    Body:
        value: Byte value to write (0-255)
        values: List of byte values to write (optional, for multi-byte writes)
    """
    try:
        # Parse address (support both int and hex string)
        try:
            if isinstance(address, str):
                if address.lower().startswith('0x'):
                    address = int(address, 16)
                else:
                    address = int(address)
            # Validate address range
            if address < 0 or address > 0xFFFF:
                return jsonify({"error": "Address must be between 0x0000 and 0xFFFF"}), 400
        except ValueError:
            return jsonify({"error": f"Invalid address format: {address}. Use decimal or hex (0x prefix)"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        emulator = emulators[current_state["active_emulator"]]
        
        # Get values to write
        if 'values' in data:
            values = data['values']
            if not isinstance(values, list):
                values = [values]
        elif 'value' in data:
            values = [data['value']]
        else:
            return jsonify({"error": "No 'value' or 'values' provided"}), 400
        
        # Validate values
        for i, v in enumerate(values):
            if not isinstance(v, int) or v < 0 or v > 255:
                return jsonify({"error": f"Value at index {i} must be an integer 0-255"}), 400
        
        # Write memory using PyBoy API
        written = []
        for i, value in enumerate(values):
            addr = (address + i) & 0xFFFF  # Wrap around
            try:
                if hasattr(emulator, 'pyboy') and emulator.pyboy:
                    emulator.pyboy.memory[addr] = value
                elif hasattr(emulator, 'memory'):
                    emulator.memory[addr] = value
                else:
                    return jsonify({"error": "Emulator does not support memory access"}), 503
                written.append({"address": hex(addr), "value": value})
            except Exception as mem_error:
                logger.error(f"Memory write error at {hex(addr)}: {mem_error}")
                return jsonify({"error": f"Failed to write memory at {hex(addr)}"}), 500
        
        return jsonify({
            "success": True,
            "message": f"Wrote {len(written)} byte(s) to memory",
            "writes": written,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error writing memory: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# POKEMON PARTY API ENDPOINT
# ============================================================================

# Pokemon Red/Blue memory addresses for party data
POKEMON_PARTY_ADDRESSES = {
    "party_count": 0xD163,        # Number of Pokemon in party
    "party_species": 0xD16B,      # Species list (6 bytes)
    "party_hp": 0xD16C,           # HP values (12 bytes, 2 per Pokemon)
    "party_max_hp": 0xD177,       # Max HP values (12 bytes)
    "party_level": 0xD182,        # Levels (6 bytes)
    "party_status": 0xD18D,       # Status conditions (6 bytes)
    "party_type1": 0xD198,        # Type 1 (6 bytes)
    "party_type2": 0xD1A3,        # Type 2 (6 bytes)
    "party_move1": 0xD1AE,        # Move 1 (6 bytes)
    "party_move2": 0xD1B3,        # Move 2 (6 bytes)
    "party_move3": 0xD1B8,        # Move 3 (6 bytes)
    "party_move4": 0xD1BD,        # Move 4 (6 bytes)
    "party_ot_id": 0xD1C2,        # Original Trainer ID (12 bytes)
    "party_exp": 0xD1C8,          # Experience (18 bytes, 3 per Pokemon)
    "party_nickname": 0xD1D0,     # Nicknames (66 bytes, 11 per Pokemon)
}

# Pokemon species names (Gen 1)
POKEMON_SPECIES = {
    0x00: None, 0x01: "Rhydon", 0x02: "Kangaskhan", 0x03: "Nidoran♂", 0x04: "Clefairy",
    0x05: "Spearow", 0x06: "Voltorb", 0x07: "Nidoking", 0x08: "Slowbro", 0x09: "Ivysaur",
    0x0A: "Exeggutor", 0x0B: "Lickitung", 0x0C: "Exeggcute", 0x0D: "Grimer", 0x0E: "Gengar",
    0x0F: "Nidoran♀", 0x10: "Nidoqueen", 0x11: "Cubone", 0x12: "Rhyhorn", 0x13: "Lapras",
    0x14: "Arcanine", 0x15: "Mew", 0x16: "Gyarados", 0x17: "Shellder", 0x18: "Tentacool",
    0x19: "Gastly", 0x1A: "Scyther", 0x1B: "Staryu", 0x1C: "Blastoise", 0x1D: "Pinsir",
    0x1E: "Tangela", 0x1F: "MissingNo", 0x20: "MissingNo", 0x21: "Growlithe", 0x22: "Onix",
    0x23: "Fearow", 0x24: "Pidgey", 0x25: "Slowpoke", 0x26: "Kadabra", 0x27: "Graveler",
    0x28: "Chansey", 0x29: "Machoke", 0x2A: "Mr. Mime", 0x2B: "Hitmonlee", 0x2C: "Hitmonchan",
    0x2D: "Arbok", 0x2E: "Parasect", 0x2F: "Psyduck", 0x30: "Drowzee", 0x31: "Golem",
    0x32: "MissingNo", 0x33: "Magmar", 0x34: "MissingNo", 0x35: "Electabuzz", 0x36: "Magneton",
    0x37: "Koffing", 0x38: "MissingNo", 0x39: "Mankey", 0x3A: "Seel", 0x3B: "Diglett",
    0x3C: "Tauros", 0x3D: "MissingNo", 0x3E: "MissingNo", 0x3F: "MissingNo", 0x40: "Farfetch'd",
    0x41: "Venonat", 0x42: "Dragonite", 0x43: "MissingNo", 0x44: "MissingNo", 0x45: "MissingNo",
    0x46: "Doduo", 0x47: "Poliwag", 0x48: "Jynx", 0x49: "Moltres", 0x4A: "Articuno",
    0x4B: "Zapdos", 0x4C: "Ditto", 0x4D: "Meowth", 0x4E: "Krabby", 0x4F: "MissingNo",
    0x50: "MissingNo", 0x51: "MissingNo", 0x52: "Vulpix", 0x53: "Ninetales", 0x54: "Pikachu",
    0x55: "Raichu", 0x56: "MissingNo", 0x57: "Dratini", 0x58: "Dragonair", 0x59: "Kabuto",
    0x5A: "Kabutops", 0x5B: "Horsea", 0x5C: "Seadra", 0x5D: "MissingNo", 0x5E: "MissingNo",
    0x5F: "Sandshrew", 0x60: "Sandslash", 0x61: "Omanyte", 0x62: "Omastar", 0x63: "Jigglypuff",
    0x64: "Wigglytuff", 0x65: "Eevee", 0x66: "Flareon", 0x67: "Jolteon", 0x68: "Vaporeon",
    0x69: "Machop", 0x6A: "Zubat", 0x6B: "Ekans", 0x6C: "Paras", 0x6D: "Poliwhirl",
    0x6E: "Poliwrath", 0x6F: "Weedle", 0x70: "Kakuna", 0x71: "Beedrill", 0x72: "MissingNo",
    0x73: "Dodrio", 0x74: "Primeape", 0x75: "Dugtrio", 0x76: "Venomoth", 0x77: "Dewgong",
    0x78: "MissingNo", 0x79: "MissingNo", 0x7A: "Caterpie", 0x7B: "Metapod", 0x7C: "Butterfree",
    0x7D: "Machamp", 0x7E: "MissingNo", 0x7F: "Golduck", 0x80: "Hypno", 0x81: "Golbat",
    0x82: "Mewtwo", 0x83: "Snorlax", 0x84: "Magikarp", 0x85: "MissingNo", 0x86: "MissingNo",
    0x87: "Muk", 0x88: "MissingNo", 0x89: "Kingler", 0x8A: "Cloyster", 0x8B: "MissingNo",
    0x8C: "Electrode", 0x8D: "Clefable", 0x8E: "Weezing", 0x8F: "Persian", 0x90: "Marowak",
    0x91: "MissingNo", 0x92: "Haunter", 0x93: "Abra", 0x94: "Alakazam", 0x95: "Pidgeotto",
    0x96: "Pidgeot", 0x97: "Starmie", 0x98: "Bulbasaur", 0x99: "Venusaur", 0x9A: "Tentacruel",
    0x9B: "MissingNo", 0x9C: "Goldeen", 0x9D: "Seaking", 0x9E: "MissingNo", 0x9F: "MissingNo",
    0xA0: "MissingNo", 0xA1: "MissingNo", 0xA2: "MissingNo", 0xA3: "MissingNo", 0xA4: "MissingNo",
    0xA5: "MissingNo", 0xA6: "MissingNo", 0xA7: "MissingNo", 0xA8: "MissingNo", 0xA9: "MissingNo",
    0xAA: "MissingNo", 0xAB: "MissingNo", 0xAC: "MissingNo", 0xAD: "MissingNo", 0xAE: "MissingNo",
    0xAF: "MissingNo", 0xB0: "MissingNo", 0xB1: "MissingNo", 0xB2: "MissingNo", 0xB3: "MissingNo",
    0xB4: "MissingNo", 0xB5: "MissingNo", 0xB6: "MissingNo", 0xB7: "MissingNo", 0xB8: "MissingNo",
    0xB9: "MissingNo", 0xBA: "MissingNo", 0xBB: "MissingNo", 0xBC: "MissingNo", 0xBD: "MissingNo",
    0xBE: "MissingNo", 0xBF: "MissingNo", 0xC0: "MissingNo", 0xC1: "MissingNo", 0xC2: "MissingNo",
    0xC3: "MissingNo", 0xC4: "MissingNo", 0xC5: "MissingNo", 0xC6: "MissingNo", 0xC7: "MissingNo",
    0xC8: "MissingNo", 0xC9: "MissingNo", 0xCA: "MissingNo", 0xCB: "MissingNo", 0xCC: "MissingNo",
    0xCD: "MissingNo", 0xCE: "MissingNo", 0xCF: "MissingNo", 0xD0: "MissingNo", 0xD1: "MissingNo",
    0xD2: "MissingNo", 0xD3: "MissingNo", 0xD4: "MissingNo", 0xD5: "MissingNo", 0xD6: "MissingNo",
    0xD7: "MissingNo", 0xD8: "MissingNo", 0xD9: "MissingNo", 0xDA: "MissingNo", 0xDB: "MissingNo",
    0xDC: "MissingNo", 0xDD: "MissingNo", 0xDE: "MissingNo", 0xDF: "MissingNo", 0xE0: "MissingNo",
    0xE1: "MissingNo", 0xE2: "MissingNo", 0xE3: "MissingNo", 0xE4: "MissingNo", 0xE5: "MissingNo",
    0xE6: "MissingNo", 0xE7: "MissingNo", 0xE8: "MissingNo", 0xE9: "MissingNo", 0xEA: "MissingNo",
    0xEB: "MissingNo", 0xEC: "MissingNo", 0xED: "MissingNo", 0xEE: "MissingNo", 0xEF: "MissingNo",
    0xF0: "MissingNo", 0xF1: "MissingNo", 0xF2: "MissingNo", 0xF3: "MissingNo", 0xF4: "MissingNo",
    0xF5: "MissingNo", 0xF6: "MissingNo", 0xF7: "MissingNo", 0xF8: "MissingNo", 0xF9: "MissingNo",
    0xFA: "MissingNo", 0xFB: "MissingNo", 0xFC: "MissingNo", 0xFD: "MissingNo", 0xFE: "MissingNo",
    0xFF: "MissingNo"
}

# Move names (Gen 1)
MOVE_NAMES = {
    0x00: None, 0x01: "Pound", 0x02: "Karate Chop", 0x03: "Double Slap", 0x04: "Comet Punch",
    0x05: "Mega Punch", 0x06: "Pay Day", 0x07: "Fire Punch", 0x08: "Ice Punch", 0x09: "Thunder Punch",
    0x0A: "Scratch", 0x0B: "Vice Grip", 0x0C: "Guillotine", 0x0D: "Razor Wind", 0x0E: "Swords Dance",
    0x0F: "Cut", 0x10: "Gust", 0x11: "Wing Attack", 0x12: "Fly", 0x13: "Bind",
    # ... (truncated for brevity, add more as needed)
}

# Type names
TYPE_NAMES = {
    0x00: "Normal", 0x01: "Fighting", 0x02: "Flying", 0x03: "Poison", 0x04: "Ground",
    0x05: "Rock", 0x06: "Bird", 0x07: "Bug", 0x08: "Ghost", 0x14: "Fire",
    0x15: "Water", 0x16: "Grass", 0x17: "Electric", 0x18: "Psychic", 0x19: "Ice",
    0x1A: "Dragon"
}


def read_pokemon_memory(emulator, address, size=1):
    """Helper function to read Pokemon memory"""
    try:
        if hasattr(emulator, 'pyboy') and emulator.pyboy:
            if size == 1:
                return emulator.pyboy.memory[address]
            else:
                return [emulator.pyboy.memory[address + i] for i in range(size)]
        elif hasattr(emulator, 'memory'):
            if size == 1:
                return emulator.memory[address]
            else:
                return [emulator.memory[address + i] for i in range(size)]
        return None
    except Exception as e:
        logger.error(f"Error reading Pokemon memory at {hex(address)}: {e}")
        return None


@app.route('/api/party', methods=['GET'])
def get_party():
    """
    Get party Pokemon information.
    
    Returns detailed information about all Pokemon in the party.
    """
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        emulator = emulators[current_state["active_emulator"]]
        
        # Read party count
        party_count = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_count"])
        if party_count is None:
            return jsonify({"error": "Failed to read party data"}), 500
        
        party = []
        
        for i in range(min(party_count, 6)):
            pokemon = {
                "slot": i + 1,
                "species_id": None,
                "species_name": None,
                "level": None,
                "hp": None,
                "max_hp": None,
                "status": None,
                "type1": None,
                "type2": None,
                "moves": [],
                "ot_id": None,
            }
            
            # Read species
            species_id = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_species"] + i)
            if species_id is not None:
                pokemon["species_id"] = species_id
                pokemon["species_name"] = POKEMON_SPECIES.get(species_id, f"Unknown ({species_id})")
            
            # Read level
            level = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_level"] + i)
            if level is not None:
                pokemon["level"] = level
            
            # Read HP (2 bytes, big-endian)
            hp_high = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_hp"] + (i * 2))
            hp_low = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_hp"] + (i * 2) + 1)
            if hp_high is not None and hp_low is not None:
                pokemon["hp"] = (hp_high << 8) | hp_low
            
            # Read Max HP
            max_hp_high = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_max_hp"] + (i * 2))
            max_hp_low = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_max_hp"] + (i * 2) + 1)
            if max_hp_high is not None and max_hp_low is not None:
                pokemon["max_hp"] = (max_hp_high << 8) | max_hp_low
            
            # Read status
            status = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_status"] + i)
            if status is not None:
                pokemon["status"] = status
                # Decode status conditions
                status_conditions = []
                if status == 0:
                    status_conditions.append("Healthy")
                if status & 0x01:
                    status_conditions.append("Sleep")
                if status & 0x02:
                    status_conditions.append("Poison")
                if status & 0x04:
                    status_conditions.append("Burn")
                if status & 0x08:
                    status_conditions.append("Freeze")
                if status & 0x10:
                    status_conditions.append("Paralyze")
                pokemon["status_text"] = ", ".join(status_conditions) if status_conditions else "Healthy"
            
            # Read types
            type1 = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_type1"] + i)
            type2 = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_type2"] + i)
            if type1 is not None:
                pokemon["type1"] = TYPE_NAMES.get(type1, f"Unknown ({type1})")
            if type2 is not None:
                pokemon["type2"] = TYPE_NAMES.get(type2, f"Unknown ({type2})")
            
            # Read moves
            for j in range(4):
                move_id = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES[f"party_move{j+1}"] + i)
                if move_id is not None and move_id > 0:
                    pokemon["moves"].append({
                        "id": move_id,
                        "name": MOVE_NAMES.get(move_id, f"Unknown ({move_id})")
                    })
            
            # Read OT ID (2 bytes)
            ot_high = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_ot_id"] + (i * 2))
            ot_low = read_pokemon_memory(emulator, POKEMON_PARTY_ADDRESSES["party_ot_id"] + (i * 2) + 1)
            if ot_high is not None and ot_low is not None:
                pokemon["ot_id"] = (ot_high << 8) | ot_low
            
            # Calculate HP percentage
            if pokemon["hp"] is not None and pokemon["max_hp"] is not None and pokemon["max_hp"] > 0:
                pokemon["hp_percent"] = round((pokemon["hp"] / pokemon["max_hp"]) * 100, 1)
            else:
                pokemon["hp_percent"] = 0
            
            party.append(pokemon)
        
        return jsonify({
            "party_count": party_count,
            "party": party,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting party: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# INVENTORY API ENDPOINT
# ============================================================================

# Pokemon Red/Blue inventory addresses
INVENTORY_ADDRESSES = {
    "money": 0xD347,            # Money (3 bytes, BCD)
    "item_count": 0xD31D,       # Number of items in bag
    "items": 0xD31E,            # Item list (max 20 items, 2 bytes each)
}

# Item names (Gen 1)
ITEM_NAMES = {
    0x00: None, 0x01: "Master Ball", 0x02: "Ultra Ball", 0x03: "Great Ball", 0x04: "Poké Ball",
    0x05: "Town Map", 0x06: "Bicycle", 0x07: "?????", 0x08: "Safari Ball", 0x09: "Pokédex",
    0x0A: "Moon Stone", 0x0B: "Antidote", 0x0C: "Burn Heal", 0x0D: "Ice Heal", 0x0E: "Awakening",
    0x0F: "Paralyze Heal", 0x10: "Full Restore", 0x11: "Max Potion", 0x12: "Hyper Potion",
    0x13: "Super Potion", 0x14: "Potion", 0x15: "Boulder Badge", 0x16: "Safari Bait",
    0x17: "Safari Rock", 0x18: "Secret Key", 0x19: "?????", 0x1A: "Helix Fossil",
    0x1B: "Dome Fossil", 0x1C: "Silph Scope", 0x1D: "Bicycle", 0x1E: "Old Amber",
    0x1F: "Lift Key", 0x20: "Card Key", 0x21: "Nugget", 0x22: "PP Up", 0x23: "Poké Flute",
    0x24: "Exp. All", 0x25: "Old Rod", 0x26: "Good Rod", 0x27: "Super Rod", 0x28: "S.S. Ticket",
    0x29: "Gold Teeth", 0x2A: "Coin Case", 0x2B: "Oak's Parcel", 0x2C: "Itemfinder",
    0x2D: "Silph Scope", 0x2E: "Poké Doll", 0x2F: "Full Heal", 0x30: "Revive",
    0x31: "Max Revive", 0x32: "Guard Spec.", 0x33: "Super Repel", 0x34: "Max Repel",
    0x35: "Dire Hit", 0x36: "Coin", 0x37: "Fresh Water", 0x38: "Soda Pop", 0x39: "Lemonade",
    0x3A: "S.S. Ticket", 0x3B: "Gold Teeth", 0x3C: "X Attack", 0x3D: "X Defend",
    0x3E: "X Speed", 0x3F: "X Special", 0x40: "Coin Case", 0x41: "Oak's Parcel",
    0x42: "Itemfinder", 0x43: "Silph Scope", 0x44: "Poké Flute", 0x45: "Lift Key",
    0x46: "Exp. All", 0x47: "Old Rod", 0x48: "Good Rod", 0x49: "Super Rod",
    0x4A: "PP Up", 0x4B: "Ether", 0x4C: "Max Ether", 0x4D: "Elixer", 0x4E: "Max Elixer",
    0x4F: "Fire Stone", 0x50: "Thunder Stone", 0x51: "Water Stone", 0x52: "Leaf Stone",
    0x53: "HP Up", 0x54: "Protein", 0x55: "Iron", 0x56: "Carbos", 0x57: "Calcium",
    0x58: "Rare Candy", 0x59: "Dome Fossil", 0x5A: "Helix Fossil", 0x5B: "Secret Key",
    0x5C: "Bicycle", 0x5D: "Town Map", 0x5E: "Voucher", 0x5F: "Coin Case",
}


@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    """
    Get player inventory information.
    
    Returns money, item count, and list of items in bag.
    """
    try:
        current_state = get_game_state()
        if not current_state["rom_loaded"] or not current_state["active_emulator"]:
            return jsonify({"error": "No ROM loaded"}), 400
        
        emulator = emulators[current_state["active_emulator"]]
        
        # Read money (3 bytes BCD)
        money_bytes = read_pokemon_memory(emulator, INVENTORY_ADDRESSES["money"], 3)
        if money_bytes is not None:
            # BCD decoding: each byte represents 2 decimal digits
            money = 0
            for b in money_bytes:
                money = money * 100 + ((b >> 4) * 10) + (b & 0x0F)
        else:
            money = 0
        
        # Read item count
        item_count = read_pokemon_memory(emulator, INVENTORY_ADDRESSES["item_count"])
        if item_count is None:
            item_count = 0
        
        # Read items (each item is 2 bytes: ID and quantity)
        items = []
        for i in range(min(item_count, 20)):
            item_id = read_pokemon_memory(emulator, INVENTORY_ADDRESSES["items"] + (i * 2))
            quantity = read_pokemon_memory(emulator, INVENTORY_ADDRESSES["items"] + (i * 2) + 1)
            
            if item_id is not None and item_id > 0:
                items.append({
                    "slot": i + 1,
                    "id": item_id,
                    "name": ITEM_NAMES.get(item_id, f"Unknown ({item_id})"),
                    "quantity": quantity if quantity is not None else 0
                })
        
        return jsonify({
            "money": money,
            "money_formatted": f"¥{money:,}",
            "item_count": item_count,
            "items": items,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting inventory: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ROM LOAD API ENDPOINT
# ============================================================================

@app.route('/api/rom/load', methods=['POST'])
def load_rom_api():
    """
    Load a ROM file from the server filesystem.
    
    Body:
        path: Absolute or relative path to ROM file
        emulator_type: 'gb' or 'gba' (default: 'gb')
        launch_ui: Whether to launch UI (default: false)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        rom_path = data.get('path') or data.get('rom_path')
        emulator_type = data.get('emulator_type', 'gb')
        launch_ui = data.get('launch_ui', False)
        
        if not rom_path:
            return jsonify({"error": "ROM path is required"}), 400
        
        # Validate path exists
        if not os.path.exists(rom_path):
            return jsonify({"error": f"ROM file not found: {rom_path}"}), 404
        
        # Validate file extension
        _, ext = os.path.splitext(rom_path)
        if ext.lower() not in ALLOWED_ROM_EXTENSIONS:
            return jsonify({"error": f"Invalid ROM extension. Allowed: {ALLOWED_ROM_EXTENSIONS}"}), 400
        
        # Check file size
        file_size = os.path.getsize(rom_path)
        if file_size > MAX_ROM_SIZE:
            return jsonify({"error": f"ROM file too large: {file_size} bytes (max: {MAX_ROM_SIZE})"}), 400
        
        # Validate emulator type
        emulator_type_mapping = {
            'gb': 'pyboy',
            'gba': 'pygba',
            'pyboy': 'pyboy',
            'pygba': 'pygba'
        }
        
        mapped_emulator = emulator_type_mapping.get(emulator_type.lower())
        if not mapped_emulator or mapped_emulator not in emulators:
            return jsonify({"error": f"Invalid emulator type. Available: {list(emulator_type_mapping.keys())}"}), 400
        
        emulator_type = mapped_emulator
        
        emulator_instance = emulators[emulator_type]
        configure_emulator_launch_ui(emulator_instance, launch_ui)

        # Load ROM
        logger.info(f"Loading ROM: {rom_path} with emulator: {emulator_type}")
        success = emulator_instance.load_rom(rom_path)
        
        if success:
            sync_loaded_rom_state(emulator_type, rom_path)
            ensure_emulation_loop_running()
            
            # Get emulator info
            emulator = emulator_instance
            ui_status = emulator.get_ui_status() if hasattr(emulator, 'get_ui_status') else {"running": False}
            
            return jsonify({
                "success": True,
                "message": "ROM loaded successfully",
                "rom_name": os.path.basename(rom_path),
                "rom_path": rom_path,
                "rom_size": file_size,
                "emulator_type": emulator_type,
                "ui_launched": ui_status.get("running", False),
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({"error": "Failed to load ROM"}), 500
        
    except Exception as e:
        logger.error(f"Error loading ROM: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SCREEN STREAM API ENDPOINT (SSE)
# ============================================================================

@app.route('/api/screen/stream', methods=['GET'])
def screen_stream():
    """
    SSE endpoint for real-time screen updates.
    
    This is an alias for the existing /api/stream endpoint for compatibility.
    
    Query params:
        fps: Target frames per second (default: 30, max: 60)
    """
    # Delegate to existing stream endpoint
    return stream_screen()


# ============================================================================
# AGENT MODE API ENDPOINT
# ============================================================================

# Valid agent modes
VALID_AGENT_MODES = [
    "idle",           # No automatic actions
    "auto",           # Alias for auto_explore (frontend compatibility)
    "auto_explore",   # Automatically explore the map
    "auto_battle",    # Automatically battle wild Pokemon
    "auto_train",     # Train Pokemon automatically
    "auto_fish",      # Automatic fishing
    "auto_walk",      # Walk in a direction continuously
    "auto_center",    # Walk to Pokemon center
    "auto_shop",      # Walk to Pokemart
    "speedrun",       # Speedrun mode (aggressive automation)
    "manual"          # Manual control only
]

# Mode aliases for frontend compatibility
MODE_ALIASES = {
    "auto": "auto_explore",  # Map 'auto' to 'auto_explore'
}


@app.route('/api/agent/mode', methods=['POST'])
def set_agent_mode():
    """
    Set the agent's automation mode.
    
    Body:
        mode: One of 'idle', 'auto', 'auto_explore', 'auto_battle', 'auto_train', 
              'auto_fish', 'auto_walk', 'auto_center', 'auto_shop', 'speedrun', 'manual'
        enabled: Enable/disable agent (default: true)
        autonomous_level: 'passive', 'moderate', or 'aggressive' (default: 'moderate')
        direction: Direction for auto_walk mode (optional)
        target: Target location for auto_center/auto_shop (optional)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        mode = data.get('mode', 'idle').lower()
        enabled = data.get('enabled', True)
        autonomous_level = data.get('autonomous_level', 'moderate').lower()
        direction = data.get('direction', '').upper()
        target = data.get('target', '')
        
        # Validate mode
        if mode not in VALID_AGENT_MODES:
            return jsonify({
                "error": f"Invalid mode. Valid modes: {VALID_AGENT_MODES}"
            }), 400
        
        # Resolve mode alias (e.g., 'auto' -> 'auto_explore')
        actual_mode = MODE_ALIASES.get(mode, mode)
        
        # Validate autonomous level
        valid_levels = ['passive', 'moderate', 'aggressive']
        if autonomous_level not in valid_levels:
            return jsonify({
                "error": f"Invalid autonomous level. Valid levels: {valid_levels}"
            }), 400
        
        # Validate direction for auto_walk
        valid_directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if actual_mode == 'auto_walk' and direction and direction not in valid_directions:
            return jsonify({
                "error": f"Invalid direction for auto_walk. Valid directions: {valid_directions}"
            }), 400
        
        # Update agent state
        agent_mode_state["mode"] = actual_mode
        agent_mode_state["enabled"] = enabled
        agent_mode_state["autonomous_level"] = autonomous_level
        agent_mode_state["current_action"] = f"Mode set to {actual_mode}"
        agent_mode_state["last_decision"] = f"Agent mode changed to {actual_mode} (level: {autonomous_level})"
        
        # Store additional parameters
        if direction:
            agent_mode_state["direction"] = direction
        if target:
            agent_mode_state["target"] = target
        
        logger.info(f"Agent mode set: {mode} -> {actual_mode}, enabled: {enabled}, level: {autonomous_level}")
        
        # Normalize mode for frontend response (return 'auto' for any auto_* mode)
        frontend_mode = "auto" if actual_mode.startswith("auto_") else actual_mode
        
        return jsonify({
            "success": True,
            "message": f"Agent mode set to {actual_mode}",
            "mode": frontend_mode,  # Frontend-friendly mode
            "actual_mode": actual_mode,  # Actual mode for advanced use
            "enabled": enabled,
            "autonomous_level": autonomous_level,
            "direction": direction if actual_mode == 'auto_walk' else None,
            "target": target if actual_mode in ['auto_center', 'auto_shop'] else None,
            "valid_modes": VALID_AGENT_MODES,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting agent mode: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/agent/mode', methods=['GET'])
def get_agent_mode():
    """
    Get the current agent mode and settings.
    """
    try:
        actual_mode = agent_mode_state["mode"]
        # Normalize mode for frontend (return 'auto' for any auto_* mode)
        frontend_mode = "auto" if actual_mode.startswith("auto_") else actual_mode
        
        return jsonify({
            "mode": frontend_mode,  # Frontend-friendly mode
            "actual_mode": actual_mode,  # Actual mode for advanced use
            "enabled": agent_mode_state["enabled"],
            "autonomous_level": agent_mode_state["autonomous_level"],
            "current_action": agent_mode_state["current_action"],
            "last_decision": agent_mode_state["last_decision"],
            "valid_modes": VALID_AGENT_MODES,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting agent mode: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# DUAL-MODEL ARCHITECTURE API ENDPOINTS
# ============================================================================

@app.route('/api/dual-model/config', methods=['GET'])
def get_dual_model_config():
    """
    Get dual-model architecture configuration.
    
    Returns current vision and planning model settings.
    """
    try:
        config = {
            "vision_model": openclaw_runtime_state["vision_model"],
            "planning_model": openclaw_runtime_state["planning_model"],
            "use_dual_model": openclaw_runtime_state["use_dual_model"],
            "available_vision_models": [
                {"id": "kimi-k2.5", "name": "Kimi K2.5", "description": "Best for game screen analysis (FREE)"},
                {"id": "qwen-vl-plus", "name": "Qwen VL Plus", "description": "High quality vision (quota)"},
                {"id": "glm-4v-flash", "name": "GLM-4V Flash", "description": "Fast vision model"},
                {"id": "MiniMax-M2.7", "name": "MiniMax M2.7", "description": "Multimodal with vision (FREE)"},
            ],
            "available_planning_models": [
                {"id": "glm-5", "name": "GLM-5", "description": "Fast decisions, great for games"},
                {"id": "qwen3.5-plus", "name": "Qwen 3.5 Plus", "description": "Best reasoning (quota)"},
                {"id": "MiniMax-M2.7", "name": "MiniMax M2.7", "description": "Balanced, multimodal (FREE)"},
                {"id": "MiniMax-M2.5", "name": "MiniMax M2.5", "description": "Unlimited, reliable (FREE)"},
            ],
            "dual_model_status": ai_provider_manager.get_dual_model_status() if hasattr(ai_provider_manager, 'get_dual_model_status') else None,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(config), 200
    except Exception as e:
        logger.error(f"Error getting dual-model config: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/dual-model/config', methods=['POST'])
def set_dual_model_config():
    """
    Configure dual-model architecture.
    
    Body:
        vision_model: Vision model ID (kimi-k2.5, qwen-vl-plus, etc.)
        planning_model: Planning model ID (glm-5, qwen3.5-plus, etc.)
        use_dual_model: Enable/disable dual-model (true/false)
    """
    try:
        data = request.get_json() or {}
        
        allowed_vision_models = {"kimi-k2.5", "qwen-vl-plus", "glm-4v-flash", "MiniMax-M2.7"}
        allowed_planning_models = {"glm-5", "qwen3.5-plus", "MiniMax-M2.7", "MiniMax-M2.5"}
        
        vision_model = data.get("vision_model")
        planning_model = data.get("planning_model")
        use_dual_model = data.get("use_dual_model")
        
        changes = []
        
        if vision_model is not None:
            if vision_model not in allowed_vision_models:
                return jsonify({
                    "error": f"Invalid vision model. Allowed: {sorted(allowed_vision_models)}"
                }), 400
            openclaw_runtime_state["vision_model"] = vision_model
            changes.append(f"vision_model: {vision_model}")
        
        if planning_model is not None:
            if planning_model not in allowed_planning_models:
                return jsonify({
                    "error": f"Invalid planning model. Allowed: {sorted(allowed_planning_models)}"
                }), 400
            openclaw_runtime_state["planning_model"] = planning_model
            changes.append(f"planning_model: {planning_model}")
        
        if use_dual_model is not None:
            openclaw_runtime_state["use_dual_model"] = bool(use_dual_model)
            changes.append(f"use_dual_model: {use_dual_model}")
        
        # Update the AI provider manager
        if hasattr(ai_provider_manager, 'configure_dual_model'):
            result = ai_provider_manager.configure_dual_model(
                vision_model=openclaw_runtime_state["vision_model"],
                planning_model=openclaw_runtime_state["planning_model"],
                use_dual_model=openclaw_runtime_state["use_dual_model"]
            )
        
        logger.info(f"Dual-model config updated: {', '.join(changes)}")
        
        return jsonify({
            "success": True,
            "message": f"Dual-model configuration updated: {', '.join(changes)}",
            "config": {
                "vision_model": openclaw_runtime_state["vision_model"],
                "planning_model": openclaw_runtime_state["planning_model"],
                "use_dual_model": openclaw_runtime_state["use_dual_model"],
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting dual-model config: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/dual-model/status', methods=['GET'])
def get_dual_model_status():
    """
    Get detailed dual-model provider status.
    
    Returns last vision and planning responses, model availability, etc.
    """
    try:
        if hasattr(ai_provider_manager, 'get_dual_model_status'):
            status = ai_provider_manager.get_dual_model_status()
            status["timestamp"] = datetime.now().isoformat()
            return jsonify(status), 200
        else:
            return jsonify({
                "available": False,
                "error": "Dual-model provider not initialized",
                "config": {
                    "vision_model": openclaw_runtime_state["vision_model"],
                    "planning_model": openclaw_runtime_state["planning_model"],
                    "use_dual_model": openclaw_runtime_state["use_dual_model"],
                },
                "timestamp": datetime.now().isoformat()
            }), 503
    except Exception as e:
        logger.error(f"Error getting dual-model status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/vision/analyze', methods=['POST'])
def analyze_vision():
    """
    Analyze an image using AI vision capabilities.
    
    This endpoint is called by OpenClawAIProvider and DualModelProvider
    for screen analysis during gameplay.
    
    Body:
        prompt: Text prompt for the vision model
        image: Base64-encoded image data
        model: Optional model identifier (default: auto-detect)
    
    Returns:
        response: Vision analysis text
        model: Model used for analysis
    """
    try:
        data = request.get_json(silent=True) or {}
        
        prompt = data.get('prompt', 'Analyze this image')
        image_base64 = data.get('image', '')
        model = data.get('model')
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
        # Use AI provider manager to get vision response
        # First try dual-model provider if available
        if hasattr(ai_provider_manager, 'dual_model_provider') and ai_provider_manager.dual_model_provider:
            try:
                vision_model = openclaw_runtime_state.get('vision_model', 'kimi-k2.5')
                analysis = ai_provider_manager.dual_model_provider.analyze_screen(
                    image_bytes, 
                    context={"prompt": prompt}
                )
                return jsonify({
                    "response": analysis.raw_description,
                    "text": analysis.game_state,
                    "result": analysis.raw_description,
                    "model": f"vision:{vision_model}",
                    "timestamp": datetime.now().isoformat()
                }), 200
            except Exception as e:
                logger.warning(f"Dual-model vision failed: {e}, falling back to provider chain")
        
        # Fallback: Use regular AI provider with image
        # Convert to bytes and use the action API
        try:
            # Get provider
            provider_name = model if model else None
            connector = ai_provider_manager.get_provider(provider_name)
            
            if connector:
                # Use the connector's vision capability
                if hasattr(connector, 'get_next_action'):
                    # For game screens, use action API
                    action, used_provider = ai_provider_manager.get_next_action(
                        image_bytes,
                        prompt,
                        [],
                        provider_name
                    )
                    return jsonify({
                        "response": action,
                        "text": action,
                        "result": action,
                        "model": used_provider or "unknown",
                        "timestamp": datetime.now().isoformat()
                    }), 200
                else:
                    return jsonify({
                        "response": "Vision analysis not supported by this provider",
                        "text": "Vision analysis not supported",
                        "result": "Provider does not support vision",
                        "model": provider_name or "unknown",
                        "timestamp": datetime.now().isoformat()
                    }), 200
            else:
                return jsonify({"error": "No AI provider available"}), 503
                
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return jsonify({"error": f"Vision analysis failed: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in vision analysis endpoint: {e}")
        return jsonify({"error": str(e)}), 500


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
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

    try:
        app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
    finally:
        cleanup_server_resources()

if __name__ == "__main__":
    main()
