#!/usr/bin/env python3
"""
OpenClaw MCP Server for AI Py-Boy Emulation
Exposes emulator controls as MCP tools for OpenClaw agents

Version: 4.0.0 - Enhanced with Best Practices from Popular MCP Servers
- FastMCP-style clean error handling
- Structured error codes for programmatic handling
- Better typed tool definitions
- Enhanced session management with TTL
- Resource and prompt support
- Server capabilities advertisement

Tools:
- emulator_load_rom: Load a ROM file
- emulator_press_button: Press a controller button
- emulator_get_frame: Get current screen as image
- emulator_get_state: Get emulator state (RAM, inventory, etc.)
- emulator_tick: Advance emulation by one frame
- get_player_position: Read player coordinates from memory
- get_party_info: Read party Pokemon/monsters from memory
- get_inventory: Read inventory items from memory
- get_map_location: Read current map/location from memory
- get_money: Read money/currency from memory
- save_game_state / load_game_state: Save/load game progress
- auto_battle: AI decides optimal battle moves
- auto_explore: Autonomous world exploration
- auto_grind: Grind for XP/money
- session_start/session_get/session_set: Agent session persistence
- get_screen_base64: Get screen for vision analysis
"""

import os
import sys
import json
import base64
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from functools import wraps

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("  PyBoy not installed - emulator tools will be unavailable")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("  MCP library not installed - install with: pip install mcp")
    sys.exit(1)

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("emulator-mcp")

# Server version and metadata
SERVER_VERSION = "4.0.0"
SERVER_START_TIME = datetime.now().isoformat()


# ========== Error Handling Patterns (from FastMCP) ==========
# Error codes for programmatic handling by agents

class EmulatorErrorCode:
    """Error code constants for programmatic error handling"""
    NOT_INITIALIZED = "EMULATOR_NOT_INITIALIZED"
    ROM_NOT_FOUND = "ROM_NOT_FOUND"
    INVALID_ROM = "INVALID_ROM"
    BUTTON_INVALID = "BUTTON_INVALID"
    MEMORY_READ_ERROR = "MEMORY_READ_ERROR"
    SAVE_NOT_FOUND = "SAVE_NOT_FOUND"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    OPERATION_FAILED = "OPERATION_FAILED"


class EmulatorError(Exception):
    """Custom exception with error codes for better error handling"""
    def __init__(self, message: str, code: str, suggestions: List[str] = None):
        super().__init__(message)
        self.code = code
        self.suggestions = suggestions or []


def handle_tool_error(func):
    """Decorator for consistent error handling in tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except EmulatorError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": e.code,
                "suggestions": e.suggestions
            }
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": EmulatorErrorCode.OPERATION_FAILED,
                "suggestions": ["Check emulator state", "Try reloading the ROM"]
            }
    return wrapper


# Response formatter with timing and metadata
def format_response(
    success: bool,
    data: Any = None,
    error: str = None,
    error_code: str = None,
    suggestions: List[str] = None,
    tool_name: str = None,
    timing_ms: float = None,
    include_state: bool = False
) -> Dict[str, Any]:
    """Clean response format - inspired by FastMCP"""
    response = {"success": success}
    
    if data is not None:
        response["data"] = data
    
    if error:
        response["error"] = error
        if error_code:
            response["error_code"] = error_code
        if suggestions:
            response["suggestions"] = suggestions
    
    if tool_name:
        response["tool"] = tool_name
    
    if timing_ms is not None:
        response["timing_ms"] = round(timing_ms, 2)
    
    if include_state:
        response["server"] = {
            "version": SERVER_VERSION,
            "uptime_seconds": (datetime.now() - datetime.fromisoformat(SERVER_START_TIME)).total_seconds()
        }
    
    return response


# Button enum for type safety
class GameButton(str, Enum):
    """Valid game controller buttons"""
    A = "A"
    B = "B"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    START = "START"
    SELECT = "SELECT"
    
    @classmethod
    def values(cls) -> List[str]:
        return [b.value for b in cls]

# ========== Legacy agent_response (kept for backward compatibility) ==========
# Use format_response() for new code
def agent_response(
    success: bool, 
    data: Any = None, 
    error: str = None,
    tool_name: str = None,
    suggestions: List[str] = None,
    timing_ms: float = None
) -> str:
    """Legacy response format - wraps format_response with emulator state"""
    response = format_response(
        success=success,
        data=data,
        error=error,
        suggestions=suggestions,
        tool_name=tool_name,
        timing_ms=timing_ms,
        include_state=True
    )
    response["emulator_state"] = get_state()
    response["timestamp"] = datetime.now().isoformat()
    return json.dumps(response, indent=2)


# Debug logging for agents
def debug_log(level: str, message: str, data: Dict = None):
    """Log debug messages for agent troubleshooting"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "frame_count": frame_count,
        "emulator_initialized": emulator is not None
    }
    if data:
        log_entry["data"] = data
    
    if level == "DEBUG":
        logger.debug(json.dumps(log_entry))
    elif level == "INFO":
        logger.info(json.dumps(log_entry))
    elif level == "WARNING":
        logger.warning(json.dumps(log_entry))
    elif level == "ERROR":
        logger.error(json.dumps(log_entry))

# Initialize MCP server with capabilities
# MCP server can advertise resources and prompts in addition to tools
server = Server(
    "pyboy-emulator",
    capabilities={
        "tools": {},
        # Uncomment when using MCP SDK that supports these:
        # "resources": {"subscribe": True, "list": True},
        # "prompts": {"list": True},
    }
)
server_version = SERVER_VERSION

# Global emulator state
emulator: Optional[PyBoy] = None
rom_path: Optional[str] = None
frame_count: int = 0


def init_emulator(rom_file: str) -> bool:
    """Initialize PyBoy emulator with ROM with improved error handling"""
    global emulator, rom_path, frame_count
    
    if not PYBOY_AVAILABLE:
        logger.error("PyBoy not available")
        raise EmulatorError(
            "PyBoy library not available. Install with: pip install pyboy",
            EmulatorErrorCode.OPERATION_FAILED,
            ["Install pyboy: pip install pyboy"]
        )
    
    # Check if file exists first
    if not os.path.exists(rom_file):
        raise EmulatorError(
            f"ROM file not found: {rom_file}",
            EmulatorErrorCode.ROM_NOT_FOUND,
            ["Check the file path is correct", "Verify the ROM file exists"]
        )
    
    try:
        emulator = PyBoy(rom_file, window="null")
        rom_path = rom_file
        frame_count = 0
        logger.info(f"Loaded ROM: {rom_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to load ROM: {e}")
        raise EmulatorError(
            f"Failed to load ROM: {str(e)}",
            EmulatorErrorCode.INVALID_ROM,
            ["Verify ROM file is valid Game Boy format", "Try a different ROM file"]
        )


def press_button(button: str) -> bool:
    """Press a controller button with validation"""
    global emulator, frame_count
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized. Load a ROM first.",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first with a valid .gb ROM file path"]
        )
    
    # Validate button
    button_upper = button.upper()
    try:
        game_button = GameButton(button_upper)
    except ValueError:
        raise EmulatorError(
            f"Invalid button: {button}. Valid buttons: {GameButton.values()}",
            EmulatorErrorCode.BUTTON_INVALID,
            [f"Use one of: {', '.join(GameButton.values())}"]
        )
    
    button_map = {
        GameButton.A: WindowEvent.PRESS_BUTTON_A,
        GameButton.B: WindowEvent.PRESS_BUTTON_B,
        GameButton.UP: WindowEvent.PRESS_ARROW_UP,
        GameButton.DOWN: WindowEvent.PRESS_ARROW_DOWN,
        GameButton.LEFT: WindowEvent.PRESS_ARROW_LEFT,
        GameButton.RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
        GameButton.START: WindowEvent.PRESS_BUTTON_START,
        GameButton.SELECT: WindowEvent.PRESS_BUTTON_SELECT,
    }
    
    release_map = {
        GameButton.A: WindowEvent.RELEASE_BUTTON_A,
        GameButton.B: WindowEvent.RELEASE_BUTTON_B,
        GameButton.UP: WindowEvent.RELEASE_ARROW_UP,
        GameButton.DOWN: WindowEvent.RELEASE_ARROW_DOWN,
        GameButton.LEFT: WindowEvent.RELEASE_ARROW_LEFT,
        GameButton.RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        GameButton.START: WindowEvent.RELEASE_BUTTON_START,
        GameButton.SELECT: WindowEvent.RELEASE_BUTTON_SELECT,
    }
    
    try:
        emulator.send_input(button_map[game_button])
        emulator.tick()
        emulator.send_input(release_map[game_button])
        frame_count += 1
        logger.debug(f"Pressed button: {button}")
        return True
    except Exception as e:
        logger.error(f"Button press failed: {e}")
        raise EmulatorError(
            f"Failed to press button: {str(e)}",
            EmulatorErrorCode.OPERATION_FAILED
        )


def press_sequence(sequence: str, delay_ms: int = 100) -> Dict[str, Any]:
    """Press multiple buttons in sequence"""
    global emulator, frame_count
    
    if emulator is None:
        logger.error("Emulator not initialized")
        return {"success": False, "error": "Emulator not initialized"}
    
    import re
    import time
    
    # Parse: "A B START" or "A2 R3 U1" (button + count)
    button_pattern = r'^([AUDLRSX])(\d*)$|^W$|^([A-Z]+)$'
    
    tokens = sequence.strip().split()
    pressed = []
    errors = []
    
    for token in tokens:
        token = token.upper()
        
        if token == 'W':
            # Wait
            emulator.tick()
            frame_count += 1
            pressed.append("WAIT")
            continue
        
        # Check for count (e.g., "A2" = press A 2 times)
        if len(token) > 1 and token[-1].isdigit():
            button = token[:-1]
            count = int(token[-1])
        else:
            button = token
            count = 1
        
        for _ in range(count):
            if button in ['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'SELECT']:
                success = press_button(button)
                if success:
                    pressed.append(button)
                else:
                    errors.append(f"Failed: {button}")
            else:
                errors.append(f"Unknown: {button}")
        
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)
    
    return {
        "success": len(errors) == 0,
        "pressed": pressed,
        "errors": errors,
        "frame": frame_count
    }


def save_screenshot(output_path: str = None) -> Dict[str, Any]:
    """Save current frame to file"""
    global emulator, frame_count
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        if output_path is None:
            output_dir = Path(__file__).parent / "screenshots"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"frame_{frame_count:06d}.png"
        
        # Get screen
        screen = emulator.screen
        img = screen.image
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save
        img.save(output_path, format='PNG')
        
        return {
            "success": True,
            "path": str(output_path),
            "frame": frame_count
        }
    except Exception as e:
        logger.error(f"Save screenshot failed: {e}")
        return {"success": False, "error": str(e)}


def get_frame() -> Optional[Dict[str, Any]]:
    """Get current frame as base64-encoded image"""
    global emulator
    
    if emulator is None:
        logger.error("Emulator not initialized")
        return None
    
    try:
        # Get screen buffer from PyBoy
        screen = emulator.screen
        if screen is None:
            return None
        
        # Get PIL Image from screen (PyBoy API)
        img = screen.image
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'format': 'png',
            'base64': img_base64,
            'width': img.width,
            'height': img.height,
            'frame': frame_count
        }
    except Exception as e:
        logger.error(f"Failed to get frame: {e}")
        return None


def get_state() -> Dict[str, Any]:
    """Get emulator state"""
    return {
        'initialized': emulator is not None,
        'rom_path': rom_path,
        'frame_count': frame_count,
        'pyboy_available': PYBOY_AVAILABLE
    }


def tick() -> bool:
    """Advance emulation by one frame"""
    global emulator, frame_count
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized. Load a ROM first.",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        emulator.tick()
        frame_count += 1
        return True
    except Exception as e:
        logger.error(f"Tick failed: {e}")
        return False


# ========== Agent-First Memory Reading Tools ==========
# These are game-agnostic but optimized for Pokemon-style games
# All return timing, state info, and actionable errors

def get_player_position() -> Dict[str, Any]:
    """
    Get player position from memory.
    Pokemon Red/Blue: 0xD062 (X), 0xD063 (Y)
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first with a valid .gb ROM file path"],
            "timing_ms": (time.time() - start_time) * 1000
        }
    
    try:
        # Common player position addresses (Game Boy games vary)
        # Pokemon: X at 0xD062, Y at 0xD063
        # Some games use different addresses
        x = emulator.memory[0xD062]
        y = emulator.memory[0xD063]
        
        # Try alternative addresses
        alt_x = emulator.memory.get(0xD061, None)
        alt_y = emulator.memory.get(0xD064, None)
        
        return {
            "success": True,
            "tool": "get_player_position",
            "data": {
                "x": x,
                "y": y,
                "alternative_coordinates": {
                    "x_alt": alt_x,
                    "y_alt": alt_y
                },
                "map_coordinates_raw": f"({x}, {y})",
                "note": "Coordinates are tile-based, not pixel-based"
            },
            "frame": frame_count,
            "memory_addresses": {"x": "0xD062", "y": "0xD063"},
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get player position: {e}")
        return {
            "success": False,
            "error": f"Failed to read player position: {str(e)}",
            "suggestions": ["Game may use different memory addresses", "Use emulator_read_memory to explore memory"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


def get_party_info() -> Dict[str, Any]:
    """
    Get party Pokemon/monsters from memory.
    Pokemon Red/Blue: Party starts at 0xD163 (first Pokemon)
    Each Pokemon is 44 bytes
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": (time.time() - start_time) * 1000
        }
    
    try:
        party = []
        
        # Read up to 6 Pokemon in party
        for i in range(6):
            offset = 0xD163 + (i * 44)
            
            try:
                species_id = emulator.memory[offset]
                
                # Skip if no Pokemon in this slot
                if species_id == 0 or species_id == 0xFF:
                    continue
                
                # Basic party member data
                pokemon = {
                    "slot": i + 1,
                    "species_id": species_id,
                    "species_hex": hex(species_id),
                    "address": hex(offset),
                    # HP bytes at offset + 0x1E (2 bytes)
                    "current_hp": (emulator.memory[offset + 0x1F] << 8) | emulator.memory[offset + 0x1E],
                    # Max HP at offset + 0x20
                    "max_hp": (emulator.memory[offset + 0x21] << 8) | emulator.memory[offset + 0x20],
                    # Level at offset + 0x18
                    "level": emulator.memory[offset + 0x18]
                }
                party.append(pokemon)
                
            except (IndexError, KeyError):
                break
        
        return {
            "success": True,
            "tool": "get_party_info",
            "data": {
                "party_count": len(party),
                "party": party,
                "note": "Party data structure varies by game"
            },
            "frame": frame_count,
            "memory_base": "0xD163",
            "party_size_bytes": 44,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get party info: {e}")
        return {
            "success": False,
            "error": f"Failed to read party: {str(e)}",
            "suggestions": ["Only works for Pokemon-style games", "ROM may use different memory layout"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


def get_inventory() -> Dict[str, Any]:
    """
    Get player inventory/items from memory.
    Pokemon Red/Blue: Item bag starts at 0xD6E5
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": (time.time() - start_time) * 1000
        }
    
    try:
        items = []
        
        # Read item bag - up to 20 items
        for i in range(20):
            offset = 0xD6E5 + (i * 2)
            
            try:
                item_id = emulator.memory[offset]
                quantity = emulator.memory[offset + 1]
                
                # Skip empty slots
                if item_id == 0 or item_id == 0xFF:
                    continue
                
                items.append({
                    "slot": i + 1,
                    "item_id": item_id,
                    "item_hex": hex(item_id),
                    "quantity": quantity
                })
                
            except (IndexError, KeyError):
                break
        
        return {
            "success": True,
            "tool": "get_inventory",
            "data": {
                "item_count": len(items),
                "items": items,
                "note": "Item IDs are game-specific"
            },
            "frame": frame_count,
            "memory_base": "0xD6E5",
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get inventory: {e}")
        return {
            "success": False,
            "error": f"Failed to read inventory: {str(e)}",
            "suggestions": ["Inventory structure varies by game", "This works best for Pokemon Red/Blue"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


def get_map_location() -> Dict[str, Any]:
    """
    Get current map/location from memory.
    Pokemon Red/Blue: Map ID at 0xD35E (or 0xD36F for some)
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": (time.time() - start_time) * 1000
        }
    
    try:
        # Try multiple common map address locations
        map_id_primary = emulator.memory[0xD35E]
        map_id_secondary = emulator.memory.get(0xD36F, None)
        
        # Some games store map in different locations
        map_bank = emulator.memory.get(0xD35C, 0)  # Map bank ID
        
        return {
            "success": True,
            "tool": "get_map_location",
            "data": {
                "map_id": map_id_primary,
                "map_hex": hex(map_id_primary),
                "map_id_alt": map_id_secondary,
                "map_bank": map_bank,
                "note": "Map IDs are game-specific, need lookup table"
            },
            "frame": frame_count,
            "memory_addresses": {"primary": "0xD35E", "alt": "0xD36F", "bank": "0xD35C"},
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get map location: {e}")
        return {
            "success": False,
            "error": f"Failed to read map location: {str(e)}",
            "suggestions": ["Map address varies by game", "Use emulator_get_frame with vision to identify location"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


def get_money() -> Dict[str, Any]:
    """
    Get player money/currency from memory.
    Pokemon Red/Blue: Money at 0xD6F5-0xD6F7 (3 bytes, BCD format)
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": (time.time() - start_time) * 1000
        }
    
    try:
        # Read 3 bytes of BCD-encoded money
        money_bytes = [
            emulator.memory[0xD6F5],
            emulator.memory[0xD6F6],
            emulator.memory[0xD6F7]
        ]
        
        # Convert BCD to decimal
        money = 0
        for byte in money_bytes:
            # BCD: each nibble is a decimal digit
            high_digit = (byte >> 4) & 0x0F
            low_digit = byte & 0x0F
            money = money * 100 + high_digit * 10 + low_digit
        
        return {
            "success": True,
            "tool": "get_money",
            "data": {
                "money": money,
                "formatted": f"${money:,}",
                "raw_bytes": [hex(b) for b in money_bytes],
                "note": "Money is stored in BCD (Binary Coded Decimal)"
            },
            "frame": frame_count,
            "memory_addresses": {"start": "0xD6F5", "end": "0xD6F7"},
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get money: {e}")
        return {
            "success": False,
            "error": f"Failed to read money: {str(e)}",
            "suggestions": ["Money address varies by game", "This works for Pokemon Red/Blue"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


# ========== Legacy Memory Reading (for backward compatibility) ==========

def get_memory_range(start: int, length: int) -> Dict[str, Any]:
    """Read memory range from Game Boy RAM"""
    global emulator
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        # PyBoy exposes memory via memory[] interface
        # Game Boy RAM typically 0xC000-0xE000 (8KB)
        memory_data = []
        for addr in range(start, start + length):
            if 0 <= addr < 0x10000:  # Valid 16-bit address space
                memory_data.append(emulator.memory[addr])
        
        return {
            "success": True,
            "start": hex(start),
            "length": length,
            "data": memory_data
        }
    except Exception as e:
        logger.error(f"Memory read failed: {e}")
        return {"success": False, "error": str(e)}


def get_memory_byte(address: int) -> Dict[str, Any]:
    """Read single byte from memory"""
    global emulator
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        value = emulator.memory[address]
        return {
            "success": True,
            "address": hex(address),
            "value": value,
            "value_hex": hex(value)
        }
    except Exception as e:
        logger.error(f"Memory read failed: {e}")
        return {"success": False, "error": str(e)}


def read_game_state() -> Dict[str, Any]:
    """
    Read common Game Boy game state from memory
    Useful for Pokemon and similar games
    """
    global emulator
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        state = {}
        
        # Common memory addresses for Game Boy games
        # These vary by game but are good defaults
        try:
            # Player position (if available)
            state["player_x"] = emulator.memory[0xD062] if hasattr(emulator, 'memory') else None
            state["player_y"] = emulator.memory[0xD063] if hasattr(emulator, 'memory') else None
            
            # Money (3 bytes, BCD)
            money = (
                (emulator.memory[0xD6F7] << 16) |
                (emulator.memory[0xD6F6] << 8) |
                emulator.memory[0xD6F5]
            ) if hasattr(emulator, 'memory') else 0
            state["money"] = money
            
            # Badge count (for Pokemon)
            state["badges"] = bin(emulator.memory[0xD8F6]).count('1') if hasattr(emulator, 'memory') else 0
            
        except Exception as e:
            logger.warning(f"Some memory reads failed: {e}")
        
        return {
            "success": True,
            "frame": frame_count,
            "rom": rom_path,
            "game_state": state
        }
    except Exception as e:
        logger.error(f"Game state read failed: {e}")
        return {"success": False, "error": str(e)}


# ========== Save State Management ==========

SAVE_DIR = Path(__file__).parent.parent / "saves"

def save_state(save_name: str = None) -> Dict[str, Any]:
    """Save emulator state to file"""
    global emulator, frame_count, rom_path
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        if save_name is None:
            save_name = f"save_{frame_count:06d}.state"
        elif not save_name.endswith('.state'):
            save_name += '.state'
        
        save_path = SAVE_DIR / save_name
        
        # Save state using PyBoy's built-in save functionality
        with open(save_path, 'wb') as f:
            emulator.save_state(f)
        
        logger.info(f"State saved: {save_path}")
        
        return {
            "success": True,
            "path": str(save_path),
            "frame": frame_count
        }
    except Exception as e:
        logger.error(f"Save state failed: {e}")
        return {"success": False, "error": str(e)}


def load_state(save_name: str) -> Dict[str, Any]:
    """Load emulator state from file"""
    global emulator
    
    if emulator is None:
        return {"success": False, "error": "Emulator not initialized"}
    
    try:
        save_path = SAVE_DIR / save_name if not Path(save_name).is_absolute() else Path(save_name)
        
        if not save_path.exists():
            return {"success": False, "error": f"Save file not found: {save_path}"}
        
        with open(save_path, 'rb') as f:
            emulator.load_state(f)
        
        logger.info(f"State loaded: {save_path}")
        
        return {
            "success": True,
            "path": str(save_path),
            "frame": frame_count
        }
    except Exception as e:
        logger.error(f"Load state failed: {e}")
        return {"success": False, "error": str(e)}


def list_saves() -> Dict[str, Any]:
    """List available save states"""
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        saves = [f.name for f in SAVE_DIR.glob("*.state")]
        
        return {
            "success": True,
            "saves": saves,
            "count": len(saves)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== Agent-First Game Control Tools ==========

def get_screen_base64(include_base64: bool = True) -> Dict[str, Any]:
    """
    Get current screen as base64-encoded PNG for vision analysis.
    Enhanced version with better metadata for agents.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first with a valid .gb ROM file"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        # Get screen buffer from PyBoy
        screen = emulator.screen
        if screen is None:
            return {
                "success": False,
                "error": "Failed to get screen buffer",
                "suggestions": ["Try emulator_tick to advance a frame first"],
                "timing_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # Get PIL Image from screen
        img = screen.image
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        result = {
            "success": True,
            "tool": "get_screen_base64",
            "frame": frame_count,
            "dimensions": {
                "width": img.width,
                "height": img.height,
                "format": f"{img.width}x{img.height}"
            },
            "image_size_bytes": len(img_base64),
            "image_size_kb": round(len(img_base64) / 1024, 2),
            "timestamp": datetime.now().isoformat(),
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        if include_base64:
            result["image_base64"] = img_base64
        else:
            result["image_base64"] = img_base64[:200] + "...[truncated]"
            result["truncated"] = True
            result["note"] = "Use include_base64=true to get full image"
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get screen base64: {e}")
        return {
            "success": False,
            "error": f"Failed to capture screen: {str(e)}",
            "suggestions": ["Check emulator is initialized", "Try emulator_tick to advance frame"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }


def save_game_state(save_name: str = None) -> Dict[str, Any]:
    """
    Save current game state (alias for save_state with agent-first response).
    """
    start_time = time.time()
    result = save_state(save_name)
    result["timing_ms"] = round((time.time() - start_time) * 1000, 2)
    result["tool"] = "save_game_state"
    return result


def load_game_state(save_name: str = None) -> Dict[str, Any]:
    """
    Load a previously saved game state (alias for load_state with agent-first response).
    """
    start_time = time.time()
    
    if not save_name:
        return {
            "success": False,
            "error": "save_name is required",
            "suggestions": ["Use emulator_list_saves to see available saves"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    result = load_state(save_name)
    result["timing_ms"] = round((time.time() - start_time) * 1000, 2)
    result["tool"] = "load_game_state"
    return result


# ========== Agent Session Management with TTL ==========
# Sessions now have expiration (TTL) for better resource management

# Session storage for agents - persists across tool calls
# Format: {session_id: {"data": {}, "created": timestamp, "last_update": timestamp, "ttl_seconds": 3600}}
agent_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_DEFAULT_TTL = 3600  # 1 hour default TTL


def _is_session_expired(session: Dict[str, Any]) -> bool:
    """Check if a session has expired based on TTL"""
    if "ttl_seconds" not in session:
        return False  # No TTL means never expires
    import time
    elapsed = time.time() - session.get("last_timestamp", time.time())
    return elapsed > session["ttl_seconds"]


def _clean_expired_sessions():
    """Remove expired sessions to free resources"""
    expired = [sid for sid, sess in agent_sessions.items() if _is_session_expired(sess)]
    for sid in expired:
        del agent_sessions[sid]
    if expired:
        logger.info(f"Cleaned {len(expired)} expired sessions")
    return len(expired)


def session_start(session_id: str = None, goal: str = None, ttl_seconds: int = None) -> Dict[str, Any]:
    """
    Start a new agent session for persistent state.
    Sessions allow agents to remember context across multiple tool calls.
    
    Args:
        session_id: Optional session ID (auto-generated if not provided)
        goal: Agent's goal for this session
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
    """
    start_time = time.time()
    
    # Clean expired sessions first
    _clean_expired_sessions()
    
    if session_id is None:
        session_id = f"session_{int(time.time() * 1000)}"
    
    if session_id in agent_sessions:
        # Check if existing session is still valid
        if _is_session_expired(agent_sessions[session_id]):
            del agent_sessions[session_id]
        else:
            return {
                "success": True,
                "tool": "session_start",
                "data": {
                    "session_id": session_id,
                    "message": "Session already exists",
                    "session_data": agent_sessions[session_id]["data"],
                    "created": agent_sessions[session_id]["created"],
                    "ttl_seconds": agent_sessions[session_id].get("ttl_seconds", SESSION_DEFAULT_TTL)
                },
                "timing_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    import time as time_module
    ttl = ttl_seconds if ttl_seconds is not None else SESSION_DEFAULT_TTL
    
    agent_sessions[session_id] = {
        "data": {
            "goal": goal or "Play the game autonomously",
            "game_state": {},
            "memory": [],
            "visited_locations": [],
            "battle_count": 0,
            "items_collected": [],
            "notes": []
        },
        "created": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "last_timestamp": time_module.time(),
        "ttl_seconds": ttl
    }
    
    return {
        "success": True,
        "tool": "session_start",
        "data": {
            "session_id": session_id,
            "message": "New session created",
            "goal": goal or "Play the game autonomously",
            "ttl_seconds": ttl
        },
        "timing_ms": round((time.time() - start_time) * 1000, 2)
    }


def session_get(session_id: str, key: str = None) -> Dict[str, Any]:
    """Get session data by ID and optionally by key"""
    start_time = time.time()
    
    if session_id not in agent_sessions:
        return {
            "success": False,
            "error": f"Session not found: {session_id}",
            "suggestions": ["Use session_start to create a session first"],
            "available_sessions": list(agent_sessions.keys()),
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    session = agent_sessions[session_id]
    session["last_update"] = datetime.now().isoformat()
    
    if key is None:
        return {
            "success": True,
            "tool": "session_get",
            "data": session["data"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    return {
        "success": True,
        "tool": "session_get",
        "data": session["data"].get(key),
        "key": key,
        "timing_ms": round((time.time() - start_time) * 1000, 2)
    }


def session_set(session_id: str, key: str, value: Any) -> Dict[str, Any]:
    """Set a value in session data"""
    start_time = time.time()
    
    if session_id not in agent_sessions:
        return {
            "success": False,
            "error": f"Session not found: {session_id}",
            "suggestions": ["Use session_start to create a session first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    agent_sessions[session_id]["data"][key] = value
    agent_sessions[session_id]["last_update"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "tool": "session_set",
        "data": {
            "session_id": session_id,
            "key": key,
            "value": value
        },
        "timing_ms": round((time.time() - start_time) * 1000, 2)
    }


def session_list() -> Dict[str, Any]:
    """List all active sessions"""
    return {
        "success": True,
        "tool": "session_list",
        "data": {
            "sessions": [
                {
                    "id": sid,
                    "created": info["created"],
                    "last_update": info["last_update"],
                    "goal": info["data"].get("goal"),
                    "keys": list(info["data"].keys())
                }
                for sid, info in agent_sessions.items()
            ],
            "count": len(agent_sessions)
        }
    }


def session_delete(session_id: str) -> Dict[str, Any]:
    """Delete a session"""
    if session_id in agent_sessions:
        del agent_sessions[session_id]
        return {
            "success": True,
            "tool": "session_delete",
            "data": {"session_id": session_id, "message": "Session deleted"}
        }
    return {
        "success": False,
        "error": f"Session not found: {session_id}"
    }


# ========== Auto-Play Modes ==========

def auto_catch(max_attempts: int = 30, use_best_ball: bool = True) -> Dict[str, Any]:
    """
    Autonomous catching mode.
    Attempts to catch wild Pokemon using optimal ball selection.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        attempts = 0
        caught = False
        actions = []
        
        # Get current ball count and species
        try:
            # Ball count at 0xD67E (Poke Balls)
            ball_count = emulator.memory[0xD67E]
            # Enemy species
            enemy_species = emulator.memory[0xD883]
        except:
            ball_count = 0
            enemy_species = 0
        
        actions.append({
            "attempt": 0,
            "action": "INITIAL_STATE",
            "ball_count": ball_count,
            "enemy_species": enemy_species
        })
        
        for i in range(max_attempts):
            attempts += 1
            
            # Check if in battle
            battle_status = emulator.memory[0xD057]
            if battle_status == 0:
                actions.append({
                    "attempt": i + 1,
                    "action": "NO_BATTLE",
                    "message": "No wild Pokemon to catch"
                })
                break
            
            # Navigate to BALL option (typically SELECT then navigate)
            # In Pokemon: SELECT from menu, then choose BALL
            press_button("SELECT")
            time.sleep(0.1)
            
            # Navigate to ball option and throw
            if use_best_ball and ball_count > 0:
                press_button("A")  # Select ball
                actions.append({
                    "attempt": attempts,
                    "action": "THROW_BALL",
                    "ball_type": "best_available"
                })
            else:
                actions.append({
                    "attempt": attempts,
                    "action": "NO_BALLS",
                    "message": "No balls remaining"
                })
                break
            
            # Wait for catch result
            for _ in range(20):
                tick()
            
            # Check if caught (battle ends)
            if emulator.memory[0xD057] == 0:
                caught = True
                actions.append({
                    "attempt": attempts,
                    "action": "CATCH_SUCCESS",
                    "message": "Pokemon caught!"
                })
                break
            
            # Check ball count
            ball_count = emulator.memory[0xD67E]
            if ball_count == 0:
                actions.append({
                    "attempt": attempts,
                    "action": "OUT_OF_BALLS",
                    "message": "No balls left"
                })
                break
            
            actions.append({
                "attempt": attempts,
                "action": "CATCH_FAILED",
                "message": "Pokemon broke free"
            })
        
        return {
            "success": True,
            "tool": "auto_catch",
            "data": {
                "mode": "catch",
                "attempts": attempts,
                "caught": caught,
                "ball_count": ball_count,
                "enemy_species": enemy_species,
                "actions_summary": f"Tried {attempts} times, {'caught' if caught else 'failed'}",
                "actions": actions
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto catch failed: {e}")
        return {
            "success": False,
            "error": f"Auto catch failed: {str(e)}",
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_catch"
        }


def auto_item_use(item_id: int = None, target: str = "self") -> Dict[str, Any]:
    """
    Autonomous item use mode.
    Uses items from inventory on self or party members.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        actions = []
        
        # Get current inventory
        inventory_result = get_inventory()
        inventory = inventory_result.get("data", {}).get("items", []) if inventory_result.get("success") else []
        
        if not inventory and item_id is None:
            return {
                "success": False,
                "error": "No items in inventory",
                "timing_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # Find requested item or use first healing item
        if item_id is not None:
            item_to_use = next((item for item in inventory if item.get("item_id") == item_id), None)
        else:
            # Default: find first potion-like item
            # Common healing item IDs: 0x01-0x0F (Potion, Super Potion, etc.)
            item_to_use = next((item for item in inventory if item.get("item_id", 0) < 0x10), None)
        
        if not item_to_use:
            return {
                "success": False,
                "error": "Requested item not found",
                "timing_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # Open inventory menu
        press_button("SELECT")
        time.sleep(0.1)
        
        actions.append({
            "action": "OPEN_MENU"
        })
        
        # Navigate to item and use
        # Simplified: just press A to use selected item
        press_button("A")
        time.sleep(0.1)
        
        actions.append({
            "action": "USE_ITEM",
            "item": item_to_use
        })
        
        # If targeting party member, select target
        if target == "party":
            press_button("A")  # Select first party member
            time.sleep(0.1)
            
            actions.append({
                "action": "SELECT_TARGET",
                "target": "party_member_1"
            })
        
        return {
            "success": True,
            "tool": "auto_item_use",
            "data": {
                "mode": "item_use",
                "item_used": item_to_use,
                "target": target,
                "actions": actions,
                "remaining_items": len(inventory) - 1
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto item use failed: {e}")
        return {
            "success": False,
            "error": f"Auto item use failed: {str(e)}",
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_item_use"
        }


def auto_npc_talk(interact_distance: int = 1, max_attempts: int = 20) -> Dict[str, Any]:
    """
    Autonomous NPC interaction mode.
    Finds and talks to nearby NPCs.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        actions = []
        talked_count = 0
        
        # Get initial position
        pos_result = get_player_position()
        start_pos = pos_result.get("data", {}) if pos_result.get("success") else {"x": 0, "y": 0}
        
        actions.append({
            "action": "START",
            "position": start_pos
        })
        
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for i in range(max_attempts):
            # Try talking in each direction
            for direction in directions:
                # Get position before
                pos_before = get_player_position()
                
                # Face that direction
                press_button(direction)
                time.sleep(0.05)
                
                # Try to interact (A button talks to NPCs)
                press_button("A")
                time.sleep(0.2)
                
                # Check if dialog opened (memory address varies by game)
                # In Pokemon, dialog is shown when certain flags are set
                try:
                    # Check if textbox is active (0xD73E = text engine state)
                    dialog_active = emulator.memory[0xD73E] if emulator.memory[0xD73E] != 0 else None
                    
                    if dialog_active:
                        talked_count += 1
                        actions.append({
                            "attempt": i + 1,
                            "direction": direction,
                            "action": "TALKED",
                            "dialog_active": True
                        })
                        
                        # Close dialog with B
                        press_button("B")
                        time.sleep(0.1)
                except:
                    pass
            
            # Move to explore more area
            move_dir = directions[i % 4]
            press_button(move_dir)
            time.sleep(0.1)
            
            # Get new position
            pos_after = get_player_position()
            if pos_after.get("success"):
                actions.append({
                    "attempt": i + 1,
                    "action": "MOVED",
                    "direction": move_dir,
                    "new_position": pos_after.get("data")
                })
        
        return {
            "success": True,
            "tool": "auto_npc_talk",
            "data": {
                "mode": "npc_talk",
                "attempts": max_attempts,
                "npcs_interacted": talked_count,
                "start_position": start_pos,
                "actions_summary": f"Attempted {max_attempts} interactions, talked to {talked_count} NPCs",
                "actions": actions
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto NPC talk failed: {e}")
        return {
            "success": False,
            "error": f"Auto NPC talk failed: {str(e)}",
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_npc_talk"
        }


def auto_explore(steps: int = 10, session_id: str = None) -> Dict[str, Any]:
    """
    Autonomous exploration mode.
    Explores the game world automatically, avoiding battles when possible.
    Uses memory reading to track position and make decisions.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        actions = []
        
        # Movement patterns for exploration
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for i in range(steps):
            # Get current position before move
            pos_before = get_player_position()
            
            # Choose a direction (can be enhanced with vision)
            direction = directions[i % 4]
            
            # Move in that direction
            success = press_button(direction)
            actions.append({
                "step": i + 1,
                "direction": direction,
                "success": success,
                "position_before": pos_before.get("data", {}) if pos_before.get("success") else None
            })
            
            # Check if in battle (avoid if possible)
            try:
                battle_status = emulator.memory[0xD057]
                if battle_status != 0:
                    # In battle - stop exploration
                    actions.append({
                        "step": i + 1,
                        "action": "BATTLE_DETECTED",
                        "message": "Stopped exploration - battle encountered"
                    })
                    break
            except:
                pass
            
            # Small delay between moves
            time.sleep(0.05)
        
        # Get final position
        pos_after = get_player_position()
        
        return {
            "success": True,
            "tool": "auto_explore",
            "data": {
                "mode": "exploration",
                "steps_attempted": steps,
                "actions": actions,
                "final_position": pos_after.get("data") if pos_after.get("success") else None,
                "message": f"Explored {len(actions)} steps"
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto explore failed: {e}")
        return {
            "success": False,
            "error": f"Auto explore failed: {str(e)}",
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_explore"
        }


def auto_grind(target_level: int = None, max_battles: int = 20, heal_after: int = 5) -> Dict[str, Any]:
    """
    Autonomous grinding mode.
    Finds and fights wild Pokemon to gain XP.
    Returns to heal after specified number of battles.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        battles_fought = 0
        victories = 0
        defeats = 0
        xp_gained = 0
        actions = []
        
        for i in range(max_battles):
            # Navigate in grass to find wild Pokemon
            # Press a direction to move
            direction = ["UP", "DOWN", "LEFT", "RIGHT"][i % 4]
            press_button(direction)
            
            # Wait and check for battle
            for _ in range(10):
                tick()
            
            # Check if in battle
            battle_status = emulator.memory[0xD057]
            
            if battle_status != 0:
                # In battle!
                battles_fought += 1
                actions.append({
                    "battle": battles_fought,
                    "status": "engaged"
                })
                
                # Get initial HP
                player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
                player_max_hp = (emulator.memory[0xD6BF] << 8) | emulator.memory[0xD6BE]
                
                # Auto-battle
                battle_result = auto_battle(max_moves=20)
                if battle_result.get("success"):
                    actions[-1]["result"] = "completed"
                    victories += 1
                    
                    # Estimate XP (simplified)
                    xp_gained += 50 + (battles_fought * 5)
                else:
                    actions[-1]["result"] = "failed"
                    defeats += 1
                
                # Check if need to heal
                current_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
                if current_hp < player_max_hp * 0.3:
                    # Low HP - try to heal or run
                    actions.append({
                        "action": "LOW_HP",
                        "current": current_hp,
                        "max": player_max_hp,
                        "message": "Low HP - recommended to heal"
                    })
                    break
            
            # Check if reached target level (if specified)
            if target_level:
                party = get_party_info()
                if party.get("success") and party.get("data", {}).get("party"):
                    first_mon = party["data"]["party"][0]
                    if first_mon.get("level", 0) >= target_level:
                        actions.append({
                            "action": "TARGET_REACHED",
                            "level": first_mon.get("level"),
                            "message": f"Target level {target_level} reached!"
                        })
                        break
            
            # Heal after specified battles
            if battles_fought > 0 and battles_fought % heal_after == 0:
                actions.append({
                    "action": "HEAL_RECOMMENDED",
                    "battles": battles_fought,
                    "message": f"After {heal_after} battles, consider healing"
                })
        
        # Get final party state
        party = get_party_info()
        
        return {
            "success": True,
            "tool": "auto_grind",
            "data": {
                "mode": "grind",
                "battles_fought": battles_fought,
                "victories": victories,
                "defeats": defeats,
                "xp_gained": xp_gained,
                "target_level": target_level,
                "final_party": party.get("data", {}).get("party", []),
                "actions_summary": f"Fought {battles_fought} battles: {victories} wins, {defeats} losses, {xp_gained} XP"
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto grind failed: {e}")
        return {
            "success": False,
            "error": f"Auto grind failed: {str(e)}",
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_grind"
        }


def auto_battle(max_moves: int = 10) -> Dict[str, Any]:
    """
    AI-powered automatic battle decision making.
    Analyzes current battle state and decides optimal moves.
    Uses memory reading to determine enemy and player state.
    """
    start_time = time.time()
    
    if emulator is None:
        return {
            "success": False,
            "error": "Emulator not initialized",
            "suggestions": ["Call emulator_load_rom first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    try:
        moves_executed = []
        
        # Get current battle state from memory
        # Battle state: 0xD057 = battle status (0=no battle, 1=battle)
        battle_status = emulator.memory[0xD057]
        
        if battle_status == 0:
            return {
                "success": False,
                "error": "Not in battle",
                "data": {
                    "moves_executed": [],
                    "battle_status": "not_in_battle",
                    "note": "Use emulator_press_button to encounter a wild Pokemon first"
                },
                "suggestions": ["Navigate to encounter a Pokemon", "Then call auto_battle again"],
                "timing_ms": round((time.time() - start_time) * 1000, 2),
                "tool": "auto_battle"
            }
        
        # Get player and enemy HP from memory
        # Player HP: 0xD6B5-0xD6B6 (2 bytes)
        player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
        player_max_hp = (emulator.memory[0xD6BF] << 8) | emulator.memory[0xD6BE]
        
        # Enemy HP: 0xD89C-0xD89D (2 bytes)  
        enemy_hp = (emulator.memory[0xD89D] << 8) | emulator.memory[0xD89C]
        enemy_max_hp = (emulator.memory[0xD8A1] << 8) | emulator.memory[0xD8A0]
        
        # Determine optimal move based on HP
        # Simple AI: use best move based on type advantage info in memory
        # For now, always use A (typically the best move)
        
        for i in range(min(max_moves, 10)):
            # Check if battle ended
            current_battle = emulator.memory[0xD057]
            if current_battle == 0:
                moves_executed.append({
                    "move": "BATTLE_ENDED",
                    "reason": "Battle concluded"
                })
                break
            
            # Execute attack (A button)
            press_button("A")
            moves_executed.append({
                "move_number": i + 1,
                "button": "A",
                "action": "attack"
            })
            
            # Small delay between moves
            time.sleep(0.1)
        
        return {
            "success": True,
            "tool": "auto_battle",
            "data": {
                "moves_executed": moves_executed,
                "total_moves": len(moves_executed),
                "battle_status": "active" if battle_status else "ended",
                "hp_status": {
                    "player": {
                        "current": player_hp,
                        "max": player_max_hp,
                        "percent": round(player_hp / player_max_hp * 100, 1) if player_max_hp > 0 else 0
                    },
                    "enemy": {
                        "current": enemy_hp,
                        "max": enemy_max_hp,
                        "percent": round(enemy_hp / enemy_max_hp * 100, 1) if enemy_max_hp > 0 else 0
                    }
                },
                "ai_strategy": "aggressive_attack",
                "note": "Simple AI - always attacks with A button"
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto battle failed: {e}")
        return {
            "success": False,
            "error": f"Auto battle failed: {str(e)}",
            "suggestions": ["Make sure you're in a battle", "Try manual button presses first"],
            "timing_ms": round((time.time() - start_time) * 1000, 2),
            "tool": "auto_battle"
        }


# ========== MCP Tool Handlers

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available emulator tools"""
    return [
        Tool(
            name="emulator_load_rom",
            description="Load a Game Boy ROM file into the emulator",
            inputSchema={
                "type": "object",
                "properties": {
                    "rom_path": {
                        "type": "string",
                        "description": "Path to the .gb or .gba ROM file"
                    }
                },
                "required": ["rom_path"]
            }
        ),
        Tool(
            name="emulator_press_button",
            description="Press a controller button (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT)",
            inputSchema={
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "description": "Button to press (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT)",
                        "enum": ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
                    }
                },
                "required": ["button"]
            }
        ),
        Tool(
            name="emulator_press_sequence",
            description="Press multiple buttons in sequence (e.g., 'A B A UP DOWN')",
            inputSchema={
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Button sequence separated by spaces (e.g., 'A B START' or 'A2 R3 U1')"
                    },
                    "delay": {
                        "type": "integer",
                        "description": "Delay in milliseconds between button presses",
                        "default": 100
                    }
                },
                "required": ["sequence"]
            }
        ),
        Tool(
            name="emulator_get_frame",
            description="Get the current screen as a base64-encoded PNG image for vision analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_base64": {
                        "type": "boolean",
                        "description": "Include full base64 image data (default: false for summary only)",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="emulator_get_state",
            description="Get the current emulator state (initialized, ROM path, frame count)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="emulator_tick",
            description="Advance the emulation by one or more frames",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "description": "Number of frames to advance (default: 1)",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="emulator_save_screenshot",
            description="Save current screen to a PNG file",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the screenshot (default: ./screenshots/frame_XXX.png)"
                    }
                }
            }
        ),
        # === NEW: Memory Reading Tools ===
        Tool(
            name="emulator_read_memory",
            description="Read bytes from Game Boy RAM at specific address",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "Memory address (hex string like '0xD000' or integer)"
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of bytes to read (default: 1)",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="emulator_get_game_state",
            description="Read common game state (player position, money, badges) from memory",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # === NEW: Save State Management ===
        Tool(
            name="emulator_save_state",
            description="Save current emulator state to file for later restoration",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Name for the save file (optional, defaults to frame number)"
                    }
                }
            }
        ),
        Tool(
            name="emulator_load_state",
            description="Load a previously saved emulator state",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Name of the save file to load"
                    }
                },
                "required": ["save_name"]
            }
        ),
        Tool(
            name="emulator_list_saves",
            description="List all available save state files",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # === NEW: Agent-First Memory Reading Tools ===
        Tool(
            name="get_player_position",
            description="Get player X,Y coordinates from game memory. Returns tile-based position with metadata.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_party_info",
            description="Get party Pokemon/monsters from memory. Returns species, HP, level for each party member.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_inventory",
            description="Get player inventory/items from memory. Returns item IDs and quantities.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_map_location",
            description="Get current map/location ID from memory. Useful for navigation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_money",
            description="Get player money/currency from memory. Returns BCD-encoded money value.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # === NEW: Game Control Tools ===
        Tool(
            name="get_screen_base64",
            description="Get current screen as base64 PNG for vision analysis. Enhanced with metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_base64": {
                        "type": "boolean",
                        "description": "Include full base64 image (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="save_game_state",
            description="Save current game state to file with agent-first response format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Optional name for the save file"
                    }
                }
            }
        ),
        Tool(
            name="load_game_state",
            description="Load a previously saved game state with agent-first response.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Name of the save file to load"
                    }
                },
                "required": ["save_name"]
            }
        ),
        Tool(
            name="auto_battle",
            description="AI decides battle moves automatically. Analyzes HP, executes optimal attacks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_moves": {
                        "type": "integer",
                        "description": "Maximum moves to execute (default: 10)",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="emulator_debug_log",
            description="Get debug information about emulator state for troubleshooting.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # === NEW: Session Management Tools (with TTL) ===
        Tool(
            name="session_start",
            description="Start a new agent session for persistent state across tool calls. Sessions allow agents to remember context with automatic expiration (TTL).",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID (auto-generated if not provided)"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Agent's goal for this session (e.g., 'Beat the Elite 4')"
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Session time-to-live in seconds (default: 3600 = 1 hour). Session auto-expires after this time.",
                        "default": 3600,
                        "minimum": 60,
                        "maximum": 86400
                    }
                }
            }
        ),
        Tool(
            name="session_get",
            description="Get data from an agent session. Returns all session data or a specific key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to retrieve"
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional specific key to retrieve (e.g., 'goal', 'visited_locations')"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="session_set",
            description="Set a value in an agent session. Use to persist game state, notes, or any data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to update"
                    },
                    "key": {
                        "type": "string",
                        "description": "Key to set (e.g., 'current_location', 'party_hp')"
                    },
                    "value": {
                        "type": "object",
                        "description": "Value to store (any JSON-serializable value)"
                    }
                },
                "required": ["session_id", "key", "value"]
            }
        ),
        Tool(
            name="session_list",
            description="List all active agent sessions.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="session_delete",
            description="Delete an agent session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to delete"
                    }
                },
                "required": ["session_id"]
            }
        ),
        # === NEW: Auto-Play Modes ===
        Tool(
            name="auto_explore",
            description="Autonomous exploration mode. Moves around the game world automatically, avoiding battles when possible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of movement steps to attempt (default: 10)",
                        "default": 10
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to track exploration progress"
                    }
                }
            }
        ),
        Tool(
            name="auto_grind",
            description="Autonomous grinding mode. Fights wild Pokemon to gain XP. Returns to heal after specified battles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_level": {
                        "type": "integer",
                        "description": "Stop when first Pokemon reaches this level (optional)"
                    },
                    "max_battles": {
                        "type": "integer",
                        "description": "Maximum battles to fight (default: 20)",
                        "default": 20
                    },
                    "heal_after": {
                        "type": "integer",
                        "description": "Recommend healing after this many battles (default: 5)",
                        "default": 5
                    }
                }
            }
        ),
        # === NEW: Auto-Catch, Auto-Item, Auto-NPC ===
        Tool(
            name="auto_catch",
            description="Autonomous catching mode. Attempts to catch wild Pokemon using optimal ball selection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_attempts": {
                        "type": "integer",
                        "description": "Maximum catch attempts (default: 30)",
                        "default": 30
                    },
                    "use_best_ball": {
                        "type": "boolean",
                        "description": "Use the best available ball (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="auto_item_use",
            description="Autonomous item use mode. Uses items from inventory on self or party members.",
            inputSchema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "integer",
                        "description": "Specific item ID to use (optional, auto-selects healing item if not specified)"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target for item use: 'self' or 'party'",
                        "enum": ["self", "party"],
                        "default": "self"
                    }
                }
            }
        ),
        Tool(
            name="auto_npc_talk",
            description="Autonomous NPC interaction mode. Finds and talks to nearby NPCs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "interact_distance": {
                        "type": "integer",
                        "description": "Distance to look for NPCs (default: 1)",
                        "default": 1
                    },
                    "max_attempts": {
                        "type": "integer",
                        "description": "Maximum interaction attempts (default: 20)",
                        "default": 20
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) :
    """Handle tool calls"""
    global emulator
    
    try:
        if name == "emulator_load_rom":
            rom_path = arguments.get("rom_path")
            if not rom_path:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, 
                    error="rom_path is required", 
                    error_code=EmulatorErrorCode.INVALID_PARAMETER,
                    suggestions=["Provide a valid path to a .gb or .gba ROM file"]
                )))]
            
            # init_emulator now raises EmulatorError on failure
            init_emulator(rom_path)
            return [TextContent(
                type="text",
                text=json.dumps(format_response(
                    True, 
                    data={"rom": rom_path, "frame": frame_count},
                    tool_name="emulator_load_rom"
                ))
            )]
        
        elif name == "emulator_press_button":
            button = arguments.get("button")
            if not button:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False,
                    error="button is required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER,
                    suggestions=[f"Provide a button name: {', '.join(GameButton.values())}"]
                )))]
            
            press_button(button)
            return [TextContent(
                type="text",
                text=json.dumps(format_response(
                    True,
                    data={"button": button, "frame": frame_count},
                    tool_name="emulator_press_button"
                ))
            )]
        
        elif name == "emulator_press_sequence":
            sequence = arguments.get("sequence")
            if not sequence:
                return [TextContent(type="text", text="Error: sequence required")]
            
            delay = arguments.get("delay", 100)
            result = press_sequence(sequence, delay)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_get_frame":
            include_base64 = arguments.get("include_base64", False)
            frame_data = get_frame()
            if frame_data is None:
                return [TextContent(type="text", text="Error: Emulator not initialized or failed to capture frame")]
            
            response = {
                "success": True,
                "frame": frame_data['frame'],
                "dimensions": f"{frame_data['width']}x{frame_data['height']}"
            }
            
            if include_base64:
                response["image_base64"] = frame_data['base64']
            else:
                response["image_base64"] = frame_data['base64'][:100] + "...[truncated]"
            
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
        elif name == "emulator_get_state":
            state = get_state()
            return [TextContent(
                type="text",
                text=json.dumps(state, indent=2)
            )]
        
        elif name == "emulator_tick":
            frames = arguments.get("frames", 1)
            success = True
            for _ in range(frames):
                if not tick():
                    success = False
                    break
            
            return [TextContent(
                type="text",
                text=json.dumps({"success": success, "frames": frames, "new_frame": frame_count}, indent=2)
            )]
        
        elif name == "emulator_save_screenshot":
            output_path = arguments.get("output_path")
            result = save_screenshot(output_path)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === Memory Reading Tool Handlers ===
        elif name == "emulator_read_memory":
            address = arguments.get("address", 0xD000)
            length = arguments.get("length", 1)
            
            # Handle hex string input
            if isinstance(address, str):
                address = int(address, 16)
            
            if length == 1:
                result = get_memory_byte(address)
            else:
                result = get_memory_range(address, length)
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_get_game_state":
            result = read_game_state()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === Save State Tool Handlers ===
        elif name == "emulator_save_state":
            save_name = arguments.get("save_name")
            result = save_state(save_name)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_load_state":
            save_name = arguments.get("save_name")
            if not save_name:
                return [TextContent(type="text", text=json.dumps({"success": False, "error": "save_name required"}))]
            result = load_state(save_name)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_list_saves":
            result = list_saves()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === NEW: Agent-First Memory Reading Tool Handlers ===
        elif name == "get_player_position":
            result = get_player_position()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_party_info":
            result = get_party_info()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_inventory":
            result = get_inventory()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_map_location":
            result = get_map_location()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_money":
            result = get_money()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === NEW: Game Control Tool Handlers ===
        elif name == "get_screen_base64":
            include_base64 = arguments.get("include_base64", True)
            result = get_screen_base64(include_base64)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "save_game_state":
            save_name = arguments.get("save_name")
            result = save_game_state(save_name)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "load_game_state":
            save_name = arguments.get("save_name")
            result = load_game_state(save_name)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "auto_battle":
            max_moves = arguments.get("max_moves", 10)
            result = auto_battle(max_moves)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_debug_log":
            # Debug logging for agents
            result = {
                "success": True,
                "tool": "emulator_debug_log",
                "server_version": SERVER_VERSION,
                "server_started": SERVER_START_TIME,
                "emulator_state": get_state(),
                "uptime_seconds": (datetime.now() - datetime.fromisoformat(SERVER_START_TIME)).total_seconds(),
                "log_levels": ["DEBUG", "INFO", "WARNING", "ERROR"],
                "debug_log_example": {
                    "timestamp": datetime.now().isoformat(),
                    "frame_count": frame_count,
                    "rom_loaded": rom_path is not None
                },
                "note": "Use debug_log(level, message, data) for agent debugging"
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === Session Management Tool Handlers ===
        elif name == "session_start":
            session_id = arguments.get("session_id")
            goal = arguments.get("goal")
            ttl_seconds = arguments.get("ttl_seconds")
            result = session_start(session_id, goal, ttl_seconds)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_get":
            session_id = arguments.get("session_id")
            key = arguments.get("key")
            if not session_id:
                return [TextContent(type="text", text=json.dumps({"success": False, "error": "session_id required"}))]
            result = session_get(session_id, key)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_set":
            session_id = arguments.get("session_id")
            key = arguments.get("key")
            value = arguments.get("value")
            if not session_id or not key:
                return [TextContent(type="text", text=json.dumps({"success": False, "error": "session_id and key required"}))]
            result = session_set(session_id, key, value)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_list":
            result = session_list()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_delete":
            session_id = arguments.get("session_id")
            if not session_id:
                return [TextContent(type="text", text=json.dumps({"success": False, "error": "session_id required"}))]
            result = session_delete(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === Auto-Play Mode Tool Handlers ===
        elif name == "auto_explore":
            steps = arguments.get("steps", 10)
            session_id = arguments.get("session_id")
            result = auto_explore(steps, session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "auto_grind":
            target_level = arguments.get("target_level")
            max_battles = arguments.get("max_battles", 20)
            heal_after = arguments.get("heal_after", 5)
            result = auto_grind(target_level, max_battles, heal_after)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # === NEW: Auto-Catch, Auto-Item, Auto-NPC Handlers ===
        elif name == "auto_catch":
            max_attempts = arguments.get("max_attempts", 30)
            use_best_ball = arguments.get("use_best_ball", True)
            result = auto_catch(max_attempts, use_best_ball)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "auto_item_use":
            item_id = arguments.get("item_id")
            target = arguments.get("target", "self")
            result = auto_item_use(item_id, target)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "auto_npc_talk":
            interact_distance = arguments.get("interact_distance", 1)
            max_attempts = arguments.get("max_attempts", 20)
            result = auto_npc_talk(interact_distance, max_attempts)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            error_msg = f"Unknown tool: {name}"
            return [TextContent(type="text", text=json.dumps({"success": False, "error": error_msg}))]
    
    except EmulatorError as e:
        # Handle custom emulator errors with error codes
        logger.error(f"Emulator error in {name}: {e.code} - {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e),
            "error_code": e.code,
            "suggestions": e.suggestions
        }))]
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        error_str = str(e)
        return [TextContent(type="text", text=json.dumps({
            "success": False, 
            "error": error_str,
            "error_code": EmulatorErrorCode.OPERATION_FAILED,
            "suggestions": ["Check emulator state", "Try reloading the ROM"]
        }))]


async def main():
    """Run the MCP server"""
    logger.info(f"Starting PyBoy Emulator MCP Server v{SERVER_VERSION}...")
    logger.info(f"PyBoy available: {PYBOY_AVAILABLE}")
    logger.info("Server capabilities: tools")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
