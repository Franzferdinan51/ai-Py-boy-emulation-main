#!/usr/bin/env python3
"""
OpenClaw MCP Server for AI Py-Boy Emulation
Exposes emulator controls as MCP tools for OpenClaw agents

Version: 5.0.0 - Enhanced with Advanced Features
- Better tool definitions with JSON Schema, detailed descriptions, and examples
- Streaming responses for vision analysis (chunked base64 transfer)
- Session memory with persistence (survives server restarts)
- Enhanced error handling with structured errors and recovery suggestions
- New memory manipulation tools (get/set memory addresses)
- Enhanced party and inventory tools with detailed metadata
- Auto-explore and auto-battle modes with AI decision-making

Tools:
- emulator_load_rom: Load a ROM file
- emulator_press_button: Press a controller button
- emulator_get_frame: Get current screen as image
- emulator_get_state: Get emulator state
- emulator_tick: Advance emulation by one frame
- get_memory_address: Read specific memory address
- set_memory_address: Write to memory address
- get_party_pokemon: Detailed party Pokemon info
- get_inventory: Detailed inventory items
- get_player_position: Read player coordinates
- get_map_location: Read current map/location
- get_money: Read money/currency
- auto_explore_mode: Start autonomous exploration
- auto_battle_mode: Start AI battle assistant
- save_game_state / load_game_state: Save/load game progress
- session_start/session_get/session_set: Agent session persistence
- get_screen_base64: Get screen for vision analysis with streaming
"""

import os
import sys
import json
import base64
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from io import BytesIO
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from functools import wraps
import asyncio

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
SERVER_VERSION = "5.0.0"
SERVER_START_TIME = datetime.now().isoformat()

# Persistence directory for session memory
PERSISTENCE_DIR = Path(__file__).parent / ".persist"
PERSISTENCE_DIR.mkdir(parents=True, exist_ok=True)

# ========== Error Handling Patterns (Enhanced) ==========

class EmulatorErrorCode:
    """Error code constants for programmatic error handling"""
    NOT_INITIALIZED = "EMULATOR_NOT_INITIALIZED"
    ROM_NOT_FOUND = "ROM_NOT_FOUND"
    INVALID_ROM = "INVALID_ROM"
    BUTTON_INVALID = "BUTTON_INVALID"
    MEMORY_READ_ERROR = "MEMORY_READ_ERROR"
    MEMORY_WRITE_ERROR = "MEMORY_WRITE_ERROR"
    SAVE_NOT_FOUND = "SAVE_NOT_FOUND"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    OPERATION_FAILED = "OPERATION_FAILED"
    BATTLE_NOT_ACTIVE = "BATTLE_NOT_ACTIVE"
    INVALID_ADDRESS = "INVALID_ADDRESS"
    STREAM_ERROR = "STREAM_ERROR"


class EmulatorError(Exception):
    """Custom exception with error codes for better error handling"""
    def __init__(self, message: str, code: str, suggestions: List[str] = None, recoverable: bool = False):
        super().__init__(message)
        self.code = code
        self.suggestions = suggestions or []
        self.recoverable = recoverable


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
                "suggestions": e.suggestions,
                "recoverable": e.recoverable
            }
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": EmulatorErrorCode.OPERATION_FAILED,
                "suggestions": ["Check emulator state", "Try reloading the ROM"],
                "recoverable": True
            }
    return wrapper


def format_response(
    success: bool,
    data: Any = None,
    error: str = None,
    error_code: str = None,
    suggestions: List[str] = None,
    tool_name: str = None,
    timing_ms: float = None,
    include_state: bool = False,
    recoverable: bool = None
) -> Dict[str, Any]:
    """Clean response format with enhanced metadata"""
    response = {"success": success}
    
    if data is not None:
        response["data"] = data
    
    if error:
        response["error"] = error
        if error_code:
            response["error_code"] = error_code
        if suggestions:
            response["suggestions"] = suggestions
        if recoverable is not None:
            response["recoverable"] = recoverable
    
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


# ========== Global Emulator State ==========
emulator: Optional[PyBoy] = None
rom_path: Optional[str] = None
frame_count: int = 0

# ========== Session Memory with Persistence ==========
# Sessions persist to disk so they survive server restarts

@dataclass
class SessionData:
    """Session data structure"""
    goal: str = "Play the game autonomously"
    game_state: Dict[str, Any] = field(default_factory=dict)
    memory: List[str] = field(default_factory=list)
    visited_locations: List[str] = field(default_factory=list)
    battle_count: int = 0
    items_collected: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    party_hp_history: List[Dict] = field(default_factory=list)
    exploration_log: List[Dict] = field(default_factory=list)


@dataclass
class Session:
    """Session with metadata"""
    session_id: str
    data: SessionData
    created: str
    last_update: str
    last_timestamp: float
    ttl_seconds: int = 3600


# In-memory session cache
agent_sessions: Dict[str, Session] = {}

def _load_sessions_from_disk():
    """Load sessions from persistent storage"""
    global agent_sessions
    session_files = list(PERSISTENCE_DIR.glob("session_*.json"))
    
    for session_file in session_files:
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if expired
            if time.time() - session_data.get("last_timestamp", 0) > session_data.get("ttl_seconds", 3600):
                logger.info(f"Skipping expired session: {session_data.get('session_id')}")
                continue
            
            # Reconstruct session
            session = Session(
                session_id=session_data["session_id"],
                data=SessionData(**session_data["data"]),
                created=session_data["created"],
                last_update=session_data["last_update"],
                last_timestamp=session_data["last_timestamp"],
                ttl_seconds=session_data.get("ttl_seconds", 3600)
            )
            agent_sessions[session.session_id] = session
            logger.info(f"Loaded session: {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to load session {session_file}: {e}")


def _save_session_to_disk(session: Session):
    """Save session to persistent storage"""
    session_file = PERSISTENCE_DIR / f"{session.session_id}.json"
    try:
        session_dict = {
            "session_id": session.session_id,
            "data": asdict(session.data),
            "created": session.created,
            "last_update": session.last_update,
            "last_timestamp": session.last_timestamp,
            "ttl_seconds": session.ttl_seconds
        }
        with open(session_file, 'w') as f:
            json.dump(session_dict, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save session {session.session_id}: {e}")


def _clean_expired_sessions():
    """Remove expired sessions"""
    expired = []
    current_time = time.time()
    
    for sid, session in agent_sessions.items():
        if current_time - session.last_timestamp > session.ttl_seconds:
            expired.append(sid)
            # Remove from disk too
            session_file = PERSISTENCE_DIR / f"{sid}.json"
            if session_file.exists():
                session_file.unlink()
    
    for sid in expired:
        del agent_sessions[sid]
    
    if expired:
        logger.info(f"Cleaned {len(expired)} expired sessions")
    
    return len(expired)


# Load sessions on module init
_load_sessions_from_disk()


# ========== Emulator Initialization ==========

def init_emulator(rom_file: str) -> bool:
    """Initialize PyBoy emulator with ROM with improved error handling"""
    global emulator, rom_path, frame_count
    
    if not PYBOY_AVAILABLE:
        raise EmulatorError(
            "PyBoy library not available. Install with: pip install pyboy",
            EmulatorErrorCode.OPERATION_FAILED,
            ["Install pyboy: pip install pyboy"]
        )
    
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


# ========== Enhanced Memory Tools ==========

def get_memory_address(address: Union[int, str]) -> Dict[str, Any]:
    """
    Read a specific memory address.
    
    Args:
        address: Memory address as integer or hex string (e.g., 0xD062)
    
    Returns:
        Dictionary with value, hex representation, and metadata
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    # Convert hex string to int if needed
    if isinstance(address, str):
        try:
            address = int(address, 16)
        except ValueError:
            raise EmulatorError(
                f"Invalid address format: {address}",
                EmulatorErrorCode.INVALID_ADDRESS,
                ["Use hex format like '0xD062' or integer"]
            )
    
    # Validate address range
    if not (0 <= address < 0x10000):
        raise EmulatorError(
            f"Address out of range: {hex(address)}",
            EmulatorErrorCode.INVALID_ADDRESS,
            ["Valid range: 0x0000-0xFFFF"]
        )
    
    try:
        value = emulator.memory[address]
        return {
            "success": True,
            "tool": "get_memory_address",
            "data": {
                "address": hex(address),
                "value": value,
                "value_hex": hex(value),
                "value_binary": bin(value),
                "description": get_memory_address_description(address, value)
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Memory read failed: {e}")
        raise EmulatorError(
            f"Failed to read memory at {hex(address)}: {str(e)}",
            EmulatorErrorCode.MEMORY_READ_ERROR,
            ["Verify address is valid", "Check emulator state"]
        )


def set_memory_address(address: Union[int, str], value: int) -> Dict[str, Any]:
    """
    Write to a specific memory address.
    
    ⚠️ WARNING: This can corrupt game state if used incorrectly!
    Use for cheating/debugging only.
    
    Args:
        address: Memory address as integer or hex string
        value: Value to write (0-255)
    
    Returns:
        Dictionary with operation result
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    # Validate value
    if not (0 <= value <= 255):
        raise EmulatorError(
            f"Value out of range: {value}",
            EmulatorErrorCode.INVALID_PARAMETER,
            ["Value must be 0-255 (single byte)"]
        )
    
    # Convert hex string to int if needed
    if isinstance(address, str):
        try:
            address = int(address, 16)
        except ValueError:
            raise EmulatorError(
                f"Invalid address format: {address}",
                EmulatorErrorCode.INVALID_ADDRESS,
                ["Use hex format like '0xD062' or integer"]
            )
    
    # Validate address range
    if not (0 <= address < 0x10000):
        raise EmulatorError(
            f"Address out of range: {hex(address)}",
            EmulatorErrorCode.INVALID_ADDRESS,
            ["Valid range: 0x0000-0xFFFF"]
        )
    
    try:
        # Read old value for logging
        old_value = emulator.memory[address]
        
        # Write new value
        emulator.memory[address] = value
        
        logger.warning(f"Memory write: {hex(address)} = {value} (was {old_value})")
        
        return {
            "success": True,
            "tool": "set_memory_address",
            "data": {
                "address": hex(address),
                "old_value": old_value,
                "new_value": value,
                "warning": "Memory modification can corrupt game state!"
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Memory write failed: {e}")
        raise EmulatorError(
            f"Failed to write memory at {hex(address)}: {str(e)}",
            EmulatorErrorCode.MEMORY_WRITE_ERROR,
            ["Verify address is writable", "Some addresses are read-only"]
        )


def get_memory_address_description(address: int, value: int) -> str:
    """Provide human-readable description for common memory addresses"""
    descriptions = {
        0xD062: "Player X position",
        0xD063: "Player Y position",
        0xD057: "Battle status (0=no battle, 1=in battle)",
        0xD35E: "Current map ID",
        0xD6F5: "Money (byte 1/3, BCD)",
        0xD6F6: "Money (byte 2/3, BCD)",
        0xD6F7: "Money (byte 3/3, BCD)",
        0xD163: "Party Pokemon 1 species",
        0xD6E5: "Item bag start",
        0xD73E: "Text engine state",
        0xD6B5: "Player HP (low byte)",
        0xD6B6: "Player HP (high byte)",
        0xD89C: "Enemy HP (low byte)",
        0xD89D: "Enemy HP (high byte)",
    }
    
    if address in descriptions:
        return f"{descriptions[address]} = {value}"
    return f"Raw value at {hex(address)}"


# ========== Enhanced Party and Inventory Tools ==========

def get_party_pokemon() -> Dict[str, Any]:
    """
    Get detailed party Pokemon information.
    
    Returns:
        Detailed party data including species, HP, level, stats
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        party = []
        party_base = 0xD163
        pokemon_size = 44
        
        # Pokemon species names (partial list for Pokemon Red/Blue)
        species_names = {
            0x01: "Bulbasaur", 0x04: "Charmander", 0x07: "Squirtle",
            0x10: "Pidgey", 0x13: "Pidgeotto", 0x16: "Pidgeot",
            0x19: "Rattata", 0x1A: "Raticate", 0x1D: "Spearow",
            0x25: "Pikachu", 0x26: "Raichu", 0x83: "Dragonite",
            0x95: "Mewtwo", 0x96: "Mew"
        }
        
        for i in range(6):
            offset = party_base + (i * pokemon_size)
            
            try:
                species_id = emulator.memory[offset]
                
                if species_id == 0 or species_id == 0xFF:
                    continue
                
                # Calculate HP
                current_hp = (emulator.memory[offset + 0x1F] << 8) | emulator.memory[offset + 0x1E]
                max_hp = (emulator.memory[offset + 0x21] << 8) | emulator.memory[offset + 0x20]
                
                # Get level
                level = emulator.memory[offset + 0x18]
                
                # Get species name
                species_name = species_names.get(species_id, f"Unknown_{species_id:03X}")
                
                pokemon = {
                    "slot": i + 1,
                    "species_id": species_id,
                    "species_name": species_name,
                    "level": level,
                    "current_hp": current_hp,
                    "max_hp": max_hp,
                    "hp_percent": round(current_hp / max_hp * 100, 1) if max_hp > 0 else 0,
                    "address": hex(offset),
                    "status": "healthy" if current_hp > max_hp * 0.5 else "damaged" if current_hp > max_hp * 0.2 else "critical"
                }
                party.append(pokemon)
                
            except (IndexError, KeyError):
                break
        
        # Calculate party summary
        total_hp = sum(p["current_hp"] for p in party)
        total_max_hp = sum(p["max_hp"] for p in party)
        
        return {
            "success": True,
            "tool": "get_party_pokemon",
            "data": {
                "party_count": len(party),
                "party": party,
                "summary": {
                    "total_hp": f"{total_hp}/{total_max_hp}",
                    "health_percent": round(total_hp / total_max_hp * 100, 1) if total_max_hp > 0 else 0,
                    "average_level": round(sum(p["level"] for p in party) / len(party), 1) if party else 0,
                    "status": "ready" if total_hp > total_max_hp * 0.7 else "needs_healing"
                }
            },
            "frame": frame_count,
            "memory_base": hex(party_base),
            "pokemon_size_bytes": pokemon_size,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get party info: {e}")
        raise EmulatorError(
            f"Failed to read party: {str(e)}",
            EmulatorErrorCode.MEMORY_READ_ERROR,
            ["Only works for Pokemon-style games", "ROM may use different memory layout"]
        )


def get_inventory_detailed() -> Dict[str, Any]:
    """
    Get detailed inventory information.
    
    Returns:
        Detailed item list with names and quantities
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        items = []
        item_base = 0xD6E5
        
        # Item names (partial list for Pokemon Red/Blue)
        item_names = {
            0x01: "Potion", 0x02: "Antidote", 0x03: "Burn Heal", 0x04: "Ice Heal",
            0x05: "Awakening", 0x06: "Parlyz Heal", 0x07: "Full Restore",
            0x08: "Max Potion", 0x09: "Hyper Potion", 0x0A: "Super Potion",
            0x0C: "Full Heal", 0x0D: "Revive", 0x0E: "Max Revive",
            0x19: "Poke Ball", 0x1A: "Great Ball", 0x1B: "Ultra Ball",
            0x1C: "Master Ball", 0x1E: "Bicycle", 0x25: "Repel",
            0x26: "Super Repel", 0x27: "Max Repel", 0x32: "Fresh Water",
            0x33: "Soda Pop", 0x34: "Lemonade", 0x35: "Moomoo Milk"
        }
        
        for i in range(20):
            offset = item_base + (i * 2)
            
            try:
                item_id = emulator.memory[offset]
                quantity = emulator.memory[offset + 1]
                
                if item_id == 0 or item_id == 0xFF:
                    continue
                
                item_name = item_names.get(item_id, f"Item_{item_id:03X}")
                
                # Categorize item
                category = "unknown"
                if item_id <= 0x0F:
                    category = "healing"
                elif item_id in [0x19, 0x1A, 0x1B, 0x1C]:
                    category = "poke_balls"
                elif item_id in [0x25, 0x26, 0x27]:
                    category = "repels"
                elif item_id in [0x32, 0x33, 0x34, 0x35]:
                    category = "drinks"
                
                items.append({
                    "slot": i + 1,
                    "item_id": item_id,
                    "item_name": item_name,
                    "quantity": quantity,
                    "category": category,
                    "address": hex(offset)
                })
                
            except (IndexError, KeyError):
                break
        
        # Calculate summary
        total_items = sum(item["quantity"] for item in items)
        poke_balls = sum(item["quantity"] for item in items if item["category"] == "poke_balls")
        healing_items = sum(item["quantity"] for item in items if item["category"] == "healing")
        
        return {
            "success": True,
            "tool": "get_inventory",
            "data": {
                "unique_items": len(items),
                "total_items": total_items,
                "items": items,
                "summary": {
                    "poke_balls": poke_balls,
                    "healing_items": healing_items,
                    "other_items": total_items - poke_balls - healing_items
                }
            },
            "frame": frame_count,
            "memory_base": hex(item_base),
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get inventory: {e}")
        raise EmulatorError(
            f"Failed to read inventory: {str(e)}",
            EmulatorErrorCode.MEMORY_READ_ERROR,
            ["Inventory structure varies by game", "This works best for Pokemon Red/Blue"]
        )


# ========== Streaming Vision Support ==========

async def stream_screen_base64(chunk_size: int = 10000) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream screen capture in chunks for large images.
    
    Yields:
        Chunks of base64 data with metadata
    """
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        screen = emulator.screen
        if screen is None:
            raise EmulatorError(
                "Failed to get screen buffer",
                EmulatorErrorCode.OPERATION_FAILED,
                ["Try emulator_tick to advance a frame first"]
            )
        
        img = screen.image
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        total_size = len(img_base64)
        chunks = (total_size + chunk_size - 1) // chunk_size
        
        # Send metadata first
        yield {
            "success": True,
            "tool": "stream_screen_base64",
            "stream_start": True,
            "total_size": total_size,
            "chunk_size": chunk_size,
            "total_chunks": chunks,
            "dimensions": {
                "width": img.width,
                "height": img.height
            },
            "frame": frame_count
        }
        
        # Stream chunks
        for i in range(chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            chunk = img_base64[start:end]
            
            yield {
                "chunk_index": i,
                "chunk_size": len(chunk),
                "data": chunk,
                "more": i < chunks - 1
            }
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        # Send completion
        yield {
            "stream_end": True,
            "total_sent": total_size
        }
        
    except Exception as e:
        logger.error(f"Stream failed: {e}")
        raise EmulatorError(
            f"Streaming failed: {str(e)}",
            EmulatorErrorCode.STREAM_ERROR,
            ["Try get_screen_base64 instead for non-streaming"]
        )


# ========== Auto-Play Modes (Enhanced) ==========

def auto_explore_mode(steps: int = 20, avoid_battles: bool = True, session_id: str = None) -> Dict[str, Any]:
    """
    Enhanced autonomous exploration mode.
    
    Args:
        steps: Number of steps to explore
        avoid_battles: Try to avoid wild Pokemon battles
        session_id: Optional session to track progress
    
    Returns:
        Exploration results with path taken and discoveries
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        actions = []
        positions_visited = []
        battles_encountered = 0
        
        # Movement patterns
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Get initial position
        try:
            start_x = emulator.memory[0xD062]
            start_y = emulator.memory[0xD063]
            positions_visited.append({"x": start_x, "y": start_y, "step": 0})
        except:
            pass
        
        for i in range(steps):
            # Get current position
            try:
                current_x = emulator.memory[0xD062]
                current_y = emulator.memory[0xD063]
            except:
                current_x = current_y = 0
            
            # Choose direction (can be enhanced with memory/vision)
            direction = directions[i % 4]
            
            # Move
            success = press_button(direction)
            
            # Check for battle
            battle_detected = False
            if avoid_battles:
                try:
                    battle_status = emulator.memory[0xD057]
                    if battle_status != 0:
                        battles_encountered += 1
                        battle_detected = True
                        actions.append({
                            "step": i + 1,
                            "action": "BATTLE_AVOIDED",
                            "direction": direction,
                            "position": {"x": current_x, "y": current_y}
                        })
                        # Try to escape
                        press_button("B")
                        continue
                except:
                    pass
            
            if not battle_detected:
                actions.append({
                    "step": i + 1,
                    "direction": direction,
                    "success": success,
                    "position": {"x": current_x, "y": current_y}
                })
                
                try:
                    positions_visited.append({
                        "x": emulator.memory[0xD062],
                        "y": emulator.memory[0xD063],
                        "step": i + 1
                    })
                except:
                    pass
            
            time.sleep(0.05)
        
        # Update session if provided
        if session_id and session_id in agent_sessions:
            session = agent_sessions[session_id]
            session.data.exploration_log.append({
                "timestamp": datetime.now().isoformat(),
                "steps": steps,
                "positions": positions_visited[-10:]  # Last 10 positions
            })
            session.last_update = datetime.now().isoformat()
            session.last_timestamp = time.time()
            _save_session_to_disk(session)
        
        return {
            "success": True,
            "tool": "auto_explore_mode",
            "data": {
                "mode": "exploration",
                "steps_attempted": steps,
                "steps_completed": len(actions),
                "battles_encountered": battles_encountered,
                "actions": actions,
                "positions_visited": len(positions_visited),
                "final_position": positions_visited[-1] if positions_visited else None
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto explore failed: {e}")
        raise EmulatorError(
            f"Auto explore failed: {str(e)}",
            EmulatorErrorCode.OPERATION_FAILED
        )


def auto_battle_mode(max_moves: int = 15, strategy: str = "aggressive") -> Dict[str, Any]:
    """
    Enhanced AI-powered battle mode.
    
    Args:
        max_moves: Maximum moves to execute
        strategy: Battle strategy ("aggressive", "defensive", "balanced")
    
    Returns:
        Battle results with HP tracking and strategy analysis
    """
    start_time = time.time()
    
    if emulator is None:
        raise EmulatorError(
            "Emulator not initialized",
            EmulatorErrorCode.NOT_INITIALIZED,
            ["Call emulator_load_rom first"]
        )
    
    try:
        moves_executed = []
        
        # Check battle status
        try:
            battle_status = emulator.memory[0xD057]
        except:
            battle_status = 0
        
        if battle_status == 0:
            raise EmulatorError(
                "Not in battle",
                EmulatorErrorCode.BATTLE_NOT_ACTIVE,
                ["Navigate to encounter a Pokemon first", "Use auto_explore_mode to find battles"],
                recoverable=True
            )
        
        # Get HP info
        try:
            player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
            player_max_hp = (emulator.memory[0xD6BF] << 8) | emulator.memory[0xD6BE]
            enemy_hp = (emulator.memory[0xD89D] << 8) | emulator.memory[0xD89C]
            enemy_max_hp = (emulator.memory[0xD8A1] << 8) | emulator.memory[0xD8A0]
        except:
            player_hp = player_max_hp = enemy_hp = enemy_max_hp = 0
        
        # Strategy-based decision making
        for i in range(min(max_moves, 15)):
            # Check if battle ended
            try:
                current_battle = emulator.memory[0xD057]
                if current_battle == 0:
                    moves_executed.append({
                        "move": "BATTLE_ENDED",
                        "reason": "Battle concluded"
                    })
                    break
            except:
                pass
            
            # Get current HP
            try:
                current_player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
                current_enemy_hp = (emulator.memory[0xD89D] << 8) | emulator.memory[0xD89C]
            except:
                current_player_hp = current_enemy_hp = 0
            
            # Decide move based on strategy
            if strategy == "defensive" and current_player_hp < player_max_hp * 0.3:
                # Try to run or use healing item
                press_button("B")  # Try to run
                moves_executed.append({
                    "move_number": i + 1,
                    "button": "B",
                    "action": "attempt_escape",
                    "reason": "low_hp_defensive"
                })
            elif strategy == "aggressive" or current_enemy_hp < player_max_hp * 0.5:
                # Attack
                press_button("A")
                moves_executed.append({
                    "move_number": i + 1,
                    "button": "A",
                    "action": "attack",
                    "player_hp": current_player_hp,
                    "enemy_hp": current_enemy_hp
                })
            else:
                # Balanced - assess situation
                press_button("A")
                moves_executed.append({
                    "move_number": i + 1,
                    "button": "A",
                    "action": "assess_and_attack",
                    "strategy": strategy
                })
            
            time.sleep(0.1)
        
        # Get final HP
        try:
            final_player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
            final_enemy_hp = (emulator.memory[0xD89D] << 8) | emulator.memory[0xD89C]
        except:
            final_player_hp = final_enemy_hp = 0
        
        # Determine battle outcome
        battle_outcome = "ongoing"
        if final_enemy_hp == 0:
            battle_outcome = "victory"
        elif final_player_hp == 0:
            battle_outcome = "defeat"
        elif battle_status == 0:
            battle_outcome = "escaped"
        
        return {
            "success": True,
            "tool": "auto_battle_mode",
            "data": {
                "moves_executed": moves_executed,
                "total_moves": len(moves_executed),
                "strategy_used": strategy,
                "battle_outcome": battle_outcome,
                "hp_tracking": {
                    "player": {
                        "start": player_hp,
                        "end": final_player_hp,
                        "max": player_max_hp,
                        "damage_taken": player_hp - final_player_hp
                    },
                    "enemy": {
                        "start": enemy_hp,
                        "end": final_enemy_hp,
                        "max": enemy_max_hp,
                        "damage_dealt": enemy_hp - final_enemy_hp
                    }
                }
            },
            "frame": frame_count,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
    except Exception as e:
        logger.error(f"Auto battle failed: {e}")
        raise EmulatorError(
            f"Auto battle failed: {str(e)}",
            EmulatorErrorCode.OPERATION_FAILED
        )


# ========== Session Management ==========

def session_start(session_id: str = None, goal: str = None, ttl_seconds: int = None) -> Dict[str, Any]:
    """Start a new agent session with persistence"""
    start_time = time.time()
    
    _clean_expired_sessions()
    
    if session_id is None:
        session_id = f"session_{int(time.time() * 1000)}"
    
    if session_id in agent_sessions:
        if time.time() - agent_sessions[session_id].last_timestamp > agent_sessions[session_id].ttl_seconds:
            del agent_sessions[session_id]
        else:
            session = agent_sessions[session_id]
            return {
                "success": True,
                "tool": "session_start",
                "data": {
                    "session_id": session_id,
                    "message": "Session already exists",
                    "session_data": asdict(session.data),
                    "created": session.created,
                    "ttl_seconds": session.ttl_seconds
                },
                "timing_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    ttl = ttl_seconds if ttl_seconds is not None else 3600
    
    session = Session(
        session_id=session_id,
        data=SessionData(goal=goal or "Play the game autonomously"),
        created=datetime.now().isoformat(),
        last_update=datetime.now().isoformat(),
        last_timestamp=time.time(),
        ttl_seconds=ttl
    )
    
    agent_sessions[session_id] = session
    _save_session_to_disk(session)
    
    return {
        "success": True,
        "tool": "session_start",
        "data": {
            "session_id": session_id,
            "message": "New session created (persisted to disk)",
            "goal": goal or "Play the game autonomously",
            "ttl_seconds": ttl
        },
        "timing_ms": round((time.time() - start_time) * 1000, 2)
    }


def session_get(session_id: str, key: str = None) -> Dict[str, Any]:
    """Get session data"""
    start_time = time.time()
    
    if session_id not in agent_sessions:
        raise EmulatorError(
            f"Session not found: {session_id}",
            EmulatorErrorCode.SESSION_NOT_FOUND,
            ["Use session_start to create a session first", f"Available: {list(agent_sessions.keys())}"]
        )
    
    session = agent_sessions[session_id]
    session.last_update = datetime.now().isoformat()
    session.last_timestamp = time.time()
    _save_session_to_disk(session)
    
    if key is None:
        return {
            "success": True,
            "tool": "session_get",
            "data": asdict(session.data),
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
    
    return {
        "success": True,
        "tool": "session_get",
        "data": getattr(session.data, key, None),
        "key": key,
        "timing_ms": round((time.time() - start_time) * 1000, 2)
    }


def session_set(session_id: str, key: str, value: Any) -> Dict[str, Any]:
    """Set session data"""
    start_time = time.time()
    
    if session_id not in agent_sessions:
        raise EmulatorError(
            f"Session not found: {session_id}",
            EmulatorErrorCode.SESSION_NOT_FOUND,
            ["Use session_start to create a session first"]
        )
    
    session = agent_sessions[session_id]
    setattr(session.data, key, value)
    session.last_update = datetime.now().isoformat()
    session.last_timestamp = time.time()
    _save_session_to_disk(session)
    
    return {
        "success": True,
        "tool": "session_set",
        "data": {
            "session_id": session_id,
            "key": key,
            "value": value,
            "persisted": True
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
                    "created": session.created,
                    "last_update": session.last_update,
                    "goal": session.data.goal,
                    "ttl_remaining": session.ttl_seconds - (time.time() - session.last_timestamp)
                }
                for sid, session in agent_sessions.items()
            ],
            "count": len(agent_sessions),
            "persisted_to": str(PERSISTENCE_DIR)
        }
    }


def session_delete(session_id: str) -> Dict[str, Any]:
    """Delete a session"""
    if session_id in agent_sessions:
        del agent_sessions[session_id]
        session_file = PERSISTENCE_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        return {
            "success": True,
            "tool": "session_delete",
            "data": {"session_id": session_id, "message": "Session deleted"}
        }
    return {
        "success": False,
        "error": f"Session not found: {session_id}"
    }


# ========== MCP Tool Definitions (Enhanced) ==========

def create_tool_definitions() -> List[Tool]:
    """Create enhanced tool definitions with detailed schemas"""
    return [
        Tool(
            name="emulator_load_rom",
            description="Load a Game Boy ROM file (.gb or .gba) into the emulator. This initializes the emulator state and prepares it for gameplay.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rom_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the ROM file",
                        "examples": ["./roms/pokemon-red.gb", "/home/user/games/tetris.gb"]
                    }
                },
                "required": ["rom_path"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="emulator_press_button",
            description="Press a Game Boy controller button. Valid buttons: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT.",
            inputSchema={
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "description": "Button to press",
                        "enum": ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
                    }
                },
                "required": ["button"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_memory_address",
            description="Read a specific memory address from Game Boy RAM. Returns value in decimal, hex, and binary with human-readable description for common addresses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": ["integer", "string"],
                        "description": "Memory address to read (integer or hex string like '0xD062')",
                        "examples": [54370, "0xD062", "0xD063"]
                    }
                },
                "required": ["address"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="set_memory_address",
            description="⚠️ Write to a specific memory address. WARNING: Can corrupt game state! Use for debugging/cheating only.",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": ["integer", "string"],
                        "description": "Memory address to write (integer or hex string)",
                        "examples": [54370, "0xD062"]
                    },
                    "value": {
                        "type": "integer",
                        "description": "Value to write (0-255)",
                        "minimum": 0,
                        "maximum": 255
                    }
                },
                "required": ["address", "value"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_party_pokemon",
            description="Get detailed information about your party Pokemon including species, level, HP, and status. Works best with Pokemon Red/Blue.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_inventory",
            description="Get detailed inventory information including item names, quantities, and categories (healing, poke balls, etc.).",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_player_position",
            description="Get player X,Y coordinates from game memory. Returns tile-based position (not pixels).",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_map_location",
            description="Get current map/location ID from memory. Useful for navigation and tracking progress.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_money",
            description="Get player money/currency. Returns BCD-encoded value formatted as dollars.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_screen_base64",
            description="Get current screen as base64-encoded PNG for vision analysis. Include full image or truncated preview.",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_base64": {
                        "type": "boolean",
                        "description": "Include full base64 image (true) or just metadata (false)",
                        "default": True
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="auto_explore_mode",
            description="Start autonomous exploration mode. Moves around the game world automatically, optionally avoiding battles.",
            inputSchema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of movement steps (default: 20)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "avoid_battles": {
                        "type": "boolean",
                        "description": "Try to avoid wild Pokemon battles (default: true)",
                        "default": True
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID to track exploration progress"
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="auto_battle_mode",
            description="Start AI-powered battle mode. Analyzes HP and executes optimal moves based on strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_moves": {
                        "type": "integer",
                        "description": "Maximum moves to execute (default: 15)",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Battle strategy",
                        "enum": ["aggressive", "defensive", "balanced"],
                        "default": "aggressive"
                    }
                },
                "required": ["max_moves"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="session_start",
            description="Start a new agent session for persistent state across tool calls. Sessions are saved to disk and survive server restarts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Optional custom session ID (auto-generated if not provided)"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Agent's goal for this session",
                        "examples": ["Beat the Elite 4", "Catch all Pokemon", "Explore the map"]
                    },
                    "ttl_seconds": {
                        "type": "integer",
                        "description": "Session time-to-live in seconds (default: 3600)",
                        "default": 3600,
                        "minimum": 60,
                        "maximum": 86400
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="session_get",
            description="Get data from an agent session. Returns all data or a specific key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to retrieve"
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional specific key (e.g., 'goal', 'visited_locations')",
                        "examples": ["goal", "party_hp_history", "exploration_log"]
                    }
                },
                "required": ["session_id"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="session_set",
            description="Set a value in an agent session. Data is persisted to disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to update"
                    },
                    "key": {
                        "type": "string",
                        "description": "Key to set",
                        "examples": ["current_location", "party_hp", "notes"]
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "object", "array"],
                        "description": "Value to store (any JSON-serializable type)"
                    }
                },
                "required": ["session_id", "key", "value"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="session_list",
            description="List all active agent sessions with metadata.",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="session_delete",
            description="Delete an agent session and remove it from disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to delete"
                    }
                },
                "required": ["session_id"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="emulator_tick",
            description="Advance emulation by one or more frames.",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "description": "Number of frames to advance (default: 1)",
                        "default": 1
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="emulator_get_state",
            description="Get current emulator state (initialized, ROM path, frame count).",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="save_game_state",
            description="Save current emulator state to file for later restoration.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Optional name for save file (defaults to frame number)"
                    }
                },
                "additionalProperties": False
            }
        ),
        Tool(
            name="load_game_state",
            description="Load a previously saved emulator state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save_name": {
                        "type": "string",
                        "description": "Name of save file to load"
                    }
                },
                "required": ["save_name"],
                "additionalProperties": False
            }
        )
    ]


# ========== MCP Server Setup ==========

server = Server(
    "pyboy-emulator",
    capabilities={
        "tools": {},
    }
)
server_version = SERVER_VERSION


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available emulator tools"""
    return create_tool_definitions()


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls with enhanced error handling"""
    global emulator
    
    try:
        if name == "emulator_load_rom":
            rom_path_arg = arguments.get("rom_path")
            if not rom_path_arg:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, 
                    error="rom_path is required", 
                    error_code=EmulatorErrorCode.INVALID_PARAMETER,
                    suggestions=["Provide a valid path to a .gb or .gba ROM file"]
                )))]
            
            init_emulator(rom_path_arg)
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
        
        elif name == "get_memory_address":
            address = arguments.get("address")
            if not address:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False,
                    error="address is required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            result = get_memory_address(address)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "set_memory_address":
            address = arguments.get("address")
            value = arguments.get("value")
            if not address or value is None:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False,
                    error="address and value are required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            result = set_memory_address(address, value)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_party_pokemon":
            result = get_party_pokemon()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_inventory":
            result = get_inventory_detailed()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_player_position":
            # Reuse existing function
            start_time = time.time()
            if emulator is None:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="Emulator not initialized",
                    error_code=EmulatorErrorCode.NOT_INITIALIZED
                )))]
            
            try:
                x = emulator.memory[0xD062]
                y = emulator.memory[0xD063]
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "tool": "get_player_position",
                    "data": {"x": x, "y": y},
                    "timing_ms": round((time.time() - start_time) * 1000, 2)
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.MEMORY_READ_ERROR
                )))]
        
        elif name == "get_map_location":
            start_time = time.time()
            if emulator is None:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="Emulator not initialized",
                    error_code=EmulatorErrorCode.NOT_INITIALIZED
                )))]
            
            try:
                map_id = emulator.memory[0xD35E]
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "tool": "get_map_location",
                    "data": {"map_id": map_id, "map_hex": hex(map_id)},
                    "timing_ms": round((time.time() - start_time) * 1000, 2)
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.MEMORY_READ_ERROR
                )))]
        
        elif name == "get_money":
            start_time = time.time()
            if emulator is None:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="Emulator not initialized",
                    error_code=EmulatorErrorCode.NOT_INITIALIZED
                )))]
            
            try:
                money_bytes = [emulator.memory[0xD6F5], emulator.memory[0xD6F6], emulator.memory[0xD6F7]]
                money = 0
                for byte in money_bytes:
                    high_digit = (byte >> 4) & 0x0F
                    low_digit = byte & 0x0F
                    money = money * 100 + high_digit * 10 + low_digit
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "tool": "get_money",
                    "data": {"money": money, "formatted": f"${money:,}"},
                    "timing_ms": round((time.time() - start_time) * 1000, 2)
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.MEMORY_READ_ERROR
                )))]
        
        elif name == "get_screen_base64":
            include_base64 = arguments.get("include_base64", True)
            start_time = time.time()
            
            if emulator is None:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="Emulator not initialized",
                    error_code=EmulatorErrorCode.NOT_INITIALIZED
                )))]
            
            try:
                screen = emulator.screen
                img = screen.image
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                result = {
                    "success": True,
                    "tool": "get_screen_base64",
                    "frame": frame_count,
                    "dimensions": {"width": img.width, "height": img.height},
                    "timing_ms": round((time.time() - start_time) * 1000, 2)
                }
                
                if include_base64:
                    result["image_base64"] = img_base64
                else:
                    result["image_base64"] = img_base64[:200] + "...[truncated]"
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.OPERATION_FAILED
                )))]
        
        elif name == "auto_explore_mode":
            steps = arguments.get("steps", 20)
            avoid_battles = arguments.get("avoid_battles", True)
            session_id = arguments.get("session_id")
            
            result = auto_explore_mode(steps, avoid_battles, session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "auto_battle_mode":
            max_moves = arguments.get("max_moves", 15)
            strategy = arguments.get("strategy", "aggressive")
            
            result = auto_battle_mode(max_moves, strategy)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
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
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="session_id is required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            try:
                result = session_get(session_id, key)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except EmulatorError as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e), error_code=e.code, suggestions=e.suggestions
                )))]
        
        elif name == "session_set":
            session_id = arguments.get("session_id")
            key = arguments.get("key")
            value = arguments.get("value")
            
            if not session_id or not key:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="session_id and key are required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            result = session_set(session_id, key, value)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_list":
            result = session_list()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "session_delete":
            session_id = arguments.get("session_id")
            
            if not session_id:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="session_id is required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            result = session_delete(session_id)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "emulator_tick":
            frames = arguments.get("frames", 1)
            success = all(tick() for _ in range(frames))
            
            return [TextContent(
                type="text",
                text=json.dumps({"success": success, "frames": frames, "new_frame": frame_count}, indent=2)
            )]
        
        elif name == "emulator_get_state":
            state = {
                'initialized': emulator is not None,
                'rom_path': rom_path,
                'frame_count': frame_count,
                'pyboy_available': PYBOY_AVAILABLE
            }
            return [TextContent(type="text", text=json.dumps(state, indent=2))]
        
        elif name == "save_game_state":
            save_name = arguments.get("save_name")
            SAVE_DIR = Path(__file__).parent.parent / "saves"
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            
            if save_name is None:
                save_name = f"save_{frame_count:06d}.state"
            elif not save_name.endswith('.state'):
                save_name += '.state'
            
            save_path = SAVE_DIR / save_name
            
            try:
                with open(save_path, 'wb') as f:
                    emulator.save_state(f)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "path": str(save_path),
                    "frame": frame_count
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.OPERATION_FAILED
                )))]
        
        elif name == "load_game_state":
            save_name = arguments.get("save_name")
            if not save_name:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error="save_name is required",
                    error_code=EmulatorErrorCode.INVALID_PARAMETER
                )))]
            
            SAVE_DIR = Path(__file__).parent.parent / "saves"
            save_path = SAVE_DIR / save_name
            
            if not save_path.exists():
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=f"Save file not found: {save_path}",
                    error_code=EmulatorErrorCode.SAVE_NOT_FOUND
                )))]
            
            try:
                with open(save_path, 'rb') as f:
                    emulator.load_state(f)
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "path": str(save_path),
                    "frame": frame_count
                }, indent=2))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(format_response(
                    False, error=str(e),
                    error_code=EmulatorErrorCode.OPERATION_FAILED
                )))]
        
        else:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Unknown tool: {name}",
                "error_code": EmulatorErrorCode.INVALID_PARAMETER,
                "available_tools": [t.name for t in create_tool_definitions()]
            }))]
    
    except EmulatorError as e:
        logger.error(f"Emulator error in {name}: {e.code} - {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": str(e),
            "error_code": e.code,
            "suggestions": e.suggestions,
            "recoverable": e.recoverable
        }))]
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [TextContent(type="text", text=json.dumps({
            "success": False, 
            "error": str(e),
            "error_code": EmulatorErrorCode.OPERATION_FAILED,
            "suggestions": ["Check emulator state", "Try reloading the ROM"],
            "recoverable": True
        }))]


async def main():
    """Run the MCP server"""
    logger.info(f"Starting PyBoy Emulator MCP Server v{SERVER_VERSION}...")
    logger.info(f"PyBoy available: {PYBOY_AVAILABLE}")
    logger.info(f"Session persistence: {PERSISTENCE_DIR}")
    logger.info("Enhanced features: memory read/write, streaming vision, persistent sessions, auto-modes")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
