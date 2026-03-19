#!/usr/bin/env python3
"""
Generic Game Boy MCP Server
Works with ANY Game Boy, Game Boy Color, or Game Boy Advance game.
Not limited to Pokemon!

Based on PyBoy emulation with generic controls and memory access.
"""

import os
import sys
import json
import base64
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from io import BytesIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PIL import Image
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gb-mcp")

SERVER_VERSION = "1.0.0"

# Try to import PyBoy and MCP
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    logger.warning("PyBoy not installed - install with: pip install pyboy")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP library not installed - install with: pip install mcp")
    sys.exit(1)

# Create MCP server
server = Server("gameboy-mcp")

# Global emulator state
class EmulatorState:
    def __init__(self):
        self.pyboy: Optional[PyBoy] = None
        self.rom_path: Optional[str] = None
        self.rom_type: Optional[str] = None
        self.save_dir = Path.home() / ".gb_mcp_saves"
        self.save_dir.mkdir(exist_ok=True)
        
    def reset(self):
        if self.pyboy:
            self.pyboy.stop()
        self.pyboy = None
        self.rom_path = None
        self.rom_type = None

state = EmulatorState()

# =============================================================================
# Generic MCP Tools (work with ANY game)
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools"""
    return [
        Tool(
            name="load_rom",
            description="Load a Game Boy ROM file (.gb, .gbc, .gba). Supports any Game Boy game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "rom_path": {
                        "type": "string",
                        "description": "Path to the ROM file",
                        "examples": ["/path/to/super-mario.gb", "./pokemon-red.gbc"]
                    },
                    "game_type": {
                        "type": "string", 
                        "description": "Game type hint (auto-detected if not provided)",
                        "enum": ["gb", "gbc", "gba", "auto"],
                        "default": "auto"
                    }
                },
                "required": ["rom_path"]
            }
        ),
        Tool(
            name="press_button",
            description="Press a Game Boy button. Works with ANY game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "description": "Button to press",
                        "enum": ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]
                    },
                    "duration_frames": {
                        "type": "integer",
                        "description": "How many frames to hold the button (1 frame = ~16ms)",
                        "default": 6,
                        "minimum": 1,
                        "maximum": 60
                    }
                },
                "required": ["button"]
            }
        ),
        Tool(
            name="get_screen",
            description="Get the current game screen as an image. Works with any game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Image format",
                        "enum": ["png", "jpeg", "base64"],
                        "default": "base64"
                    }
                }
            }
        ),
        Tool(
            name="tick",
            description="Advance the emulator by N frames without pressing any button",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "description": "Number of frames to advance",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 1000
                    }
                }
            }
        ),
        Tool(
            name="get_memory",
            description="Read a value from game memory at a specific address. Works with any game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "Memory address (hex or decimal)",
                        "minimum": 0,
                        "maximum": 65535
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of bytes to read",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 256
                    }
                },
                "required": ["address"]
            }
        ),
        Tool(
            name="set_memory",
            description="Write a value to game memory. USE WITH CAUTION - can corrupt game state!",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "Memory address to write to",
                        "minimum": 0,
                        "maximum": 65535
                    },
                    "value": {
                        "type": "integer",
                        "description": "Byte value to write (0-255)"
                    }
                },
                "required": ["address", "value"]
            }
        ),
        Tool(
            name="get_state",
            description="Get current emulator state information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="save_state",
            description="Save current game state to a file. Works with any game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for this save state",
                        "default": "quick_save"
                    }
                }
            }
        ),
        Tool(
            name="load_state",
            description="Load a previously saved game state",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the save state to load",
                        "default": "quick_save"
                    }
                }
            }
        ),
        Tool(
            name="list_saves",
            description="List all available save states",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="reset",
            description="Reset the emulator (reload the current ROM)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_game_info",
            description="Get information about the currently loaded game",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="press_buttons",
            description="Press multiple buttons in sequence (for complex inputs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "buttons": {
                        "type": "array",
                        "description": "List of buttons to press in order",
                        "items": {
                            "type": "string",
                            "enum": ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]
                        }
                    },
                    "frame_delay": {
                        "type": "integer",
                        "description": "Frames to wait between button presses",
                        "default": 10
                    }
                },
                "required": ["buttons"]
            }
        ),
    ]

# =============================================================================
# Tool Implementations
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
    """Handle tool calls"""
    if not PYBOY_AVAILABLE:
        return [TextContent(type="text", text=json.dumps({"error": "PyBoy not installed"}))]
    
    if not arguments:
        arguments = {}
    
    try:
        if name == "load_rom":
            return await handle_load_rom(arguments)
        elif name == "press_button":
            return await handle_press_button(arguments)
        elif name == "get_screen":
            return await handle_get_screen(arguments)
        elif name == "tick":
            return await handle_tick(arguments)
        elif name == "get_memory":
            return await handle_get_memory(arguments)
        elif name == "set_memory":
            return await handle_set_memory(arguments)
        elif name == "get_state":
            return await handle_get_state(arguments)
        elif name == "save_state":
            return await handle_save_state(arguments)
        elif name == "load_state":
            return await handle_load_state(arguments)
        elif name == "list_saves":
            return await handle_list_saves(arguments)
        elif name == "reset":
            return await handle_reset(arguments)
        elif name == "get_game_info":
            return await handle_get_game_info(arguments)
        elif name == "press_buttons":
            return await handle_press_buttons(arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_load_rom(args: Dict[str, Any]) -> List[TextContent]:
    """Load a ROM file"""
    rom_path = args.get("rom_path")
    if not rom_path:
        return [TextContent(type="text", text=json.dumps({"error": "rom_path required"}))]
    
    # Resolve path
    if not os.path.isabs(rom_path):
        rom_path = os.path.abspath(rom_path)
    
    if not os.path.exists(rom_path):
        return [TextContent(type="text", text=json.dumps({"error": f"ROM not found: {rom_path}"}))]
    
    # Reset existing state
    state.reset()
    
    # Determine game type
    game_type = args.get("game_type", "auto")
    if game_type == "auto":
        ext = os.path.splitext(rom_path)[1].lower()
        if ext == ".gba":
            game_type = "gba"
        elif ext == ".gbc":
            game_type = "gbc"
        else:
            game_type = "gb"
    
    # Start PyBoy
    logger.info(f"Loading ROM: {rom_path}")
    os.environ['SDL_WINDOW_HIDDEN'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'disk'
    
    try:
        state.pyboy = PyBoy(rom_path, window="SDL2", scale=2, sound_emulated=False)
        state.rom_path = rom_path
        state.rom_type = game_type
        
        # Get ROM header info for game name
        rom_name = os.path.splitext(os.path.basename(rom_path))[0]
        
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "rom_path": rom_path,
            "game_type": game_type,
            "game_name": rom_name,
            "message": f"Loaded {rom_name} ({game_type.upper()})"
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": f"Failed to load ROM: {e}"}))]

async def handle_press_button(args: Dict[str, Any]) -> List[TextContent]:
    """Press a button"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    button = args.get("button", "").upper()
    duration = args.get("duration_frames", 6)
    
    valid_buttons = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT"]
    if button not in valid_buttons:
        return [TextContent(type="text", text=json.dumps({"error": f"Invalid button: {button}"}))]
    
    try:
        # Press button for specified frames
        for _ in range(duration):
            state.pyboy.button(button)
            state.pyboy.tick(1, True)
        
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "button": button,
            "frames": duration
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_get_screen(args: Dict[str, Any]) -> List[TextContent]:
    """Get current screen"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    format_type = args.get("format", "base64")
    
    try:
        # Get screen from PyBoy
        screen = state.pyboy.screen.image
        
        if format_type == "base64":
            buffer = BytesIO()
            screen.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return [TextContent(type="text", text=json.dumps({
                "format": "base64",
                "image": img_base64,
                "size": [screen.width, screen.height]
            }))]
        else:
            buffer = BytesIO()
            screen.save(buffer, format=format_type.upper())
            return [TextContent(type="text", text=json.dumps({
                "format": format_type,
                "size": [screen.width, screen.height]
            }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_tick(args: Dict[str, Any]) -> List[TextContent]:
    """Advance frames"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    frames = args.get("frames", 1)
    frames = min(max(1, frames), 1000)
    
    try:
        for _ in range(frames):
            state.pyboy.tick(1, True)
        
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "frames_advanced": frames,
            "frame_count": state.pyboy.frame_count
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_get_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Read memory"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    address = args.get("address", 0)
    length = args.get("length", 1)
    
    try:
        values = []
        for i in range(length):
            addr = address + i
            if addr < 65536:
                values.append(state.pyboy.memory[addr])
            else:
                values.append(0)
        
        return [TextContent(type="text", text=json.dumps({
            "address": address,
            "length": length,
            "values": values,
            "hex": [f"0x{v:02X}" for v in values]
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_set_memory(args: Dict[str, Any]) -> List[TextContent]:
    """Write memory"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    address = args.get("address", 0)
    value = args.get("value", 0)
    
    try:
        state.pyboy.memory[address] = value & 0xFF
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "address": address,
            "value": value & 0xFF
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_get_state(args: Dict[str, Any]) -> List[TextContent]:
    """Get emulator state"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"loaded": False}))]
    
    return [TextContent(type="text", text=json.dumps({
        "loaded": True,
        "rom_path": state.rom_path,
        "game_type": state.rom_type,
        "frame_count": state.pyboy.frame_count,
        "version": SERVER_VERSION
    }))]

async def handle_save_state(args: Dict[str, Any]) -> List[TextContent]:
    """Save state"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    name = args.get("name", "quick_save")
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    filename = f"{safe_name}.state"
    filepath = state.save_dir / filename
    
    try:
        state.pyboy.save_state(str(filepath))
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "save_name": name,
            "path": str(filepath)
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_load_state(args: Dict[str, Any]) -> List[TextContent]:
    """Load state"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    name = args.get("name", "quick_save")
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_")
    filename = f"{safe_name}.state"
    filepath = state.save_dir / filename
    
    if not filepath.exists():
        return [TextContent(type="text", text=json.dumps({"error": f"Save not found: {name}"}))]
    
    try:
        state.pyboy.load_state(str(filepath))
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "save_name": name
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def handle_list_saves(args: Dict[str, Any]) -> List[TextContent]:
    """List save states"""
    saves = list(state.save_dir.glob("*.state"))
    return [TextContent(type="text", text=json.dumps({
        "saves": [s.stem for s in saves],
        "count": len(saves)
    }))]

async def handle_reset(args: Dict[str, Any]) -> List[TextContent]:
    """Reset emulator"""
    if state.rom_path and os.path.exists(state.rom_path):
        state.reset()
        os.environ['SDL_WINDOW_HIDDEN'] = '1'
        state.pyboy = PyBoy(state.rom_path, window="SDL2", scale=2, sound_emulated=False)
        return [TextContent(type="text", text=json.dumps({"success": True, "message": "Emulator reset"}))]
    return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]

async def handle_get_game_info(args: Dict[str, Any]) -> List[TextContent]:
    """Get game info"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    rom_name = os.path.splitext(os.path.basename(state.rom_path))[0] if state.rom_path else "Unknown"
    
    return [TextContent(type="text", text=json.dumps({
        "name": rom_name,
        "type": state.rom_type,
        "path": state.rom_path,
        "frame_count": state.pyboy.frame_count
    }))]

async def handle_press_buttons(args: Dict[str, Any]) -> List[TextContent]:
    """Press multiple buttons"""
    if not state.pyboy:
        return [TextContent(type="text", text=json.dumps({"error": "No ROM loaded"}))]
    
    buttons = args.get("buttons", [])
    delay = args.get("frame_delay", 10)
    
    if not buttons:
        return [TextContent(type="text", text=json.dumps({"error": "No buttons specified"}))]
    
    try:
        for button in buttons:
            state.pyboy.button(button.upper())
            for _ in range(6):
                state.pyboy.tick(1, True)
            for _ in range(delay):
                state.pyboy.tick(1, True)
        
        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "buttons_pressed": buttons,
            "frame_delay": delay
        }))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

# =============================================================================
# Main Server
# =============================================================================

async def main():
    """Run the MCP server"""
    logger.info(f"Starting Generic Game Boy MCP Server v{SERVER_VERSION}")
    logger.info(f"PyBoy available: {PYBOY_AVAILABLE}")
    logger.info("Supports ANY Game Boy game (Pokemon, Mario, Zelda, etc.)!")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
