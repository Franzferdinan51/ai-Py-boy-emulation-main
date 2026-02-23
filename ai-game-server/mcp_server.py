#!/usr/bin/env python3
"""
OpenClaw MCP Server for AI Py-Boy Emulation
Exposes emulator controls as MCP tools for OpenClaw agents

Tools:
- emulator_load_rom: Load a ROM file
- emulator_press_button: Press a controller button
- emulator_get_frame: Get current screen as image
- emulator_get_state: Get emulator state (RAM, inventory, etc.)
- emulator_tick: Advance emulation by one frame
"""

import os
import sys
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️  PyBoy not installed - emulator tools will be unavailable")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP library not installed - install with: pip install mcp")
    sys.exit(1)

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emulator-mcp")

# Initialize MCP server
server = Server("pyboy-emulator")

# Global emulator state
emulator: Optional[PyBoy] = None
rom_path: Optional[str] = None
frame_count: int = 0


def init_emulator(rom_file: str) -> bool:
    """Initialize PyBoy emulator with ROM"""
    global emulator, rom_path, frame_count
    
    if not PYBOY_AVAILABLE:
        logger.error("PyBoy not available")
        return False
    
    try:
        emulator = PyBoy(rom_file, window="null")
        rom_path = rom_file
        frame_count = 0
        logger.info(f"Loaded ROM: {rom_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to load ROM: {e}")
        return False


def press_button(button: str) -> bool:
    """Press a controller button"""
    global emulator, frame_count
    
    if emulator is None:
        logger.error("Emulator not initialized")
        return False
    
    button_map = {
        'A': WindowEvent.PRESS_BUTTON_A,
        'B': WindowEvent.PRESS_BUTTON_B,
        'UP': WindowEvent.PRESS_ARROW_UP,
        'DOWN': WindowEvent.PRESS_ARROW_DOWN,
        'LEFT': WindowEvent.PRESS_ARROW_LEFT,
        'RIGHT': WindowEvent.PRESS_ARROW_RIGHT,
        'START': WindowEvent.PRESS_BUTTON_START,
        'SELECT': WindowEvent.PRESS_BUTTON_SELECT,
    }
    
    release_map = {
        'A': WindowEvent.RELEASE_BUTTON_A,
        'B': WindowEvent.RELEASE_BUTTON_B,
        'UP': WindowEvent.RELEASE_ARROW_UP,
        'DOWN': WindowEvent.RELEASE_ARROW_DOWN,
        'LEFT': WindowEvent.RELEASE_ARROW_LEFT,
        'RIGHT': WindowEvent.RELEASE_ARROW_RIGHT,
        'START': WindowEvent.RELEASE_BUTTON_START,
        'SELECT': WindowEvent.RELEASE_BUTTON_SELECT,
    }
    
    try:
        button_upper = button.upper()
        if button_upper not in button_map:
            logger.error(f"Unknown button: {button}")
            return False
        
        # Press and release
        emulator.send_input(button_map[button_upper])
        emulator.tick()
        emulator.send_input(release_map[button_upper])
        frame_count += 1
        logger.info(f"Pressed button: {button}")
        return True
    except Exception as e:
        logger.error(f"Button press failed: {e}")
        return False


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
        logger.error("Emulator not initialized")
        return False
    
    try:
        emulator.tick()
        frame_count += 1
        return True
    except Exception as e:
        logger.error(f"Tick failed: {e}")
        return False


# MCP Tool Handlers

@server.list_tools()
async def list_tools() -> list[Tool]:
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
            name="emulator_get_frame",
            description="Get the current screen as a base64-encoded PNG image",
            inputSchema={
                "type": "object",
                "properties": {}
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
            description="Advance the emulation by one frame",
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
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    global emulator
    
    try:
        if name == "emulator_load_rom":
            rom_path = arguments.get("rom_path")
            if not rom_path:
                return [TextContent(type="text", text="Error: rom_path required")]
            
            success = init_emulator(rom_path)
            return [TextContent(
                type="text",
                text=json.dumps({"success": success, "rom": rom_path}, indent=2)
            )]
        
        elif name == "emulator_press_button":
            button = arguments.get("button")
            if not button:
                return [TextContent(type="text", text="Error: button required")]
            
            success = press_button(button)
            return [TextContent(
                type="text",
                text=json.dumps({"success": success, "button": button, "frame": frame_count}, indent=2)
            )]
        
        elif name == "emulator_get_frame":
            frame_data = get_frame()
            if frame_data is None:
                return [TextContent(type="text", text="Error: Emulator not initialized or failed to capture frame")]
            
            # Return image data
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "frame": frame_data['frame'],
                    "dimensions": f"{frame_data['width']}x{frame_data['height']}",
                    "image_base64": frame_data['base64'][:100] + "...[truncated]"  # Truncate for readability
                }, indent=2)
            )]
        
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
        
        else:
            return [TextContent(type="text", text=f"Error: Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server"""
    logger.info("Starting PyBoy Emulator MCP Server...")
    logger.info(f"PyBoy available: {PYBOY_AVAILABLE}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
