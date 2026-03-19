#!/usr/bin/env python3
"""
Generic Game Boy MCP Server - Connects to existing backend API
Works with ANY Game Boy game through the web API, not its own emulator!
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

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gb-mcp-api")

SERVER_VERSION = "1.0.0"

# Default backend URL - connects to existing web UI backend
DEFAULT_BACKEND_URL = os.environ.get("GB_BACKEND_URL", "http://localhost:5002")

# Create MCP server
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.error("MCP library not installed - install with: pip install mcp")
    sys.exit(1)

server = Server("gameboy-mcp-api")

# =============================================================================
# MCP Tools (connect to existing backend)
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools"""
    return [
        Tool(
            name="load_rom",
            description="Load a ROM file via the backend API. Supports any GB/GBC/GBA game!",
            inputSchema={
                "type": "object",
                "properties": {
                    "rom_path": {
                        "type": "string",
                        "description": "Path to the ROM file"
                    }
                },
                "required": ["rom_path"]
            }
        ),
        Tool(
            name="press_a",
            description="Press the A button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_b",
            description="Press the B button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_up",
            description="Press the UP button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_down",
            description="Press the DOWN button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_left",
            description="Press the LEFT button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_right",
            description="Press the RIGHT button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_start",
            description="Press the START button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="press_select",
            description="Press the SELECT button",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_screen",
            description="Get the current game screen as an image",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="tick",
            description="Advance the emulator by N frames",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "description": "Number of frames to advance",
                        "default": 1
                    }
                }
            }
        ),
        Tool(
            name="get_memory",
            description="Read a value from game memory at a specific address",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "Memory address (hex or decimal)"
                    }
                },
                "required": ["address"]
            }
        ),
        Tool(
            name="get_state",
            description="Get current emulator state from backend",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="save_state",
            description="Save current game state",
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
                        "description": "Name of the save state to load"
                    }
                }
            }
        ),
        Tool(
            name="get_party",
            description="Get Pokemon party info (Pokemon games only)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_inventory",
            description="Get inventory info (Pokemon games only)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_position",
            description="Get player position (Pokemon games only)",
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
    ]

def api_get(endpoint: str) -> Dict[str, Any]:
    """Make GET request to backend"""
    url = f"{DEFAULT_BACKEND_URL}{endpoint}"
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def api_post(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make POST request to backend"""
    url = f"{DEFAULT_BACKEND_URL}{endpoint}"
    try:
        r = requests.post(url, json=data, timeout=10)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@server.call_tool()
async def call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> List[TextContent]:
    """Handle tool calls"""
    if not MCP_AVAILABLE:
        return [TextContent(type="text", text=json.dumps({"error": "MCP not installed"}))]
    
    if not arguments:
        arguments = {}
    
    try:
        if name == "load_rom":
            result = api_post("/api/rom/load", {"path": arguments.get("rom_path")})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "press_button":
            result = api_post("/api/game/button", {"button": arguments.get("button", "").upper()})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "press_a":
            result = api_post("/api/game/button", {"button": "A"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "A", "result": result}))]
        elif name == "press_b":
            result = api_post("/api/game/button", {"button": "B"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "B", "result": result}))]
        elif name == "press_up":
            result = api_post("/api/game/button", {"button": "UP"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "UP", "result": result}))]
        elif name == "press_down":
            result = api_post("/api/game/button", {"button": "DOWN"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "DOWN", "result": result}))]
        elif name == "press_left":
            result = api_post("/api/game/button", {"button": "LEFT"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "LEFT", "result": result}))]
        elif name == "press_right":
            result = api_post("/api/game/button", {"button": "RIGHT"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "RIGHT", "result": result}))]
        elif name == "press_start":
            result = api_post("/api/game/button", {"button": "START"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "START", "result": result}))]
        elif name == "press_select":
            result = api_post("/api/game/button", {"button": "SELECT"})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": "SELECT", "result": result}))]
        
        elif name == "get_screen":
            result = api_get("/api/screen")
            if "image" in result:
                # Return just confirmation, actual image is in result
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "frame": result.get("pyboy_frame"),
                    "image_size": len(result.get("image", "")),
                    "message": "Screen captured"
                }))]
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "tick":
            frames = arguments.get("frames", 1)
            # Tick by pressing no button but advancing frames
            result = api_post("/api/game/tick", {"frames": frames})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_memory":
            address = arguments.get("address", 0)
            result = api_get(f"/api/memory/{address}")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_state":
            result = api_get("/api/game/state")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "save_state":
            name_val = arguments.get("name", "quick_save")
            result = api_post("/api/game/save", {"name": name_val})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "load_state":
            name_val = arguments.get("name", "quick_save")
            result = api_post("/api/game/load", {"name": name_val})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_party":
            result = api_get("/api/party")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_inventory":
            result = api_get("/api/inventory")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_position":
            result = api_get("/api/memory/0xD362")  # Player X
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_game_info":
            result = api_get("/api/game/state")
            if "rom_name" in result:
                return [TextContent(type="text", text=json.dumps({
                    "name": result.get("rom_name", "Unknown"),
                    "loaded": result.get("rom_loaded", False),
                    "frame": result.get("frame_count", 0)
                }))]
            return [TextContent(type="text", text=json.dumps(result))]
        
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def main():
    """Run the MCP server"""
    logger.info(f"Starting Game Boy MCP API Client v{SERVER_VERSION}")
    logger.info(f"Connecting to backend at: {DEFAULT_BACKEND_URL}")
    logger.info("Using existing emulator instance!")
    
    # Test connection to backend
    result = api_get("/api/game/state")
    if result.get("rom_loaded"):
        logger.info(f"✅ Backend connected! ROM loaded: {result.get('rom_name', 'Unknown')}")
    else:
        logger.warning("⚠️ Backend connected but no ROM loaded")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
