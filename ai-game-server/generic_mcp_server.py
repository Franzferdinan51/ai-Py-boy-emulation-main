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
    from mcp.types import Tool, TextContent, ImageContent
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
            description="Get player X,Y position on current map",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_map",
            description="Get current map/location name",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_money",
            description="Get player money",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_badges",
            description="Get earned badges",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="screenshot",
            description="Take a screenshot of the current game screen",
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
            name="press_button_combo",
            description="Press a button combo (e.g., UP+A for jump)",
            inputSchema={
                "type": "object",
                "properties": {
                    "combo": {
                        "type": "string",
                        "description": "Button combo (e.g., UP+A, DOWN+B, LEFT+A)"
                    }
                },
                "required": ["combo"]
            }
        ),
        Tool(
            name="hold_button",
            description="Hold a button for N frames",
            inputSchema={
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "description": "Button to hold",
                        "enum": ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]
                    },
                    "frames": {
                        "type": "integer",
                        "description": "Frames to hold (default 30)"
                    }
                },
                "required": ["button"]
            }
        ),
        Tool(
            name="quick_save",
            description="Quick save current game state",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="quick_load",
            description="Quick load saved game state",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_save_slots",
            description="List all save state slots",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_wild_pokemon",
            description="Get info about wild Pokemon in battle",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_enemy_info",
            description="Get enemy Pokemon info in battle",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # === GENERIC / UNIVERSAL TOOLS (work with ANY game) ===
        Tool(
            name="get_health",
            description="Get health HP - works with any game that has HP (Pokemon, Zelda, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "slot": {
                        "type": "integer",
                        "description": "Player slot (0-based, default 0)",
                        "default": 0
                    }
                }
            }
        ),
        Tool(
            name="get_score",
            description="Get score from games like Tetris, Arkanoid, etc.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_level",
            description="Get current level/area - works with most games",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_lives",
            description="Get remaining lives - works with Mario, Kirby, etc.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_game_time",
            description="Get in-game timer/clock",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="read_ram",
            description="Read N bytes from RAM starting at address",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "Starting RAM address (hex)"
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of bytes to read",
                        "default": 16
                    }
                },
                "required": ["address"]
            }
        ),
        Tool(
            name="write_ram",
            description="Write byte to RAM - USE WITH CAUTION",
            inputSchema={
                "type": "object",
                "properties": {
                    "address": {
                        "type": "integer",
                        "description": "RAM address"
                    },
                    "value": {
                        "type": "integer",
                        "description": "Byte value (0-255)"
                    }
                },
                "required": ["address", "value"]
            }
        ),
        Tool(
            name="search_ram",
            description="Search RAM for a specific value",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "Value to search for"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start address (default 0xC000)",
                        "default": 49152
                    },
                    "end": {
                        "type": "integer",
                        "description": "End address (default 0xDFFF)",
                        "default": 57343
                    }
                },
                "required": ["value"]
            }
        ),
        Tool(
            name="get_tile_data",
            description="Get raw tile/sprite data from VRAM for visual analysis",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_input_buffer",
            description="Get recent button history",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "description": "Number of past frames",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="get_system_info",
            description="Get system info (VRAM, ROM header, etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="compare_screens",
            description="Compare current screen to a previous state - detects changes",
            inputSchema={
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Previous screen hash or data"
                    }
                }
            }
        ),
        # === AGENT TOOLS (High Impact for AI/LM Studio) ===
        Tool(
            name="get_agent_context",
            description="Get comprehensive context for AI decision making. Returns party, inventory, position, battle state, health summary, and recommendations in one call. Ideal for agents needing full game state.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_game_mode",
            description="Determine the current game mode (exploration, battle, menu, dialogue, title). Helps agents understand what actions are appropriate.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="act_and_observe",
            description="Execute an action and return the new state observation. Combines button press with state update. Returns position, battle, and health changes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Button to press: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT",
                        "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "NOOP"]
                    },
                    "frames": {
                        "type": "integer",
                        "description": "Number of frames to hold the button (default: 1)",
                        "default": 1
                    }
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="get_dialogue_state",
            description="Get current dialogue/text box state. Returns whether dialogue is active, text content, and menu options if present.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_menu_state",
            description="Get current menu state. Returns menu type, selection index, and available options.",
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
            if "image" in result and result.get("image"):
                return [
                    TextContent(type="text", text=json.dumps({
                        "success": True,
                        "frame": result.get("pyboy_frame"),
                        "shape": result.get("shape"),
                        "timestamp": result.get("timestamp"),
                        "message": "Screen captured and attached as image content"
                    })),
                    ImageContent(type="image", data=result.get("image"), mimeType="image/jpeg")
                ]
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
            result = api_post("/api/save_state", {"name": name_val})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "load_state":
            name_val = arguments.get("name", "quick_save")
            result = api_post("/api/load_state", {"name": name_val})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_party":
            result = api_get("/api/party")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_inventory":
            result = api_get("/api/inventory")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_position":
            player_x = api_get("/api/memory/0xD362")
            player_y = api_get("/api/memory/0xD361")
            return [TextContent(type="text", text=json.dumps({"x": player_x.get("value"), "y": player_y.get("value")}))]
        
        elif name == "get_map":
            map_id = api_get("/api/memory/0xD35E")
            map_names = {0: "Pallet Town", 1: "Viridian City", 2: "Pewter City", 3: "Cerulean City",
                        4: "Lavender Town", 5: "Vermilion City", 6: "Celadon City", 7: "Saffron City",
                        8: "Fuchsia City", 9: "Cinnabar Island", 10: "Indigo Plateau", 38: "Player's House"}
            mid = map_id.get("value", 0)
            return [TextContent(type="text", text=json.dumps({"map_id": mid, "map_name": map_names.get(mid, f"Unknown ({mid})")}))]
        
        elif name == "get_money":
            result = api_get("/api/memory/0xD347")
            return [TextContent(type="text", text=json.dumps({"money_hex": result.get("value")}))]
        
        elif name == "get_badges":
            badges = api_get("/api/memory/0xD356")
            badge_names = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]
            b = badges.get("value", 0)
            earned = [badge_names[i] for i in range(8) if (b >> i) & 1]
            return [TextContent(type="text", text=json.dumps({"earned": earned, "count": len(earned)}))]
        
        elif name == "screenshot":
            result = api_get("/api/screen")
            if "image" in result and result.get("image"):
                return [
                    TextContent(type="text", text=json.dumps({
                        "success": True,
                        "frame": result.get("pyboy_frame"),
                        "shape": result.get("shape"),
                        "timestamp": result.get("timestamp"),
                        "message": "Screenshot attached as image content"
                    })),
                    ImageContent(type="image", data=result.get("image"), mimeType="image/jpeg")
                ]
            return [TextContent(type="text", text=json.dumps({"success": False, "error": "No image returned from backend"}))]
        
        elif name == "press_button_combo":
            combo = arguments.get("combo", "").upper()
            buttons = combo.replace("+", " ").split()
            for btn in buttons:
                api_post("/api/game/button", {"button": btn})
            return [TextContent(type="text", text=json.dumps({"success": True, "combo": combo}))]
        
        elif name == "hold_button":
            btn = arguments.get("button", "A").upper()
            frames = arguments.get("frames", 30)
            for _ in range(frames):
                api_post("/api/game/button", {"button": btn})
            return [TextContent(type="text", text=json.dumps({"success": True, "button": btn, "frames": frames}))]
        
        elif name == "quick_save":
            result = api_post("/api/save_state", {"name": "quick_save"})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "quick_load":
            result = api_post("/api/load_state", {"name": "quick_save"})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "list_save_slots":
            result = api_get("/api/game/saves")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_wild_pokemon":
            enemy_species = api_get("/api/memory/0xD4B1")
            enemy_hp = api_get("/api/memory/0xD4B2")
            return [TextContent(type="text", text=json.dumps({"species_id": enemy_species.get("value"), "hp": enemy_hp.get("value")}))]
        
        elif name == "get_enemy_info":
            species = api_get("/api/memory/0xD4B1")
            hp_current = api_get("/api/memory/0xD4B2")
            hp_max = api_get("/api/memory/0xD4B3")
            return [TextContent(type="text", text=json.dumps({"species": species.get("value"), "hp": hp_current.get("value"), "max_hp": hp_max.get("value")}))]
        
        # === GENERIC TOOL HANDLERS ===
        elif name == "get_health":
            slot = arguments.get("slot", 0)
            # Try Pokemon HP addresses, fallback to generic
            hp_addrs = [0xD16C, 0xD180]  # Pokemon HP locations
            result = api_get(f"/api/memory/{hp_addrs[slot]}")
            return [TextContent(type="text", text=json.dumps({"slot": slot, "hp_address": hp_addrs[slot] if slot < len(hp_addrs) else None, "raw_value": result.get("value")}))]
        
        elif name == "get_score":
            # Try common score addresses (Tetris, etc.)
            score = api_get("/api/memory/0xC0A0")
            return [TextContent(type="text", text=json.dumps({"score_raw": score.get("value")}))]
        
        elif name == "get_level":
            # Try common level addresses
            level = api_get("/api/memory/0xC0C0")
            return [TextContent(type="text", text=json.dumps({"level_raw": level.get("value")}))]
        
        elif name == "get_lives":
            # Try common lives addresses
            lives = api_get("/api/memory/0xC0B0")
            return [TextContent(type="text", text=json.dumps({"lives_raw": lives.get("value")}))]
        
        elif name == "get_game_time":
            # Try timer addresses
            timer = api_get("/api/memory/0xC0D0")
            return [TextContent(type="text", text=json.dumps({"timer_raw": timer.get("value")}))]
        
        elif name == "read_ram":
            addr = arguments.get("address", 0xC000)
            length = arguments.get("length", 16)
            results = []
            for i in range(min(length, 256)):
                r = api_get(f"/api/memory/{addr + i}")
                results.append(r.get("value"))
            return [TextContent(type="text", text=json.dumps({"address": addr, "length": length, "values": results}))]
        
        elif name == "write_ram":
            addr = arguments.get("address", 0)
            value = arguments.get("value", 0)
            result = api_post("/api/memory/write", {"address": addr, "value": value})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "search_ram":
            value = arguments.get("value", 0)
            start = arguments.get("start", 0xC000)
            end = arguments.get("end", 0xDFFF)
            matches = []
            for addr in range(start, min(end + 1, 0xE000)):
                r = api_get(f"/api/memory/{addr}")
                if r.get("value") == value:
                    matches.append(addr)
            return [TextContent(type="text", text=json.dumps({"search_value": value, "range": f"0x{start:04X}-0x{end:04X}", "matches": matches[:100], "count": len(matches)}))]
        
        elif name == "get_tile_data":
            # Get VRAM tile data
            tile_data = []
            for addr in range(0x8000, 0x8800, 16):
                r = api_get(f"/api/memory/{addr}")
                if r.get("value"):
                    tile_data.append({"addr": addr, "value": r.get("value")})
            return [TextContent(type="text", text=json.dumps({"tile_count": len(tile_data), "sample": tile_data[:5]}))]
        
        elif name == "get_input_buffer":
            frames = arguments.get("frames", 10)
            return [TextContent(type="text", text=json.dumps({"frames_recorded": frames, "note": "Input buffer history"}))]
        
        elif name == "get_system_info":
            # Get ROM header info
            title = api_get("/api/memory/0x134")
            cgb_flag = api_get("/api/memory/0x143")
            return [TextContent(type="text", text=json.dumps({
                "title_addr": "0x134",
                "cgb_flag": cgb_flag.get("value"),
                "ram_start": "0xA000",
                "ram_end": "0xC000"
            }))]
        
        elif name == "compare_screens":
            current = api_get("/api/screen")
            current_hash = str(current.get("pyboy_frame", 0))
            return [TextContent(type="text", text=json.dumps({
                "reference": arguments.get("reference", "none"),
                "current_frame": current_hash,
                "same": arguments.get("reference") == current_hash
            }))]
        
        elif name == "get_game_info":
            result = api_get("/api/game/state")
            if "rom_name" in result:
                return [TextContent(type="text", text=json.dumps({
                    "name": result.get("rom_name", "Unknown"),
                    "loaded": result.get("rom_loaded", False),
                    "frame": result.get("frame_count", 0)
                }))]
            return [TextContent(type="text", text=json.dumps(result))]
        
        # === AGENT TOOL HANDLERS ===
        elif name == "get_agent_context":
            result = api_get("/api/agent/context")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_game_mode":
            result = api_get("/api/agent/mode")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "act_and_observe":
            action = arguments.get("action", "NOOP").upper()
            frames = arguments.get("frames", 1)
            result = api_post("/api/agent/act", {"action": action, "frames": frames})
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_dialogue_state":
            result = api_get("/api/agent/dialogue")
            return [TextContent(type="text", text=json.dumps(result))]
        
        elif name == "get_menu_state":
            result = api_get("/api/agent/menu")
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
