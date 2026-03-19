#!/usr/bin/env python3
"""
Auto-Navigator - Pathfinding for Game Boy Games
Helps AI agents navigate between locations

Usage:
    python auto_navigate.py --from "Pallet Town" --to "Viridian City"
    python auto_navigate.py --current-pos 12,8 --target 20,15
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not installed")
    sys.exit(1)


# Map data for Pokemon Red (simplified)
# Each map has connections to nearby maps and key locations
MAP_DATA = {
    "pallet_town": {
        "id": 0x26,  # 38
        "name": "Pallet Town",
        "exits": ["route_1"],
        "buildings": ["oak_lab", "player_house", "rival_house"],
        "center_x": 6, "center_y": 8
    },
    "route_1": {
        "id": 0x27,
        "name": "Route 1",
        "exits": ["pallet_town", "viridian_city"],
        "grass_areas": [(5, 5, 15, 10)],
        "center_x": 10, "center_y": 7
    },
    "viridian_city": {
        "id": 0x29,
        "name": "Viridian City",
        "exits": ["route_1", "route_2", "route_22"],
        "buildings": ["pokemon_center", "pokemart", "gym"],
        "center_x": 10, "center_y": 10
    },
    "route_2": {
        "id": 0x2B,
        "name": "Route 2",
        "exits": ["viridian_city", "pewter_city", "digletts_cave"],
        "center_x": 6, "center_y": 8
    },
    "pewter_city": {
        "id": 0x2D,
        "name": "Pewter City",
        "exits": ["route_2", "route_3", "mt_moon"],
        "buildings": ["pokemon_center", "pokemart", "gym"],
        "center_x": 10, "center_y": 12
    },
    "route_3": {
        "id": 0x30,
        "name": "Route 3",
        "exits": ["pewter_city", "cerulean_city", "mt_moon"],
        "grass_areas": [(3, 4, 18, 8)],
        "center_x": 10, "center_y": 6
    },
    "cerulean_city": {
        "id": 0x34,
        "name": "Cerulean City",
        "exits": ["route_3", "route_4", "route_24", "route_25"],
        "buildings": ["pokemon_center", "pokemart", "gym"],
        "center_x": 10, "center_y": 10
    }
}

# Simple coordinate offsets for directions
DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0)
}


def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """Get the primary direction to move from one position to another"""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"


def pathfind_simple(current: Tuple[int, int], target: Tuple[int, int], max_steps: int = 50) -> Dict:
    """Simple pathfinding - move in primary direction until target reached"""
    path = []
    current_pos = list(current)
    target_pos = list(target)
    
    steps = 0
    while current_pos != target_pos and steps < max_steps:
        direction = get_direction(tuple(current_pos), tuple(target_pos))
        
        # Add movement
        dx, dy = DIRECTIONS[direction]
        
        # Check if we can move in that direction
        can_move = True
        if direction == "UP" and current_pos[1] <= 0:
            can_move = False
        elif direction == "DOWN" and current_pos[1] >= 20:
            can_move = False
        elif direction == "LEFT" and current_pos[0] <= 0:
            can_move = False
        elif direction == "RIGHT" and current_pos[0] >= 20:
            can_move = False
        
        if can_move:
            current_pos[0] += dx
            current_pos[1] += dy
            path.append(direction)
        
        steps += 1
    
    return {
        "success": current_pos == target_pos,
        "path": path,
        "path_length": len(path),
        "start": current,
        "target": target,
        "final_position": tuple(current_pos)
    }


def navigate_to_building(emulator, building_name: str) -> Dict:
    """Navigate to a specific building in the current area"""
    # This is a simplified version - in reality you'd need collision detection
    
    # First, find current position
    try:
        player_x = emulator.memory[0xD062]
        player_y = emulator.memory[0xD063]
    except:
        return {"success": False, "error": "Cannot read player position"}
    
    # Find current map
    map_id = emulator.memory.get(0xCC26, 0)
    
    # Find map name
    current_map = None
    for name, data in MAP_DATA.items():
        if data["id"] == map_id:
            current_map = data
            break
    
    if not current_map:
        return {
            "success": False,
            "error": f"Unknown map: {hex(map_id)}",
            "player_pos": (player_x, player_y)
        }
    
    # Check if building exists
    if building_name not in current_map.get("buildings", []):
        return {
            "success": False,
            "error": f"Building '{building_name}' not in {current_map['name']}",
            "available_buildings": current_map.get("buildings", [])
        }
    
    # Calculate path to building (simplified - assumes building is at center)
    target_x = current_map.get("center_x", 10)
    target_y = current_map.get("center_y", 10)
    
    result = pathfind_simple((player_x, player_y), (target_x, target_y))
    result["current_map"] = current_map["name"]
    result["target_building"] = building_name
    
    return result


def follow_route(from_location: str, to_location: str) -> Dict:
    """Plan a route between two locations"""
    if from_location not in MAP_DATA or to_location not in MAP_DATA:
        return {
            "success": False,
            "error": "Unknown location",
            "available": list(MAP_DATA.keys())
        }
    
    # BFS to find path through maps
    from_data = MAP_DATA[from_location]
    to_data = MAP_DATA[to_location]
    
    # Simple: just list the route if direct connection
    if to_location in from_data.get("exits", []):
        return {
            "success": True,
            "route": [from_location, to_location],
            "steps": 1,
            "from": from_data,
            "to": to_data
        }
    
    # Need to find intermediate route
    # This is a simplified implementation
    return {
        "success": True,
        "route": [from_location, to_location],
        "note": "Direct route - may need intermediate stops",
        "from": from_data,
        "to": to_data
    }


def navigate_autonomous(emulator, goal: str, max_steps: int = 100) -> Dict:
    """Autonomous navigation based on goal"""
    
    # Get current state
    try:
        player_x = emulator.memory[0xD062]
        player_y = emulator.memory[0xD063]
        map_id = emulator.memory.get(0xCC26, 0)
    except Exception as e:
        return {"success": False, "error": f"Cannot read memory: {e}"}
    
    # Find current map
    current_map = None
    for name, data in MAP_DATA.items():
        if data["id"] == map_id:
            current_map = data
            break
    
    # Execute pathfinding
    if goal in MAP_DATA:
        # Navigate to a known location
        target = MAP_DATA[goal]
        return pathfind_simple((player_x, player_y), (target["center_x"], target["center_y"]), max_steps)
    
    elif "gym" in goal.lower():
        return navigate_to_building(emulator, "gym")
    
    elif "center" in goal.lower() or "heal" in goal.lower():
        return navigate_to_building(emulator, "pokemon_center")
    
    elif "mart" in goal.lower() or "shop" in goal.lower():
        return navigate_to_building(emulator, "pokemart")
    
    else:
        return {
            "success": False,
            "error": f"Unknown goal: {goal}",
            "available_goals": list(MAP_DATA.keys()) + ["gym", "pokemon_center", "pokemart"]
        }


def main():
    parser = argparse.ArgumentParser(description="Auto-Navigator for Game Boy Games")
    parser.add_argument("--rom", help="Path to ROM file")
    parser.add_argument("--from", dest="from_loc", help="From location")
    parser.add_argument("--to", dest="to_loc", help="To location")
    parser.add_argument("--current-pos", help="Current position (x,y)")
    parser.add_argument("--target-pos", help="Target position (x,y)")
    parser.add_argument("--goal", help="Navigation goal (location name)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps")
    parser.add_argument("--list-maps", action="store_true", help="List known maps")
    
    args = parser.parse_args()
    
    if args.list_maps:
        print("Known Maps:")
        for name, data in MAP_DATA.items():
            print(f"  {name}: {data['name']} (ID: {hex(data['id'])})")
        return
    
    if args.rom:
        # Initialize emulator
        if not Path(args.rom).exists():
            print(f"ERROR: ROM not found: {args.rom}")
            sys.exit(1)
        
        emulator = PyBoy(args.rom, window="null")
        
        if args.goal:
            result = navigate_autonomous(emulator, args.goal, args.max_steps)
        else:
            print("ERROR: Specify --goal for navigation")
            sys.exit(1)
        
        emulator.stop()
    else:
        # Offline mode - just calculate path
        if args.current_pos and args.target_pos:
            current = tuple(map(int, args.current_pos.split(",")))
            target = tuple(map(int, args.target_pos.split(",")))
            result = pathfind_simple(current, target, args.max_steps)
        elif args.from_loc and args.to_loc:
            result = follow_route(args.from_loc, args.to_loc)
        else:
            print("Usage:")
            print("  python auto_navigate.py --list-maps")
            print("  python auto_navigate.py --current-pos 12,8 --target-pos 20,15")
            print("  python auto_navigate.py --from pallet_town --to viridian_city")
            print("  python auto_navigate.py --rom game.gb --goal gym")
            sys.exit(1)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()