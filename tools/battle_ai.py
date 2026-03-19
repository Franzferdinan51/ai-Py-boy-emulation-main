#!/usr/bin/env python3
"""
Battle AI - Smart Combat Decision Making for Game Boy Pokemon
Analyzes battle state and makes optimal decisions

Usage:
    python battle_ai.py --rom pokemon-red.gb --action decide
    python battle_ai.py --rom pokemon-red.gb --action auto-fight
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not installed")
    sys.exit(1)


# Pokemon type chart (simplified - only includes common types)
TYPE_CHART = {
    # attacking_type: {defending_type: multiplier}
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5}
}

# Pokemon species to type mapping (partial - main starters/normals)
POKEMON_TYPES = {
    # Gen 1 starters
    4: "fire",      # Charmander
    5: "fire",      # Charmeleon
    6: "fire_flying",# Charizard
    7: "water",     # Squirtle
    8: "water",     # Wartortle
    9: "water_flying",# Blastoise
    1: "grass",     # Bulbasaur
    2: "grass_poison", # Ivysaur
    3: "grass_poison", # Venusaur
    
    # Common Pokemon
    19: "normal",   # Rattata
    20: "normal",   # Raticate
    25: "electric", # Pikachu
    26: "electric", # Raichu
    43: "grass_poison", # Oddish
    44: "grass_poison", # Gloom
    45: "grass_poison", # Vileplume
    29: "poison",   # Nidoran F
    30: "poison",   # Nidorina
    31: "poison_ground", # Nidorino
    32: "poison",   # Nidoran M
    33: "poison",   # Nidorino
    34: "poison_ground", # Nidoking
    37: "fire",     # Vulpix
    38: "fire",     # Ninetales
    41: "poison_flying", # Zubat
    42: "poison_flying", # Golbat
    50: "ground",   # Diglett
    51: "ground",   # Dugtrio
    52: "normal",   # Meowth
    53: "normal",   # Persian
    54: "water",    # Psyduck
    55: "water",    # Golduck
    56: "fighting", # Mankey
    57: "fighting", # Primeape
    63: "psychic",  # Abra
    64: "psychic",  # Kadabra
    65: "psychic",  # Alakazam
    66: "fighting", # Machop
    67: "fighting", # Machoke
    68: "fighting", # Machamp
    69: "grass_poison", # Bellsprout
    70: "grass_poison", # Weepinbell
    71: "grass_poison", # Victreebel
    72: "water",    # Tentacool
    73: "water_poison", # Tentacruel
    74: "rock_ground", # Geodude
    75: "rock_ground", # Graveler
    76: "rock_ground", # Golem
    92: "ghost_poison", # Gastly
    93: "ghost_poison", # Haunter
    94: "ghost_poison", # Gengar
    95: "rock_ground", # Onix
    96: "psychic",  # Drowzee
    97: "psychic",  # Krabby
    98: "water",    # Kingler
    99: "water",    # Voltorb
    100: "electric",# Electrode
    102: "grass",   # Exeggcute
    103: "grass_psychic", # Exeggutor
    104: "ground",  # Cubone
    105: "ground",  # Marowak
}

# Move priority levels (higher = goes first)
# Pokemon moves with priority (from Pokemon games)
MOVE_PRIORITY = {
    # Priority moves (typically go first)
    "quick_attack": 1,
    "extreme_speed": 2,
    "sucker_punch": 1,
    "fake_out": 1,
    "bullet_punch": 1,
    "mach_punch": 1,
    "aqua_jet": 1,
    "ice_shard": 1,
    "shadow_sneak": 1,
    "vacuum_wave": 1,
    "volt_tackle": 1,
    # Default moves
    "default": 0,
}

# Status moves (non-damaging moves)
STATUS_MOVES = {
    # Healing moves
    "recover": "heal",
    "rest": "heal",
    "softboiled": "heal",
    "milk_drink": "heal",
    "potion": "heal",  # Item
    "super_potion": "heal",
    "hyper_potion": "heal",
    "max_potion": "heal",
    # Status-inducing moves
    "thunder_wave": "paralyze",
    "toxic": "poison",
    "will_o_wisp": "burn",
    "sleep_powder": "sleep",
    "spore": "sleep",
    "sing": "sleep",
    "hypnosis": "sleep",
    "glare": "paralyze",
    "body_slam": "paralyze",
    "stun_spore": "paralyze",
    # Stat-boosting moves
    "swords_dance": "attack_up",
    "amnesia": "defense_up",
    "barrier": "defense_up",
    "aurora_beam": "attack_down",
    "growl": "attack_down",
    "tail_whip": "defense_down",
    # Utility moves
    "teleport": "escape",
    "whirlwind": "escape",
    "roar": "escape",
    "fly": "escape",
    "dig": "escape",
    "flash": "evasion_down_enemy,
}

# Move power and accuracy (simplified)
MOVE_DATA = {
    # Moves by ID (Gen 1 simplified)
    1: {"name": "pound", "power": 40, "accuracy": 100, "type": "normal", "priority": 0},
    2: {"name": "karate_chop", "power": 50, "accuracy": 100, "type": "fighting", "priority": 0},
    3: {"name": "double_kick", "power": 30, "accuracy": 100, "type": "fighting", "priority": 0},
    4: {"name": "comet_punch", "power": 18, "accuracy": 85, "type": "normal", "priority": 0},
    5: {"name": "mega_punch", "power": 80, "accuracy": 85, "type": "normal", "priority": 0},
    6: {"name": "pay_day", "power": 40, "accuracy": 100, "type": "normal", "priority": 0},
    7: {"name": "fire_punch", "power": 75, "accuracy": 100, "type": "fire", "priority": 0},
    8: {"name": "ice_punch", "power": 75, "accuracy": 100, "type": "ice", "priority": 0},
    9: {"name": "thunder_punch", "power": 75, "accuracy": 100, "type": "electric", "priority": 0},
    10: {"name": "scratch", "power": 40, "accuracy": 100, "type": "normal", "priority": 0},
    11: {"name": "vice_grip", "power": 55, "accuracy": 100, "type": "normal", "priority": 0},
    12: {"name": "guillotine", "power": 0, "accuracy": 30, "type": "normal", "priority": 0},  # OHKO
    13: {"name": "razor_wind", "power": 80, "accuracy": 75, "type": "normal", "priority": 0},
    14: {"name": "swords_dance", "power": 0, "accuracy": 100, "type": "normal", "status": "attack_up", "priority": 0},
    15: {"name": "cut", "power": 50, "accuracy": 95, "type": "normal", "priority": 0},
    16: {"name": "gust", "power": 40, "accuracy": 100, "type": "flying", "priority": 0},
    17: {"name": "wing_attack", "power": 35, "accuracy": 100, "type": "flying", "priority": 0},
    18: {"name": "fly", "power": 70, "accuracy": 95, "type": "flying", "priority": 0},
    19: {"name": "bind", "power": 15, "accuracy": 85, "type": "normal", "priority": 0},
    20: {"name": "vine_whip", "power": 35, "accuracy": 100, "type": "grass", "priority": 0},
    21: {"name": "tackle", "power": 35, "accuracy": 95, "type": "normal", "priority": 0},
    22: {"name": "body_slam", "power": 60, "accuracy": 100, "type": "normal", "priority": 0},
    23: {"name": "wrap", "power": 15, "accuracy": 90, "type": "normal", "priority": 0},
    24: {"name": "take_down", "power": 90, "accuracy": 85, "type": "normal", "priority": 0},
    25: {"name": "thrash", "power": 90, "accuracy": 100, "type": "normal", "priority": 0},
    26: {"name": "tail_whip", "power": 0, "accuracy": 100, "type": "normal", "status": "defense_down", "priority": 0},
    27: {"name": "poison_powder", "power": 0, "accuracy": 75, "type": "poison", "status": "poison", "priority": 0},
    28: {"name": "stun_spore", "power": 0, "accuracy": 75, "type": "grass", "status": "paralyze", "priority": 0},
    29: {"name": "sleep_powder", "power": 0, "accuracy": 75, "type": "grass", "status": "sleep", "priority": 0},
    30: {"name": "petal_dance", "power": 70, "accuracy": 100, "type": "grass", "priority": 0},
    31: {"name": "string_shot", "power": 0, "accuracy": 95, "type": "bug", "status": "speed_down", "priority": 0},
    32: {"name": "dragon_rage", "power": 40, "accuracy": 100, "type": "dragon", "priority": 0},
    33: {"name": "fire_spin", "power": 15, "accuracy": 85, "type": "fire", "priority": 0},
    34: {"name": "thunder_shock", "power": 40, "accuracy": 100, "type": "electric", "priority": 0},
    35: {"name": "thunderbolt", "power": 90, "accuracy": 100, "type": "electric", "priority": 0},
    36: {"name": "thunder_wave", "power": 0, "accuracy": 90, "type": "electric", "status": "paralyze", "priority": 0},
    37: {"name": "rock_throw", "power": 50, "accuracy": 90, "type": "rock", "priority": 0},
    38: {"name": "earthquake", "power": 100, "accuracy": 100, "type": "ground", "priority": 0},
    39: {"name": "fissure", "power": 0, "accuracy": 30, "type": "ground", "priority": 0},  # OHKO
    40: {"name": "dig", "power": 60, "accuracy": 100, "type": "ground", "priority": 0},
    41: {"name": "toxic", "power": 0, "accuracy": 90, "type": "poison", "status": "poison", "priority": 0},
    42: {"name": "confusion", "power": 50, "accuracy": 100, "type": "psychic", "priority": 0},
    43: {"name": "psychic", "power": 90, "accuracy": 100, "type": "psychic", "priority": 0},
    44: {"name": "hypnosis", "power": 0, "accuracy": 60, "type": "psychic", "status": "sleep", "priority": 0},
    45: {"name": "meditate", "power": 0, "accuracy": 100, "type": "psychic", "status": "attack_up", "priority": 0},
    46: {"name": "agility", "power": 0, "accuracy": 100, "type": "psychic", "status": "speed_up", "priority": 0},
    47: {"name": "quick_attack", "power": 40, "accuracy": 100, "type": "normal", "priority": 1},
    48: {"name": "rage", "power": 20, "accuracy": 100, "type": "normal", "priority": 0},
    49: {"name": "teleport", "power": 0, "accuracy": 100, "type": "psychic", "status": "escape", "priority": 0},
    50: {"name": "night_shade", "power": 0, "accuracy": 100, "type": "ghost", "priority": 0},  # Damage = user level
    51: {"name": "mimic", "power": 0, "accuracy": 100, "type": "normal", "priority": 0},
    52: {"name": "screech", "power": 0, "accuracy": 85, "type": "normal", "status": "defense_down", "priority": 0},
    53: {"name": "double_team", "power": 0, "accuracy": 100, "type": "normal", "status": "evasion_up", "priority": 0},
    54: {"name": "recover", "power": 0, "accuracy": 100, "type": "normal", "status": "heal", "priority": 0},
    55: {"name": "harden", "power": 0, "accuracy": 100, "type": "normal", "status": "defense_up", "priority": 0},
    56: {"name": "minimize", "power": 0, "accuracy": 100, "type": "normal", "status": "evasion_up", "priority": 0},
    57: {"name": "smokescreen", "power": 0, "accuracy": 100, "type": "normal", "status": "evasion_down_enemy", "priority": 0},
    58: {"name": "confuse_ray", "power": 0, "accuracy": 100, "type": "ghost", "status": "confuse", "priority": 0},
    59: {"name": "withdraw", "power": 0, "accuracy": 100, "type": "water", "status": "defense_up", "priority": 0},
    60: {"name": "defense_curl", "power": 0, "accuracy": 100, "type": "normal", "status": "defense_up", "priority": 0},
}


def get_type_advantage(attack_type: str, defend_types: List[str]) -> float:
    """Calculate type effectiveness"""
    multiplier = 1.0
    
    for defend_type in defend_types:
        chart = TYPE_CHART.get(attack_type, {})
        multiplier *= chart.get(defend_type, 1.0)
    
    return multiplier


def get_pokemon_type(species_id: int) -> List[str]:
    """Get Pokemon type(s) from species ID"""
    types_str = POKEMON_TYPES.get(species_id, "normal")
    
    if "_" in types_str:
        return types_str.split("_")
    return [types_str]


def get_move_priority(move_id: int) -> int:
    """Get priority level of a move"""
    move_data = MOVE_DATA.get(move_id, {})
    return move_data.get("priority", 0)


def get_move_data(move_id: int) -> Dict:
    """Get full move data"""
    return MOVE_DATA.get(move_id, {
        "name": "unknown",
        "power": 0,
        "accuracy": 100,
        "type": "normal",
        "priority": 0
    })


def is_status_move(move_id: int) -> bool:
    """Check if a move is a status move (non-damaging)"""
    move_data = MOVE_DATA.get(move_id, {})
    return "status" in move_data


def get_move_type(move_id: int) -> str:
    """Get the type of a move"""
    move_data = MOVE_DATA.get(move_id, {})
    return move_data.get("type", "normal")


def get_move_power(move_id: int) -> int:
    """Get the power of a move"""
    move_data = MOVE_DATA.get(move_id, {})
    return move_data.get("power", 0)


def prioritize_moves(move_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Sort moves by priority and type effectiveness.
    Returns list of (move_id, priority_score) sorted by priority.
    """
    move_priorities = []
    
    for move_id in move_ids:
        priority = get_move_priority(move_id)
        power = get_move_power(move_id)
        is_status = is_status_move(move_id)
        
        # Calculate priority score:
        # Higher priority moves go first
        # High-power attacks are prioritized
        # Status moves have lower base priority
        
        if is_status:
            # Status moves get lower priority unless HP is low
            priority_score = priority - 10
        else:
            # Attack moves: power + priority * 100
            priority_score = priority * 100 + power
        
        move_priorities.append((move_id, priority_score))
    
    # Sort by priority score (highest first)
    move_priorities.sort(key=lambda x: x[1], reverse=True)
    
    return move_priorities


def use_status_move_heuristic(
    player_hp_pct: float,
    enemy_hp_pct: float,
    enemy_status: str = None
) -> str:
    """
    Determine when to use status moves vs attacks.
    Returns: 'attack', 'status', 'heal', or 'run'
    """
    
    # If player is low HP, prioritize healing
    if player_hp_pct < 30:
        return "heal"
    
    # If enemy is already statused, attack
    if enemy_status is not None:
        return "attack"
    
    # If player has type advantage, attack aggressively
    if player_hp_pct > 70 and enemy_hp_pct < 50:
        return "attack"
    
    # If enemy has type advantage, try status
    if enemy_hp_pct > player_hp_pct:
        return "status"
    
    # Default to attack
    return "attack"


def analyze_battle_state(emulator) -> Dict:
    """Read current battle state from memory"""
    
    # Check if in battle
    battle_status = emulator.memory[0xD057]
    
    if battle_status == 0:
        return {
            "in_battle": False,
            "error": "Not in battle"
        }
    
    # Get player Pokemon info
    player_hp = (emulator.memory[0xD6B6] << 8) | emulator.memory[0xD6B5]
    player_max_hp = (emulator.memory[0xD6BF] << 8) | emulator.memory[0xD6BE]
    player_level = emulator.memory[0xD18C]
    player_species = emulator.memory[0xD163]
    
    # Get enemy Pokemon info
    enemy_hp = (emulator.memory[0xD89D] << 8) | emulator.memory[0xD89C]
    enemy_max_hp = (emulator.memory[0xD8A1] << 8) | emulator.memory[0xD8A0]
    enemy_level = emulator.memory[0xD8C6]
    enemy_species = emulator.memory[0xD883]
    
    # Calculate percentages
    player_hp_pct = (player_hp / player_max_hp * 100) if player_max_hp > 0 else 0
    enemy_hp_pct = (enemy_hp / enemy_max_hp * 100) if enemy_max_hp > 0 else 0
    
    # Get types
    player_types = get_pokemon_type(player_species)
    enemy_types = get_pokemon_type(enemy_species)
    
    # Calculate type advantages
    player_advantage = 0
    enemy_advantage = 0
    
    for ptype in player_types:
        if "_" in ptype:
            ptype = ptype.split("_")[0]  # Take first type
        player_advantage += get_type_advantage(ptype, enemy_types)
    
    for etype in enemy_types:
        if "_" in etype:
            etype = etype.split("_")[0]
        enemy_advantage += get_type_advantage(etype, player_types)
    
    return {
        "in_battle": True,
        "battle_status": battle_status,
        "player": {
            "species_id": player_species,
            "level": player_level,
            "hp": player_hp,
            "max_hp": player_max_hp,
            "hp_percent": round(player_hp_pct, 1),
            "types": player_types,
            "type_advantage_vs_enemy": player_advantage
        },
        "enemy": {
            "species_id": enemy_species,
            "level": enemy_level,
            "hp": enemy_hp,
            "max_hp": enemy_max_hp,
            "hp_percent": round(enemy_hp_pct, 1),
            "types": enemy_types,
            "type_advantage_vs_player": enemy_advantage
        },
        "recommendations": generate_recommendations(
            player_hp_pct, enemy_hp_pct,
            player_advantage, enemy_advantage,
            player_species, enemy_species,
            player_level, enemy_level
        )
    }


def generate_recommendations(
    player_hp_pct: float,
    enemy_hp_pct: float,
    player_advantage: float,
    enemy_advantage: float,
    player_species: int,
    enemy_species: int,
    player_level: int,
    enemy_level: int
) -> Dict:
    """Generate battle recommendations"""
    
    recommendations = []
    action = "fight"
    confidence = "medium"
    
    # Check if should run
    if player_hp_pct < 20:
        action = "run"
        confidence = "high"
        recommendations.append("Critical HP - consider running!")
    elif player_advantage < 0.5:
        action = "run"
        confidence = "medium"
        recommendations.append("Bad type matchup - consider running")
    elif enemy_hp_pct > player_hp_pct * 2 and player_hp_pct < 50:
        action = "run"
        confidence = "medium"
        recommendations.append("Enemy has significantly more HP")
    
    # Check if can win
    if enemy_hp_pct < 20:
        action = "fight"
        confidence = "high"
        recommendations.append("Enemy low - can finish!")
    
    # Check type advantages
    if player_advantage >= 2:
        recommendations.append(f"Strong type advantage ({player_advantage}x)!")
        action = "fight"
    elif player_advantage <= 0.5:
        recommendations.append(f"Weak type matchup ({player_advantage}x)")
    
    # Level check
    if player_level > enemy_level + 5:
        recommendations.append("You have level advantage")
    elif enemy_level > player_level + 5:
        recommendations.append("Enemy has level advantage - be careful")
    
    return {
        "recommended_action": action,
        "confidence": confidence,
        "reasoning": recommendations
    }


def decide_move(emulator) -> Dict:
    """Decide what move to make"""
    
    state = analyze_battle_state(emulator)
    
    if not state.get("in_battle"):
        return state
    
    recommendations = state.get("recommendations", {})
    action = recommendations.get("recommended_action", "fight")
    confidence = recommendations.get("confidence", "low")
    
    # Determine button sequence
    if action == "run":
        # Try to run (START + UP typically in Pokemon)
        button_sequence = "START"
    elif action == "fight":
        # Select FIGHT option (usually A from menu)
        button_sequence = "A"
    else:
        button_sequence = "A"  # Default to attack
    
    return {
        "decision": action,
        "confidence": confidence,
        "button_sequence": button_sequence,
        "battle_state": state,
        "reasoning": recommendations.get("reasoning", [])
    }


def auto_fight(emulator, max_moves: int = 10) -> Dict:
    """Execute automatic battle"""
    
    if not analyze_battle_state(emulator).get("in_battle"):
        return {"success": False, "error": "Not in battle"}
    
    moves = []
    
    for i in range(max_moves):
        # Check if still in battle
        if emulator.memory[0xD057] == 0:
            break
        
        # Get decision
        decision = decide_move(emulator)
        
        # Execute
        emulator.send_input(emulator.memory)
        # Press A to select action/move
        from pyboy.utils import WindowEvent
        emulator.send_input(WindowEvent.PRESS_BUTTON_A)
        emulator.tick()
        emulator.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        moves.append({
            "move": i + 1,
            "decision": decision.get("decision"),
            "buttons": decision.get("button_sequence")
        })
        
        # Check if battle ended
        if emulator.memory[0xD057] == 0:
            break
    
    # Get final state
    final_state = analyze_battle_state(emulator)
    
    return {
        "success": True,
        "moves_executed": len(moves),
        "moves": moves,
        "final_state": final_state,
        "victory": final_state.get("enemy", {}).get("hp", 1) == 0
    }


def main():
    parser = argparse.ArgumentParser(description="Battle AI for Pokemon Games")
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument("--action", choices=["analyze", "decide", "auto-fight"], default="decide")
    parser.add_argument("--max-moves", type=int, default=10, help="Max moves for auto-fight")
    
    args = parser.parse_args()
    
    if not Path(args.rom).exists():
        print(f"ERROR: ROM not found: {args.rom}")
        sys.exit(1)
    
    # Initialize emulator
    emulator = PyBoy(args.rom, window="null")
    
    if args.action == "analyze":
        result = analyze_battle_state(emulator)
    elif args.action == "decide":
        result = decide_move(emulator)
    elif args.action == "auto-fight":
        result = auto_fight(emulator, args.max_moves)
    
    print(json.dumps(result, indent=2))
    
    emulator.stop()


if __name__ == "__main__":
    main()