#!/usr/bin/env python3
"""
Game State - Comprehensive Game State Manager for Pokemon Games
Tracks game state, makes decisions, and plans actions

Features:
    - GameState class for tracking all game data
    - DecisionEngine for AI decision making
    - ActionPlanner for planning action sequences
    - State persistence and history

Usage:
    python game_state.py --rom pokemon-red.gb --status
    python game_state.py --rom pokemon-red.gb --decide "explore"
    python game_state.py --rom pokemon-red.gb --plan "heal"
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import deque

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not installed")
    sys.exit(1)


class GameMode(Enum):
    """Current game mode/activity"""
    IDLE = "idle"
    EXPLORING = "exploring"
    BATTLE = "battle"
    GRINDING = "grinding"
    HEALING = "healing"
    CATCHING = "catching"
    TRADING = "trading"
    MENU = "menu"


class ActionType(Enum):
    """Types of actions available"""
    MOVE = "move"
    ATTACK = "attack"
    ITEM = "item"
    RUN = "run"
    TALK = "talk"
    MENU = "menu"
    WAIT = "wait"


@dataclass
class PlayerState:
    """Current player state"""
    x: int = 0
    y: int = 0
    money: int = 0
    badges: List[str] = field(default_factory=list)
    location: str = "unknown"
    map_id: int = 0


@dataclass
class PokemonPartyState:
    """Party Pokemon state"""
    species: str = ""
    level: int = 0
    current_hp: int = 0
    max_hp: int = 0
    attack: int = 0
    defense: int = 0
    speed: int = 0
    status: str = "none"
    hp_percent: float = 100.0


@dataclass
class BattleState:
    """Current battle state"""
    in_battle: bool = False
    battle_type: str = "none"  # wild, trainer
    enemy_species: str = ""
    enemy_level: int = 0
    enemy_hp: int = 0
    enemy_max_hp: int = 0
    player_hp: int = 0
    player_max_hp: int = 0
    turn: int = 0


@dataclass
class GameState:
    """
    Complete game state manager
    Tracks all game data and provides decision-making
    """
    # Core state
    mode: str = GameMode.IDLE.value
    frame_count: int = 0
    last_update: str = ""
    
    # Player data
    player: PlayerState = field(default_factory=PlayerState)
    
    # Party data
    party: List[PokemonPartyState] = field(default_factory=list)
    party_count: int = 0
    
    # Battle data
    battle: BattleState = field(default_factory=BattleState)
    
    # Inventory
    inventory: List[Dict] = field(default_factory=list)
    balls: int = 0
    
    # History for decision making
    action_history: List[Dict] = field(default_factory=list)
    location_history: List[str] = field(default_factory=list)
    battle_count: int = 0
    
    # Session goals
    current_goal: str = ""
    goal_progress: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert enums to strings
        data["mode"] = self.mode
        return data
    
    def update_from_scanner(self, scanner) -> None:
        """Update state from memory scanner"""
        self.last_update = datetime.now().isoformat()
        
        # Update player position
        pos = scanner.read_position()
        if "error" not in pos:
            self.player.x = pos.get("x", 0)
            self.player.y = pos.get("y", 0)
        
        # Update money
        money_data = scanner.read_money()
        if "error" not in money_data:
            self.player.money = money_data.get("money", 0)
        
        # Update badges
        self.player.badges = scanner.read_badges()
        
        # Update map
        map_data = scanner.read_map()
        if "error" not in map_data:
            self.player.map_id = map_data.get("map_id", 0)
        
        # Update party
        party_data = scanner.read_party_stats()
        self.party_count = len(party_data)
        self.party = []
        for p in party_data:
            self.party.append(PokemonPartyState(
                species=p.get("species_name", "Unknown"),
                level=p.get("level", 0),
                current_hp=p.get("current_hp", 0),
                max_hp=p.get("max_hp", 0),
                attack=p.get("attack", 0),
                defense=p.get("defense", 0),
                speed=p.get("speed", 0),
                status=p.get("status", "none"),
                hp_percent=p.get("hp_percent", 100.0)
            ))
        
        # Update inventory
        self.inventory = scanner.read_inventory()
        
        # Update balls
        ball_data = scanner.read_balls()
        self.balls = ball_data.get("poke_balls", 0)
        
        # Update battle state
        battle_data = scanner.read_battle_state()
        self.battle.in_battle = battle_data.get("in_battle", False)
        self.battle.battle_type = battle_data.get("battle_type", "none")
        
        if self.battle.in_battle:
            hp_data = scanner.read_hp()
            self.battle.player_hp = hp_data.get("player_current", 0)
            self.battle.player_max_hp = hp_data.get("player_max", 1)
            self.battle.enemy_hp = hp_data.get("enemy_current", 0)
            self.battle.enemy_max_hp = hp_data.get("enemy_max", 1)
            self.battle.turn += 1
            self.mode = GameMode.BATTLE.value
        else:
            self.battle.turn = 0
    
    def get_first_pokemon(self) -> Optional[PokemonPartyState]:
        """Get first Pokemon in party"""
        if self.party:
            return self.party[0]
        return None
    
    def needs_healing(self, threshold: float = 30.0) -> bool:
        """Check if any Pokemon needs healing"""
        for pokemon in self.party:
            if pokemon.hp_percent < threshold:
                return True
        return False
    
    def can_catch(self) -> bool:
        """Check if can catch Pokemon"""
        return self.balls > 0 and self.battle.in_battle
    
    def get_action_summary(self) -> str:
        """Get readable action summary"""
        if self.battle.in_battle:
            return f"In battle vs {self.battle.enemy_species} (HP: {self.battle.enemy_hp}/{self.battle.enemy_max_hp})"
        elif self.mode == GameMode.EXPLORING.value:
            return f"Exploring at ({self.player.x}, {self.player.y})"
        elif self.mode == GameMode.GRINDING.value:
            return f"Grinding - Battles: {self.battle_count}"
        else:
            return f"Idle - Money: ${self.player.money:,}"


class DecisionEngine:
    """
    AI Decision Engine for Pokemon games
    Analyzes game state and makes optimal decisions
    """
    
    def __init__(self, game_state: GameState):
        self.state = game_state
        self.decision_history: List[Dict] = []
    
    def should_heal(self) -> Tuple[bool, str]:
        """Decide if player should heal"""
        first_pokemon = self.state.get_first_pokemon()
        
        if not first_pokemon:
            return False, "No Pokemon in party"
        
        # Critical HP - always heal
        if first_pokemon.hp_percent < 20:
            return True, "Critical HP"
        
        # In battle with low HP - should run or heal
        if self.state.battle.in_battle and first_pokemon.hp_percent < 30:
            if self.state.battle.enemy_hp > first_pokemon.hp:
                return True, "Low HP in battle"
        
        # Low HP during grinding
        if self.state.mode == GameMode.GRINDING.value and first_pokemon.hp_percent < 50:
            return True, "Low HP during grinding"
        
        return False, "HP OK"
    
    def should_run_from_battle(self) -> Tuple[bool, str]:
        """Decide if should run from battle"""
        if not self.state.battle.in_battle:
            return False, "Not in battle"
        
        first_pokemon = self.state.get_first_pokemon()
        if not first_pokemon:
            return True, "No Pokemon"
        
        # Critical HP
        if first_pokemon.hp_percent < 15:
            return True, "Critical HP"
        
        # Enemy has type advantage and we're low
        if first_pokemon.hp_percent < 30 and self.state.battle.enemy_hp > first_pokemon.hp:
            return True, "Enemy stronger"
        
        # No balls for catching
        if self.state.balls == 0:
            return True, "No balls"
        
        return False, "Can fight"
    
    def should_catch(self) -> Tuple[bool, str]:
        """Decide if should attempt catch"""
        if not self.state.battle.in_battle:
            return False, "Not in battle"
        
        if self.state.balls == 0:
            return False, "No balls"
        
        first_pokemon = self.state.get_first_pokemon()
        if not first_pokemon:
            return False, "No Pokemon"
        
        # Enemy low HP - good chance to catch
        enemy_hp_pct = (self.state.battle.enemy_hp / max(1, self.state.battle.enemy_max_hp)) * 100
        if enemy_hp_pct < 30:
            return True, "Enemy low HP"
        
        # Rare Pokemon (based on species - simplified)
        rare_species = ["Mewtwo", "Mew", "Dragonite", "Lapras", "Snorlax", "Articuno", "Zapdos", "Moltres"]
        if self.state.battle.enemy_species in rare_species:
            return True, "Rare Pokemon"
        
        # Player has full health and plenty of balls
        if first_pokemon.hp_percent > 70 and self.state.balls > 5:
            return True, "Can afford to try"
        
        return False, "Not worth catching"
    
    def get_battle_action(self) -> Dict[str, Any]:
        """Decide battle action"""
        if not self.state.battle.in_battle:
            return {"action": "wait", "reason": "Not in battle"}
        
        first_pokemon = self.state.get_first_pokemon()
        if not first_pokemon:
            return {"action": "run", "reason": "No Pokemon"}
        
        # Check if should run
        should_run, run_reason = self.should_run_from_battle()
        if should_run:
            return {"action": "run", "reason": run_reason}
        
        # Check if should catch
        should_catch, catch_reason = self.should_catch()
        if should_catch:
            return {"action": "catch", "reason": catch_reason}
        
        # Default: attack
        return {
            "action": "attack",
            "reason": "Attack for damage",
            "move": "best"
        }
    
    def get_explore_action(self) -> Dict[str, Any]:
        """Decide exploration action"""
        # Simple exploration: move in a pattern
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Use frame count to determine direction
        direction = directions[self.state.frame_count % 4]
        
        return {
            "action": "move",
            "direction": direction,
            "reason": "Explore"
        }
    
    def get_grind_action(self) -> Dict[str, Any]:
        """Decide grinding action"""
        # Check if need to heal
        should_heal, heal_reason = self.should_heal()
        if should_heal:
            return {
                "action": "heal",
                "reason": heal_reason
            }
        
        # Check battle
        if self.state.battle.in_battle:
            return self.get_battle_action()
        
        # Move to find wild Pokemon
        return {
            "action": "move",
            "direction": ["UP", "DOWN", "LEFT", "RIGHT"][self.state.frame_count % 4],
            "reason": "Find wild Pokemon"
        }
    
    def make_decision(self) -> Dict[str, Any]:
        """Make decision based on current mode"""
        # Update decision history
        decision = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.state.mode,
            "frame": self.state.frame_count
        }
        
        if self.state.mode == GameMode.BATTLE.value:
            action = self.get_battle_action()
        elif self.state.mode == GameMode.GRINDING.value:
            action = self.get_grind_action()
        elif self.state.mode == GameMode.EXPLORING.value:
            action = self.get_explore_action()
        else:
            action = {"action": "idle", "reason": "No active mode"}
        
        decision.update(action)
        self.decision_history.append(decision)
        
        return decision


class ActionPlanner:
    """
    Plans action sequences for game automation
    Converts decisions into button press sequences
    """
    
    # Button mappings
    BUTTON_MAP = {
        "A": "A",
        "B": "B",
        "UP": "UP",
        "DOWN": "DOWN",
        "LEFT": "LEFT",
        "RIGHT": "RIGHT",
        "START": "START",
        "SELECT": "SELECT",
    }
    
    def __init__(self, game_state: GameState):
        self.state = game_state
        self.planned_actions: List[Dict] = []
    
    def plan_battle(self, action: Dict) -> List[str]:
        """Plan battle action sequence"""
        actions = []
        
        if action["action"] == "attack":
            # Press A to select FIGHT, then A again for first move
            actions = ["A", "A"]
        
        elif action["action"] == "run":
            # Run from battle (START + UP or SELECT)
            actions = ["SELECT", "DOWN", "A"]  # Run option
        
        elif action["action"] == "catch":
            # Select ball and throw
            actions = ["SELECT", "A", "A"]
        
        return actions
    
    def plan_heal(self) -> List[str]:
        """Plan healing action sequence"""
        actions = []
        
        # Need to find Pokemon Center
        # This is simplified - in real game would need pathfinding
        if self.state.player.location != "pokemon_center":
            actions = ["START", "RIGHT", "A"]  # Open menu, go to town map (simplified)
        else:
            # At Pokemon Center
            actions = ["A", "A", "A"]  # Talk to nurse, confirm healing
        
        return actions
    
    def plan_explore(self, direction: str) -> List[str]:
        """Plan exploration movement"""
        valid_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        if direction not in valid_directions:
            direction = "DOWN"  # Default
        
        return [direction]
    
    def plan_catch_sequence(self) -> List[str]:
        """Plan catch sequence"""
        # Navigate to ball, select, throw
        return ["SELECT", "A", "A"]
    
    def get_action_plan(self, decision: Dict) -> List[str]:
        """Get action plan from decision"""
        action_type = decision.get("action", "idle")
        
        if action_type == "attack":
            return self.plan_battle(decision)
        elif action_type == "run":
            return self.plan_battle(decision)
        elif action_type == "catch":
            return self.plan_catch_sequence()
        elif action_type == "heal":
            return self.plan_heal()
        elif action_type == "move":
            return self.plan_explore(decision.get("direction", "DOWN"))
        elif action_type == "idle":
            return []
        else:
            return []
    
    def execute_plan(self, emulator, plan: List[str]) -> Dict[str, Any]:
        """Execute action plan on emulator"""
        from pyboy.utils import WindowEvent
        
        executed = []
        
        for button in plan:
            if button in self.BUTTON_MAP:
                # Press button
                event = getattr(WindowEvent, f"PRESS_{self.BUTTON_MAP[button]}")
                emulator.send_input(event)
                emulator.tick()
                
                # Release button
                release_event = getattr(WindowEvent, f"RELEASE_{self.BUTTON_MAP[button]}")
                emulator.send_input(release_event)
                
                executed.append(button)
                self.state.frame_count += 1
        
        return {
            "success": True,
            "executed": executed,
            "frame": self.state.frame_count
        }


def main():
    parser = argparse.ArgumentParser(description="Game State Manager for Pokemon")
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--decide", help="Make decision for mode (explore/grind/battle)")
    parser.add_argument("--plan", help="Plan action (attack/run/heal)")
    parser.add_argument("--goal", help="Set session goal")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--mode", choices=["idle", "exploring", "grinding"], default="idle")
    
    args = parser.parse_args()
    
    if not Path(args.rom).exists():
        print(f"ERROR: ROM not found: {args.rom}")
        sys.exit(1)
    
    # Import scanner
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from memory_scanner import GameMemoryScanner
    except ImportError:
        print("ERROR: Could not import memory_scanner")
        sys.exit(1)
    
    # Initialize emulator
    print(f"Loading ROM: {args.rom}")
    emulator = PyBoy(args.rom, window="null")
    
    # Create scanner
    scanner = GameMemoryScanner(emulator)
    
    # Create game state
    game_state = GameState()
    game_state.mode = args.mode
    
    if args.goal:
        game_state.current_goal = args.goal
    
    # Update from memory
    game_state.update_from_scanner(scanner)
    
    # Create decision engine and action planner
    decision_engine = DecisionEngine(game_state)
    action_planner = ActionPlanner(game_state)
    
    results = {}
    
    if args.status:
        # Show current status
        results = game_state.to_dict()
        results["summary"] = game_state.get_action_summary()
    
    if args.decide:
        # Make a decision
        game_state.mode = args.decide
        decision_engine = DecisionEngine(game_state)  # Re-create with new mode
        decision = decision_engine.make_decision()
        results["decision"] = decision
        
        # Plan action
        plan = action_planner.get_action_plan(decision)
        results["plan"] = plan
    
    if args.plan:
        # Plan specific action
        action = {"action": args.plan}
        plan = action_planner.get_action_plan(action)
        results["planned_actions"] = plan
    
    if not results:
        # Default: full status
        results = game_state.to_dict()
        results["summary"] = game_state.get_action_summary()
    
    # Output
    output = json.dumps(results, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)
    
    emulator.stop()


if __name__ == "__main__":
    main()