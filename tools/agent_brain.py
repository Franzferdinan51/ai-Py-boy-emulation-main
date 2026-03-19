"""
AI GameBoy Agent Brain
Advanced decision engine for Game Boy emulation control
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

# ============================================================================
# TYPE EFFECTIVENESS CHART (Pokemon-style)
# ============================================================================

TYPE_EFFECTIVENESS = {
    ('normal', 'rock'): 0.5, ('normal', 'ghost'): 0, ('normal', 'steel'): 0.5,
    ('fire', 'fire'): 0.5, ('fire', 'water'): 0.5, ('fire', 'grass'): 2,
    ('fire', 'ice'): 2, ('fire', 'bug'): 2, ('fire', 'rock'): 0.5, ('fire', 'dragon'): 0.5,
    ('fire', 'steel'): 2,
    ('water', 'fire'): 2, ('water', 'water'): 0.5, ('water', 'grass'): 0.5,
    ('water', 'ground'): 2, ('water', 'rock'): 2, ('water', 'dragon'): 0.5,
    ('grass', 'fire'): 0.5, ('grass', 'water'): 2, ('grass', 'grass'): 0.5,
    ('grass', 'poison'): 0.5, ('grass', 'ground'): 2, ('grass', 'flying'): 0.5,
    ('grass', 'bug'): 0.5, ('grass', 'rock'): 2, ('grass', 'dragon'): 0.5,
    ('electric', 'water'): 2, ('electric', 'grass'): 0.5, ('electric', 'ground'): 0,
    ('electric', 'flying'): 2, ('electric', 'dragon'): 0.5,
}

def get_effectiveness(attack_type: str, defense_type: str) -> float:
    """Get type effectiveness multiplier"""
    return TYPE_EFFECTIVENESS.get((attack_type.lower(), defense_type.lower()), 1.0)

# ============================================================================
# GAME MEMORY ADDRESSES (Pokemon Red/Blue)
# ============================================================================

MEMORY_ADDRESSES = {
    'player_x': 0xD062,
    'player_y': 0xD063,
    'map_id': 0xD35E,
    'money': 0xD6F5,
    'party_count': 0xD163,
    'player_hp': 0xD16C,
    'player_max_hp': 0xD16D,
    'enemy_hp': 0xCFE6,
    'battle_state': 0xD056,
    'tileset': 0xD05E,
}

@dataclass
class GameMemory:
    """Snapshot of game memory state"""
    player_x: int = 0
    player_y: int = 0
    map_id: int = 0
    money: int = 0
    party_count: int = 0
    player_hp: int = 0
    player_max_hp: int = 100
    enemy_hp: int = 0
    battle_state: int = 0
    raw: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'GameMemory':
        """Create from raw memory dict"""
        mem = cls()
        for key, addr in MEMORY_ADDRESSES.items():
            val = data.get(addr, 0)
            setattr(mem, key, val)
            mem.raw[key] = val
        mem.money = (data.get(0xD6F5, 0) << 16) | (data.get(0xD6F6, 0) << 8) | data.get(0xD6F7, 0)
        return mem

# ============================================================================
# ACTION TYPES
# ============================================================================

class ActionType(Enum):
    NONE = "none"
    MOVE_UP = "up"
    MOVE_DOWN = "down"
    MOVE_LEFT = "left"
    MOVE_RIGHT = "right"
    BUTTON_A = "A"
    BUTTON_B = "B"
    BUTTON_START = "START"
    BUTTON_SELECT = "SELECT"
    INTERACT = "interact"
    OPEN_MENU = "menu"
    CLOSE_MENU = "close"
    WAIT = "wait"

@dataclass
class Action:
    action_type: ActionType
    duration_ms: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# GAME STATE
# ============================================================================

class GameState(Enum):
    EXPLORING = "exploring"
    BATTLE = "battle"
    MENU = "menu"
    DIALOG = "dialog"
    CUTSCENE = "cutscene"
    UNKNOWN = "unknown"

@dataclass
class AgentState:
    """Current agent state"""
    game_state: GameState = GameState.UNKNOWN
    position: Tuple[int, int] = (0, 0)
    explored_maps: set = field(default_factory=set)
    visited_cells: set = field(default_factory=set)
    prior_actions: List[Action] = field(default_factory=list)
    cooldowns: Dict[str, int] = field(default_factory=dict)
    last_battle_result: Optional[str] = None
    confidence: float = 0.5

# ============================================================================
# MEMORY SCANNER
# ============================================================================

class MemoryScanner:
    """Reads and normalizes game memory"""
    
    def __init__(self, pyboy_instance=None):
        self.pyboy = pyboy_instance
        self.cache = {}
        
    def read_byte(self, address: int) -> int:
        """Read a single byte from memory"""
        if self.pyboy:
            return self.pyboy.memory[address]
        return self.cache.get(address, 0)
    
    def read_range(self, start: int, length: int) -> List[int]:
        """Read a range of bytes"""
        return [self.read_byte(start + i) for i in range(length)]
    
    def scan_party(self) -> List[Dict]:
        """Scan party Pokemon data"""
        party = []
        party_count = self.read_byte(MEMORY_ADDRESSES['party_count'])
        
        for i in range(party_count):
            offset = 0xD163 + (i * 44)
            species = self.read_byte(offset)
            hp = self.read_byte(offset + 1) | (self.read_byte(offset + 2) << 8)
            max_hp = self.read_byte(offset + 3) | (self.read_byte(offset + 4) << 8)
            level = self.read_byte(offset + 8)
            
            party.append({
                'slot': i + 1,
                'species_id': species,
                'hp': hp,
                'max_hp': max_hp,
                'level': level,
                'hp_percent': (hp / max_hp * 100) if max_hp > 0 else 0
            })
        
        return party
    
    def scan_inventory(self) -> List[Dict]:
        """Scan inventory items"""
        items = []
        bag_start = 0xD6E5
        for i in range(20):
            item_id = self.read_byte(bag_start + (i * 2))
            quantity = self.read_byte(bag_start + (i * 2) + 1)
            if item_id > 0:
                items.append({'id': item_id, 'quantity': quantity})
        return items
    
    def get_game_state(self) -> GameState:
        """Determine current game state"""
        battle = self.read_byte(MEMORY_ADDRESSES['battle_state'])
        if battle > 0:
            return GameState.BATTLE
        
        # Check for menu
        menu_flag = self.read_byte(0xD735)
        if menu_flag:
            return GameState.MENU
            
        return GameState.EXPLORING

# ============================================================================
# EXPLORATION ALGORITHM
# ============================================================================

class ExplorationAlgorithm:
    """Handles world exploration with flood fill and pathfinding"""
    
    def __init__(self, map_width: int = 256, map_height: int = 256):
        self.map_width = map_width
        self.map_height = map_height
        self.reachable_cache = {}
        self.collision_map = set()
        
    def add_collision(self, x: int, y: int):
        """Mark cell as impassable"""
        self.collision_map.add((x, y))
    
    def flood_fill(self, start_x: int, start_y: int, visited: set = None) -> set:
        """Find all reachable cells from start position"""
        if visited is None:
            visited = set()
            
        stack = [(start_x, start_y)]
        reachable = set()
        
        while stack:
            x, y = stack.pop()
            if (x, y) in visited or (x, y) in self.collision_map:
                continue
                
            visited.add((x, y))
            reachable.add((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if (nx, ny) not in visited:
                        stack.append((nx, ny))
        
        return reachable
    
    def find_nearest_unexplored(self, current: Tuple[int, int], explored: set) -> Optional[Tuple[int, int]]:
        """BFS to find nearest unexplored cell"""
        if len(explored) == 0:
            return None
            
        queue = [(current, 0)]
        visited = {current}
        
        while queue:
            (x, y), dist = queue.pop(0)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and (nx, ny) not in self.collision_map:
                    visited.add((nx, ny))
                    if (nx, ny) not in explored:
                        return (nx, ny), dist
                    queue.append(((nx, ny), dist + 1))
        
        return None
    
    def astar_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding from start to goal"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            open_set.discard(current)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in self.collision_map:
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_set.add(neighbor)
        
        return []

# ============================================================================
# BATTLE AI
# ============================================================================

class BattleAI:
    """AI for Pokemon-style battles"""
    
    def __init__(self):
        self.type_chart = TYPE_EFFECTIVENESS
        
        # Move database (simplified)
        self.moves = {
            1: {'name': 'Pound', 'type': 'normal', 'power': 40, 'accuracy': 100},
            2: {'name': 'Karate Chop', 'type': 'fighting', 'power': 50, 'accuracy': 100},
            3: {'name': 'Ember', 'type': 'fire', 'power': 40, 'accuracy': 100},
            4: {'name': 'Water Gun', 'type': 'water', 'power': 40, 'accuracy': 100},
            5: {'name': 'Vine Whip', 'type': 'grass', 'power': 45, 'accuracy': 100},
            10: {'name': 'Gust', 'type': 'flying', 'power': 40, 'accuracy': 100},
            11: {'name': 'Tackle', 'type': 'normal', 'power': 50, 'accuracy': 100},
            33: {'name': 'Take Down', 'type': 'normal', 'power': 90, 'accuracy': 85},
            36: {'name': 'Thief', 'type': 'dark', 'power': 60, 'accuracy': 100},
            38: {'name': 'Fire Blast', 'type': 'fire', 'power': 110, 'accuracy': 85},
        }
        
    def calculate_damage(self, move_power: int, attacker_level: int, 
                        attack_stat: int, defense_stat: int,
                        effectiveness: float = 1.0) -> int:
        """Calculate damage using Pokemon damage formula"""
        if move_power == 0:
            return 0
            
        base = ((2 * attacker_level / 5 + 2) * move_power * attack_stat / defense_stat) / 50 + 2
        modifier = effectiveness
        return int(base * modifier)
    
    def rank_moves(self, my_pokemon: Dict, enemy_pokemon: Dict) -> List[Tuple[int, float]]:
        """Rank available moves by expected damage"""
        rankings = []
        
        for move_id, move_data in self.moves.items():
            # Calculate effectiveness
            effectiveness = get_effectiveness(
                move_data['type'],
                enemy_pokemon.get('type', 'normal')
            )
            
            # Skip if no effect
            if effectiveness == 0:
                continue
                
            # Calculate damage
            damage = self.calculate_damage(
                move_data['power'],
                my_pokemon.get('level', 5),
                my_pokemon.get('attack', 10),
                enemy_pokemon.get('defense', 10),
                effectiveness
            )
            
            # Score = damage * accuracy * effectiveness
            accuracy = move_data['accuracy'] / 100
            score = damage * accuracy * effectiveness
            
            # Bonus for STAB (same type attack bonus)
            if move_data['type'] == my_pokemon.get('type', 'normal'):
                score *= 1.5
                
            rankings.append((move_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def should_switch(self, my_party: List[Dict], enemy_pokemon: Dict) -> Optional[int]:
        """Decide if switching is better than attacking"""
        if len(my_party) <= 1:
            return None
            
        current = my_party[0]
        current_hp_percent = current.get('hp_percent', 100)
        
        # Switch if current Pokemon is low HP
        if current_hp_percent < 30:
            for i, pokemon in enumerate(my_party[1:], 1):
                if pokemon.get('hp_percent', 100) > 50:
                    return i
        
        return None
    
    def get_best_action(self, my_party: List[Dict], enemy: Dict) -> str:
        """Get the best battle action"""
        if not my_party:
            return 'run'
            
        current = my_party[0]
        
        # Check if should switch
        switch_to = self.should_switch(my_party, enemy)
        if switch_to is not None:
            return f'switch_{switch_to}'
        
        # Rank moves
        rankings = self.rank_moves(current, enemy)
        
        if rankings:
            best_move = rankings[0][0]
            return f'move_{best_move}'
        
        return 'struggle'

# ============================================================================
# DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """Main AI decision engine combining all systems"""
    
    def __init__(self):
        self.scanner = MemoryScanner()
        self.explorer = ExplorationAlgorithm()
        self.battle_ai = BattleAI()
        self.state = AgentState()
        self.state_manager = StateManager()
        
    def observe(self, memory: GameMemory) -> GameMemory:
        """Update internal state from memory observation"""
        self.state.position = (memory.player_x, memory.player_y)
        self.state.game_state = self.scanner.get_game_state()
        
        # Track exploration
        self.state.visited_cells.add((memory.player_x, memory.player_y))
        
        return memory
    
    def decide(self) -> List[Action]:
        """Make decision based on current state"""
        if self.state.game_state == GameState.BATTLE:
            return self._battle_decision()
        elif self.state.game_state == GameState.MENU:
            return [Action(ActionType.CLOSE_MENU)]
        else:
            return self._exploration_decision()
    
    def _battle_decision(self) -> List[Action]:
        """Decide battle action"""
        party = self.scanner.scan_party()
        if not party:
            return [Action(ActionType.BUTTON_B)]
        
        enemy = {'type': 'normal', 'defense': 10, 'hp': self.scanner.read_byte(0xCFE6)}
        best_action = self.battle_ai.get_best_action(party, enemy)
        
        if best_action.startswith('switch_'):
            slot = best_action.split('_')[1]
            return [
                Action(ActionType.BUTTON_A),  # Open party
                Action(ActionType.BUTTON_A),  # Select
            ]
        elif best_action.startswith('move_'):
            return [Action(ActionType.BUTTON_A)]
        else:
            return [Action(ActionType.BUTTON_B)]  # Try to run
    
    def _exploration_decision(self) -> List[Action]:
        """Decide exploration action"""
        # Check cooldowns
        if self.state.cooldowns.get('movement', 0) > 0:
            return [Action(ActionType.WAIT, duration_ms=100)]
        
        current = self.state.position
        
        # Find target
        target = self.explorer.find_nearest_unexplored(
            current, 
            self.state.visited_cells
        )
        
        if target is None:
            return [Action(ActionType.WAIT)]
        
        goal, dist = target
        
        # Pathfind to target
        path = self.explorer.astar_path(current, goal)
        
        if not path or len(path) < 2:
            return [Action(ActionType.WAIT)]
        
        next_step = path[1]
        direction = self._get_direction(current, next_step)
        
        return [Action(direction, duration_ms=200)]
    
    def _get_direction(self, current: Tuple, next_pos: Tuple) -> ActionType:
        """Get direction action to move from current to next"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx > 0:
            return ActionType.MOVE_RIGHT
        elif dx < 0:
            return ActionType.MOVE_LEFT
        elif dy > 0:
            return ActionType.MOVE_DOWN
        elif dy < 0:
            return ActionType.MOVE_UP
        else:
            return ActionType.NONE

# ============================================================================
# STATE MANAGER
# ============================================================================

class StateManager:
    """Centralized state transitions and lifecycle"""
    
    def __init__(self):
        self.current_state: Dict[str, Any] = {}
        self.prior_state: Dict[str, Any] = {}
        self.transition_log: List[Dict] = []
        self.action_history: List[Dict] = []
        
    def update(self, new_state: Dict[str, Any], action_taken: Optional[Action] = None):
        """Update state with transition tracking"""
        self.prior_state = self.current_state.copy()
        self.current_state = new_state.copy()
        
        if action_taken:
            self.action_history.append({
                'action': action_taken.action_type.value,
                'state': self.current_state.copy(),
                'prior': self.prior_state.copy()
            })
        
        # Log transitions
        if self.current_state != self.prior_state:
            self.transition_log.append({
                'from': self.prior_state,
                'to': self.current_state,
                'action': action_taken.action_type.value if action_taken else None
            })
    
    def get_cooldown(self, action_type: str, cooldown_ms: int = 500) -> bool:
        """Check if action is on cooldown"""
        if not self.action_history:
            return False
            
        last_same = None
        for entry in reversed(self.action_history):
            if entry['action'] == action_type:
                last_same = entry
                break
        
        if last_same is None:
            return False
            
        return True
    
    def should_retry(self, max_attempts: int = 3) -> bool:
        """Determine if failed action should retry"""
        if not self.action_history:
            return True
            
        recent_same = sum(
            1 for e in self.action_history[-10:]
            if e.get('success', True) == False
        )
        
        return recent_same < max_attempts

# ============================================================================
# MAIN AGENT BRAIN CLASS
# ============================================================================

class AgentBrain:
    """
    Main AI Brain for Game Boy Emulation
    Combines all subsystems for autonomous control
    """
    
    def __init__(self, pyboy_instance=None):
        self.pyboy = pyboy_instance
        self.decision_engine = DecisionEngine()
        self.scanner = MemoryScanner(pyboy_instance)
        self.state_manager = StateManager()
        
    def tick(self) -> List[str]:
        """
        Main tick - called every game frame
        Returns list of button presses to execute
        """
        # 1. Read memory
        memory_data = {}
        if self.pyboy:
            for addr in MEMORY_ADDRESSES.values():
                memory_data[addr] = self.pyboy.memory[addr]
        
        # 2. Create memory snapshot
        memory = GameMemory.from_dict(memory_data)
        
        # 3. Observe
        memory = self.decision_engine.observe(memory)
        
        # 4. Decide
        actions = self.decision_engine.decide()
        
        # 5. Execute
        buttons = []
        for action in actions:
            buttons.append(action.action_type.value)
            self.state_manager.action_history.append({
                'action': action.action_type.value,
                'metadata': action.metadata
            })
        
        return buttons
    
    def get_status(self) -> Dict:
        """Get current agent status"""
        return {
            'state': self.decision_engine.state.game_state.value,
            'position': self.decision_engine.state.position,
            'explored_cells': len(self.decision_engine.state.visited_cells),
            'actions_today': len(self.state_manager.action_history)
        }

if __name__ == '__main__':
    # Test
    brain = AgentBrain()
    print("Agent Brain initialized")
    print(brain.get_status())
