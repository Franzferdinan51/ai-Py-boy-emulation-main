# DuckBot Pokemon Enhancement Ideas
> Top 10 improvements inspired by leading AI gaming projects

## Research Sources
- **gpt-play-pokemon-firered** - LLM-powered autonomous agent with web dashboard
- **PokemonRedExperiments** - Reinforcement learning with PyBoy
- **poke.AI** - CNN object detection + SLAM mapping + Deep Q-Learning battles
- **PyBoy API** - Memory scanning, hooks, savestate management

---

## Priority 1: Core Enhancements (Do First)

### 1. 🗺️ Smart Minimap & World Mapping
**Priority: HIGH** | **Effort: Medium**

Based on **poke.AI**'s SLAM-style mapping approach.

**What it does:**
- Tracks player position on a global map using tile coordinates
- Marks discovered locations (Pokecenters, Pokemarts, Gyms, NPCs)
- Shows fog of war for unexplored areas

**Implementation:**
```python
# minimap_tracker.py
class MinimapTracker:
    def __init__(self):
        self.visited_tiles = set()
        self.object_locations = {}  # tile_coords -> object_type
        self.map_width = 144  # tiles
        self.map_height = 160  # tiles
    
    def get_player_tile(self, pyboy):
        # Read player position from memory
        x = pyboy.memory[0xD362]  # Player X position
        y = pyboy.memory[0xD361]  # Player Y position
        return (x, y)
    
    def update(self, pyboy):
        player_tile = self.get_player_tile(pyboy)
        self.visited_tiles.add(player_tile)
        
        # Detect nearby objects using screen analysis
        # Mark Pokecenters, Gyms, etc.
```

**Files to create:**
- `ai-game-assistant/minimap_tracker.py`

---

### 2. 🎯 Improved Battle AI with Type Matching
**Priority: HIGH** | **Effort: Medium**

Based on **poke.AI**'s Deep Q-Learning approach, simplified for LLM.

**What it does:**
- Analyzes enemy Pokemon type before choosing moves
- Recommends super-effective attacks
- Knows type advantages (Fire < Water, etc.)

**Implementation:**
```python
# battle_analyzer.py
TYPE_CHART = {
    'fire': ['grass', 'ice', 'bug', 'steel'],
    'water': ['fire', 'ground', 'rock'],
    'grass': ['water', 'ground', 'rock'],
    'electric': ['water', 'flying'],
    # ... complete chart
}

def get_type_advantages(attack_type, defender_types):
    advantages = []
    for defender in defender_types:
        if attack_type in TYPE_CHART.get(defender, []):
            advantages.append('super-effective')
        elif defender in TYPE_CHART.get(attack_type, []):
            advantages.append('not-effective')
    return advantages

def analyze_battle_state(pyboy):
    # Read enemy Pokemon type from memory
    enemy_type1 = get_memory_type(pyboy, 0xD8C5)  # Enemy type 1
    enemy_type2 = get_memory_type(pyboy, 0xD8C6)  # Enemy type 2
    
    # Get your Pokemon's moves and their types
    your_moves = get_your_moves(pyboy)
    
    # Score each move
    scored_moves = []
    for move in your_moves:
        score = 0
        advantages = get_type_advantages(move.type, [enemy_type1, enemy_type2])
        if 'super-effective' in advantages:
            score += 2
        scored_moves.append((move, score))
    
    return sorted(scored_moves, key=lambda x: x[1], reverse=True)
```

---

### 3. 💾 Savestate Management with Auto-Recovery
**Priority: HIGH** | **Effort: Low**

Essential for long play sessions (based on gpt-play-pokemon-firered).

**What it does:**
- Auto-saves before risky actions (boss battles, elite four)
- Quick-load on failure
- Rotating backup system

**Implementation:**
```python
# savestate_manager.py
import os
import shutil
from datetime import datetime

class SavestateManager:
    def __init__(self, save_dir='saves'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_saves = 10
    
    def save(self, pyboy, name):
        filepath = os.path.join(self.save_dir, f'{name}.state')
        with open(filepath, 'wb') as f:
            pyboy.save_state(f)
        self._cleanup_old()
    
    def load(self, pyboy, name):
        filepath = os.path.join(self.save_dir, f'{name}.state')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                pyboy.load_state(f)
            return True
        return False
    
    def auto_backup(self, pyboy):
        # Save before major events
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save(pyboy, f'auto_{timestamp}')
```

---

### 4. 👁️ Vision-Based Object Detection
**Priority: HIGH** | **Effort: High**

Based on **poke.AI**'s CNN approach, but using LLM vision instead.

**What it does:**
- Uses [SELECT_VISION_MODEL] to analyze screen
- Detects NPCs, buildings, items on ground
- Provides natural language context for decision making

**Implementation:**
```python
# vision_analyzer.py
import base64
from io import BytesIO

def analyze_screen(pyboy, model):
    # Capture screen
    screen = pyboy.screen.image
    buffer = BytesIO()
    screen.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    prompt = """Analyze this Pokemon Red game screen. Identify:
    1. What building/landmark is visible?
    2. Are there any NPCs visible? What are they doing?
    3. Is there an item on the ground?
    4. What direction is the player facing?
    5. Any visible text/dialogue?"""
    
    response = model.images.edit(
        image=[image_b64],
        prompt=prompt
    )
    return response
```

---

## Priority 2: Feature Enhancements

### 5. 🌐 Live Web Dashboard
**Priority: MEDIUM** | **Effort: Medium**

Based on **gpt-play-pokemon-firered**.

**What it does:**
- Real-time game view
- Minimap visualization
- Inventory/party display
- Action history log

**Tech Stack:**
- FastAPI for backend
- WebSocket for live updates
- Simple HTML/CSS/JS frontend

```python
# dashboard_server.py
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()
websockets = set()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle messages
    finally:
        websockets.remove(websocket)

async def broadcast_game_state(state):
    for ws in websockets:
        await ws.send_json(state)
```

---

### 6. 📋 Quest/Objective Tracker
**Priority: MEDIUM** | **Effort: Low**

**What it does:**
- Tracks current quest/objectives
- Auto-updates based on game progress
- Reminds player of goals

**Implementation:**
```python
# objective_tracker.py
OBJECTIVES = {
    0: "Go to Professor Oak's lab",
    1: "Choose your starter Pokemon",
    2: "Defeat Brock at Pewter Gym",
    # ... etc
}

class ObjectiveTracker:
    def __init__(self):
        self.current_objective = 0
        self.completed = []
    
    def update(self, game_state):
        # Detect progress via memory
        if self.detect_oak_lab(game_state):
            self.current_objective = 1
        elif self.detect_starter_chosen(game_state):
            self.current_objective = 2
        # ... etc
```

---

### 7. 🎒 Smart Inventory Management
**Priority: MEDIUM** | **Effort: Low**

**What it does:**
- Tracks items in bag
- Suggests when to use items (potions, repels)
- Knows key items (Bike, Surf, Cut)

**Implementation:**
```python
def get_inventory(pyboy):
    # Read from Pokemon Red memory
    # Item bag starts at 0xD31E
    items = []
    for i in range(20):  # Max 20 items
        item_id = pyboy.memory[0xD31E + (i * 2)]
        if item_id != 0:
            quantity = pyboy.memory[0xD31E + (i * 2) + 1]
            items.append((ITEM_NAMES.get(item_id, f"Unknown_{item_id}"), quantity))
    return items
```

---

### 8. 🤖 LLM-Powered Decision Engine
**Priority: MEDIUM** | **Effort: High**

The core of **gpt-play-pokemon-firered** - use LLM for high-level decisions.

**What it does:**
- Analyzes game state context
- Makes strategic decisions
- Plans next objectives
- Learns from mistakes

**Implementation:**
```python
# decision_engine.py
SYSTEM_PROMPT = """You are an expert Pokemon Red player. You analyze the game state and decide the best next action.

Current game state:
- Location: {location}
- Party: {party}
- Inventory: {inventory}
- Objectives: {objectives}
- Recent actions: {history}

Choose ONE action from: MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, A_BUTTON, B_BUTTON, START, SELECT
Explain your reasoning."""

def decide(prompt_data, model):
    prompt = SYSTEM_PROMPT.format(**prompt_data)
    response = model.chat.completions.create(
        model="bailian/[SELECT_VISION_MODEL]",
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_action(response.choices[0].message.content)
```

---

## Priority 3: Advanced Features

### 9. 🔄 Automated Exploration (Frontier-Based)
**Priority: LOW** | **Effort: High**

Based on **poke.AI**'s frontier exploration algorithm.

**What it does:**
- Systematically explores new areas
- Prioritizes interesting locations (Gyms, Pokecenters)
- Avoids revisiting explored areas

**Implementation:**
```python
# frontier_explorer.py
from collections import deque

class FrontierExplorer:
    def __init__(self, map_width=144, map_height=160):
        self.explored = set()
        self.frontier = deque()
        self.map_width = map_width
        self.map_height = map_height
    
    def add_frontier_points(self, current_pos):
        x, y = current_pos
        # Add adjacent unexplored tiles to frontier
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            new_pos = (x + dx, y + dy)
            if (0 <= new_pos[0] < self.map_width and 
                0 <= new_pos[1] < self.map_height and
                new_pos not in self.explored):
                self.frontier.append(new_pos)
    
    def get_next_action(self, current_pos, target_pos):
        # Simple BFS to find path
        # Returns list of directions to take
        pass
```

---

### 10. 📊 Training Data Collection & Metrics
**Priority: LOW** | **Effort: Low**

**What it does:**
- Tracks win rates, deaths, time played
- Logs all decisions for later analysis
- Displays progress statistics

**Implementation:**
```python
# metrics_collector.py
import json
from datetime import datetime

class MetricsCollector:
    def __init__(self):
        self.data = {
            'battles_won': 0,
            'battles_lost': 0,
            'items_used': {},
            'locations_visited': set(),
            'time_played': 0,
            'actions': []
        }
    
    def log_action(self, action, result):
        self.data['actions'].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'result': result
        })
    
    def save(self, filepath='metrics.json'):
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
```

---

## Quick Win: Immediate Implementation

### Priority Checklist

| # | Enhancement | Status | Effort |
|---|-------------|--------|--------|
| 1 | Minimap & Mapping | ⬜ Not started | Medium |
| 2 | Battle AI | ⬜ Not started | Medium |
| 3 | Savestate Manager | ⬜ Not started | Low |
| 4 | Vision Analysis | ⬜ Not started | High |
| 5 | Web Dashboard | ⬜ Not started | Medium |
| 6 | Objective Tracker | ⬜ Not started | Low |
| 7 | Inventory Manager | ⬜ Not started | Low |
| 8 | Decision Engine | ⬜ Not started | High |
| 9 | Frontier Explorer | ⬜ Not started | High |
| 10 | Metrics Collector | ⬜ Not started | Low |

### Recommended Implementation Order

1. **#3 Savestate Manager** - Low effort, high value for safety
2. **#7 Inventory Manager** - Low effort, useful immediately  
3. **#6 Objective Tracker** - Low effort, keeps AI on track
4. **#2 Battle AI** - Medium effort, immediate gameplay improvement
5. **#1 Minimap** - Medium effort, essential for exploration
6. **#8 Decision Engine** - High effort, core brain upgrade
7. **#4 Vision Analysis** - High effort, when needed
8. **#5 Dashboard** - Nice to have
9. **#9 Frontier Explorer** - Advanced
10. **#10 Metrics** - Nice to have

---

## Memory Addresses Reference (Pokemon Red)

| Address | Description |
|---------|-------------|
| 0xD361 | Player Y position |
| 0xD362 | Player X position |
| 0xD31E | Item bag start |
| 0xD16B | Money (BCD, 3 bytes) |
| 0xD8C5 | Enemy Pokemon type 1 |
| 0xD8C6 | Enemy Pokemon type 2 |
| 0xD8B0 | Enemy Pokemon HP (2 bytes) |
| 0xCC26 | Your Pokemon HP (2 bytes) |
| 0xCC5D | Your Pokemon level |
| 0xD11F | Badges (8 bits) |

*Full memory map: https://datacrystal.tucex.com/wiki/Pokémon_Red/Blue:Memory_map*

---

## Credits

- **gpt-play-pokemon-firered** - Architecture inspiration
- **PokemonRedExperiments** - RL approach
- **poke.AI** - CNN + SLAM + DQN methods
- **PyBoy** - Emulator foundation

*Document created: March 2026*