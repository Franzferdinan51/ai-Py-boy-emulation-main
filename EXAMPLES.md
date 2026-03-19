# 📖 EXAMPLES.md - Example Agent Prompts and Sessions

**Real examples of AI agents playing Game Boy games**

---

## Example Agent Prompts

### Starting a New Game

**Prompt:**
> "Load Pokemon Red and start a new game. Get the starter Pokemon from Professor Oak. Name the Pokemon 'Quackster'."

**Agent Actions:**
```json
// 1. Load ROM
{"tool": "emulator_load_rom", "args": {"rom_path": "/path/to/pokemon-red.gb"}}

// 2. Press START to begin
{"tool": "emulator_press_sequence", "args": {"sequence": "START"}}

// 3. Wait for title screen
{"tool": "emulator_tick", "args": {"frames": 30}}

// 4. Press A to select NEW GAME
{"tool": "emulator_press_button", "args": {"button": "A"}}

// 5. Wait through intro
{"tool": "emulator_tick", "args": {"frames": 100}}

// 6. Navigate to name entry
{"tool": "emulator_press_sequence", "args": {"sequence": "A W W W A"}}

// 7. Enter name "RED" (default)
// ... navigate through game

// 8. Save progress
{"tool": "save_game_state", "args": {"save_name": "start-game"}}
```

---

### Exploring a New Area

**Prompt:**
> "Explore the area around Pallet Town and find the Pokemon Center."

**Agent Actions:**
```json
// 1. Get current position
{"tool": "get_player_position", "args": {}}
// Response: {"x": 5, "y": 10, "map_id": 38}

// 2. Get screen for vision
{"tool": "get_screen_base64", "args": {"include_base64": true}}
// Response: Shows houses and sign

// 3. Start exploration
{"tool": "auto_explore", "args": {"steps": 20}}

// 4. Check if Pokemon Center found
{"tool": "get_map_location", "args": {}}
// Response: {"map_id": 1, "map_name": "Viridian City"}

// 5. Look for Pokemon Center
{"tool": "get_screen_base64", "args": {"include_base64": true}}
// Vision: See Pokemon Center building

// 6. Enter Pokemon Center
{"tool": "emulator_press_sequence", "args": {"sequence": "RIGHT RIGHT A"}}

// 7. Heal party
{"tool": "emulator_press_sequence", "args": {"sequence": "A W30 A"}}
```

---

### Battling a Wild Pokemon

**Prompt:**
> "A wild Pidgey appeared! Battle it and catch it if possible."

**Agent Actions:**
```json
// 1. Get party info
{"tool": "get_party_info", "args": {}}
// Response: Charmander Lv5, HP 20/20

// 2. Get screen for battle context
{"tool": "get_screen_base64", "args": {"include_base64": true}}
// Vision: See Pidgey, level 5

// 3. Select FIGHT
{"tool": "emulator_press_button", "args": {"button": "A"}}

// 4. Wait for menu
{"tool": "emulator_tick", "args": {"frames": 10}}

// 5. Select move (Ember - fire type advantage)
{"tool": "emulator_press_sequence", "args": {"sequence": "DOWN A"}}

// 6. Wait for attack animation
{"tool": "emulator_tick", "args": {"frames": 30}}

// 7. Check result via screen
{"tool": "get_screen_base64", "args": {"include_base64": true}}
// Vision: Pidgey fainted!

// 8. Save after battle
{"tool": "save_game_state", "args": {"save_name": "after-first-battle"}}
```

---

### Healing at Pokemon Center

**Prompt:**
> "My Charmander is low on HP. Find a Pokemon Center and heal up."

**Agent Actions:**
```json
// 1. Check party HP
{"tool": "get_party_info", "args": {}}
// Response: Charmander HP 5/20 - CRITICAL!

// 2. Get current location
{"tool": "get_map_location", "args": {}}
// Response: map_id=2 (Route 1)

// 3. Travel to nearest city
{"tool": "emulator_press_sequence", "args": {"sequence": "RIGHT RIGHT RIGHT RIGHT"}}

// 4. Enter Viridian City
{"tool": "get_map_location", "args": {}}
// Response: map_id=1 (Viridian City)

// 5. Find Pokemon Center
{"tool": "auto_explore", "args": {"steps": 10}}

// 6. Enter Center
{"tool": "emulator_press_button", "args": {"button": "A"}}

// 7. Talk to nurse
{"tool": "emulator_press_button", "args": {"button": "A"}}

// 8. Wait for healing
{"tool": "emulator_tick", "args": {"frames": 60}}

// 9. Verify healed
{"tool": "get_party_info", "args": {}}
// Response: Charmander HP 20/20
```

---

## Example Game Sessions

### Session 1: Beginning the Adventure

**Goal:** Complete the intro and get starter Pokemon

**Session Log:**
```json
{
  "session_id": "session_001",
  "goal": "Complete Pokemon Red intro",
  "start_time": "2026-03-19T10:00:00",
  
  "actions": [
    {"time": "10:00:01", "action": "emulator_load_rom", "args": {"rom_path": "pokemon-red.gb"}},
    {"time": "10:00:02", "action": "emulator_press_sequence", "args": {"sequence": "START"}},
    {"time": "10:00:05", "action": "emulator_press_button", "args": {"button": "A"}},
    {"time": "10:01:30", "action": "get_screen_base64", "result": "See Oak in lab"},
    {"time": "10:01:35", "action": "emulator_press_sequence", "args": {"sequence": "RIGHT A"}},
    {"time": "10:02:00", "action": "get_party_info", "result": "Charmander received!"},
    {"time": "10:02:01", "action": "save_game_state", "args": {"save_name": "got-starter"}}
  ],
  
  "outcome": "SUCCESS - Received Charmander",
  "duration": "2 minutes"
}
```

---

### Session 2: First Battle

**Goal:** Defeat first wild Pokemon

**Session Log:**
```json
{
  "session_id": "session_002",
  "goal": "Win first battle",
  "start_time": "2026-03-19T10:10:00",
  "parent_session": "session_001",
  
  "actions": [
    {"time": "10:10:01", "action": "load_game_state", "args": {"save_name": "got-starter"}},
    {"time": "10:10:05", "action": "emulator_press_sequence", "args": {"sequence": "DOWN DOWN DOWN"}},
    {"time": "10:10:15", "action": "get_map_location", "result": "Route 1"},
    {"time": "10:10:20", "action": "emulator_press_button", "args": {"button": "A"}, "note": "Encounter Pidgey!"},
    {"time": "10:10:30", "action": "emulator_press_button", "args": {"button": "A"}, "note": "Fight"},
    {"time": "10:10:35", "action": "emulator_press_sequence", "args": {"sequence": "DOWN A"}, "note": "Ember"},
    {"time": "10:11:00", "action": "get_screen_base64", "result": "Victory!"},
    {"time": "10:11:01", "action": "save_game_state", "args": {"save_name": "won-battle"}}
  ],
  
  "outcome": "SUCCESS - Won battle, gained XP",
  "duration": "1 minute",
  "rewards": {"exp": 25, "money": 100}
}
```

---

### Session 3: Training Level

**Goal:** Grind Charmander to level 10

**Session Log:**
```json
{
  "session_id": "session_003",
  "goal": "Grind to level 10",
  "start_time": "2026-03-19T10:15:00",
  
  "actions": [
    {"time": "10:15:01", "action": "load_game_state", "args": {"save_name": "won-battle"}},
    {"time": "10:15:05", "action": "get_party_info", "result": "Charmander Lv5"},
    
    // Auto-grind for 20 battles
    {"time": "10:15:10", "action": "auto_grind", "args": {"max_battles": 20}},
    
    {"time": "10:25:00", "action": "get_party_info", "result": "Charmander Lv9"},
    {"time": "10:25:05", "action": "save_game_state", "args": {"save_name": "level-9"}},
    
    // One more battle to hit 10
    {"time": "10:25:10", "action": "auto_battle", "args": {"max_moves": 5}},
    
    {"time": "10:26:00", "action": "get_party_info", "result": "Charmander Lv10!"},
    {"time": "10:26:01", "action": "save_game_state", "args": {"save_name": "level-10"}}
  ],
  
  "outcome": "SUCCESS - Reached level 10",
  "duration": "11 minutes",
  "battles": 21
}
```

---

## Decision Log Examples

### Example: Battle Decision Log

```
DECISION: Battle Wild Pidgey (Lv5)
==================================
Time: 10:10:20

STATE:
  - My Pokemon: Charmander Lv5, HP 20/20
  - Enemy: Pidgey Lv5 (Normal/Flying)
  - Money: $300
  - Location: Route 1

VISION ANALYSIS:
  - Pidgey looks weak
  - I have type advantage (Fire > Normal)

DECISION:
  → Use Ember (fire move, STAB bonus)
  → Expected damage: 10-12 HP
  → Enemy HP: ~18 - can defeat in 2 hits

ACTION TAKEN:
  emulator_press_sequence("A DOWN A")

RESULT:
  ✓ Hit for 12 damage!
  ✓ Enemy fainted!
  ✓ Gained 25 XP
  ✓ Level up to Lv6!
```

---

### Example: Exploration Decision Log

```
DECISION: Find Pokemon Center (Viridian City)
==============================================
Time: 10:30:00

STATE:
  - My Pokemon: Charmander Lv10, HP 5/20 (CRITICAL!)
  - Location: Viridian City (map_id=1)
  - Last save: 30 minutes ago

PROBLEM:
  - HP critical! Need healing ASAP
  - In city - Pokemon Center should be nearby

VISION:
  - See building with red roof
  - "P" symbol visible
  - That's the Pokemon Center!

DECISION:
  → Walk directly to building
  → Don't explore - time is critical

ACTIONS:
  emulator_press_sequence("RIGHT RIGHT RIGHT A")
  → Enter Pokemon Center
  
  emulator_press_button("A")
  → Talk to nurse

HEALING:
  Wait 60 frames...
  → Charmander HP: 20/20 ✓

DECISION CORRECT? YES
- Found center quickly
- Healed in time
- No fainting occurred
```

---

### Example: Shop Decision Log

```
DECISION: Buy supplies at PokeMart
==================================
Time: 11:00:00

STATE:
  - Money: $2500
  - Inventory: 3 Potions, 5 Poke Balls
  - Party: Charmander Lv12, HP 20/20

NEEDS:
  - More potions (running low)
  - More balls (for catching)
  - Possible escape rope (for caves)

DECISION TREE:
  IF money > 2000:
    → Buy Potions: 3 x $200 = $600
    → Buy Balls: 10 x $200 = $2000
    → Total: $2600 (too much!)
  
  ADJUSTED:
    → Buy Potions: 5 x $200 = $1000
    → Buy Balls: 5 x $200 = $1000
    → Total: $2000
    → Remaining: $500 (safe buffer)

ACTIONS:
  emulator_press_sequence("A") → Enter shop
  emulator_press_sequence("RIGHT A") → Buy
  emulator_press_sequence("DOWN A x5") → Buy 5 potions
  emulator_press_sequence("DOWN A x5") → Buy 5 balls
  emulator_press_button("B") → Exit shop

RESULT:
  ✓ Purchased: 5 Potions, 5 Balls
  ✓ Money remaining: $1500
```

---

## Quick Reference Prompts

### Combat
- "Battle the current Pokemon"
- "Use [move name]"
- "Run from battle"
- "Use an item on first Pokemon"

### Exploration
- "Explore this area"
- "Find the nearest Pokemon Center"
- "Go to [location name]"
- "Walk around for wild encounters"

### Inventory
- "Check my party"
- "Check my items"
- "How much money do I have?"
- "Use [item]"

### Save/Load
- "Save game"
- "Load last save"
- "Save before this battle"

### Game State
- "What map am I on?"
- "Check my Pokemon levels"
- "Where am I positioned?"

---

## Related Documentation

- [API_REFERENCE.md](guides/API_REFERENCE.md) - All MCP endpoints
- [DECISION_TREE.md](guides/DECISION_TREE.md) - How agents make decisions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [README.md](README.md) - Main documentation