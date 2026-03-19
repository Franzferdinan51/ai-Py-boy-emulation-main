# 🎮 DECISION_TREE.md - Agent Decision Making Guide

**How AI agents make decisions while playing Game Boy games**

---

## Overview

This document describes how autonomous AI agents analyze game state and make decisions to play Game Boy games effectively. The decision tree guides agents through a structured reasoning process.

---

## Core Decision Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT DECISION LOOP                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. GET STATE                                                │
│    └─► get_player_position()                                │
│    └─► get_party_info()                                     │
│    └─► get_money()                                          │
│    └─► get_map_location()                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GET VISION                                               │
│    └─► get_screen_base64() → [SELECT_VISION_MODEL] vision analysis     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ANALYZE CONTEXT                                         │
│    └─► What game? (Pokemon, Zelda, Mario, etc.)            │
│    └─► Current location? (town, route, cave, battle)       │
│    └─► Party status? (HP, levels, types)                    │
│    └─► Resources? (money, items, badges)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. DECIDE ACTION                                            │
│    └─► Select action based on game type & state           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ACT                                                      │
│    └─► emulator_press_sequence()                            │
│    └─► auto_battle() or auto_explore()                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SAVE (if needed)                                         │
│    └─► save_game_state() before risky actions              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                          [REPEAT]
```

---

## Decision Flowcharts

### Main Game State Detection

```
┌──────────────┐
│  GET SCREEN  │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────┐
│         WHAT IS ON SCREEN?                 │
└────────────────────────────────────────────┘
       │        │          │          │
   ┌───┴───┐    │    ┌────┴────┐    │   ┌───┴───┐
   │BATTLE │    │    │  MENU   │    │   │ OVER- │
   │ SCREEN│    │    │ SCREEN  │    │   │ WORLD │
   └───┬───┘    │    └────┬────┘    │   └───┬───┘
       │         │         │         │       │
       ▼         ▼         ▼         ▼       ▼
   [BATTLE]  [MENU]   [DIALOG]  [TITLE]  [EXPLORE]
```

---

### Battle Decision Tree

```
┌─────────────────────────────────────┐
│        IN BATTLE?                   │
│   (get_party_info shows combat)     │
└──────────────┬──────────────────────┘
               │
     ┌─────────┴─────────┐
     │                    │
     ▼                    ▼
┌─────────────┐    ┌──────────────────┐
│  Wild       │    │  Trainer Battle │
│  Encounter │    │                  │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       ▼                    ▼
┌─────────────────────────────────────┐
│      BATTLE DECISION TREE           │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 1. Check Enemy HP & Species         │
│    - Species ID → Type matchup      │
│    - Enemy HP → Can I win?         │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. Check My Party HP               │
│    - Current HP / Max HP ratio      │
│    - Any fainted Pokemon?          │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. Select Action                    │
│                                     │
│ IF enemy type > my type:            │
│   → Use highest damage move         │
│   → Or switch if possible           │
│                                     │
│ IF my HP < 30%:                     │
│   → Use heal/revive                 │
│   → Or run if wild battle           │
│                                     │
│ IF can win in 1-2 hits:             │
│   → Attack!                         │
│                                     │
│ IF enemy too strong:                │
│   → Use status moves                │
│   → Or attempt to run               │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. Execute & Learn                  │
│    - Press attack button            │
│    - Save state after battle        │
│    - Log result for future learning │
└─────────────────────────────────────┘
```

---

### Exploration Decision Tree

```
┌─────────────────────────────────────┐
│      OVERWORLD EXPLORATION          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Where am I?                        │
│  get_map_location() → map_id       │
│  - Pallet Town (38)                 │
│  - Viridian City (1)                │
│  - Route 1 (2)                     │
│  - Pewter City (3)                 │
│  - etc.                            │
└─────────────────────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│  TOWN/CITY     │  │  ROUTE/WILD      │
│  (has Mart,    │  │  (encounters,    │
│   Pokemon      │  │   items, NPCs)   │
│   Center)      │  │                  │
└───────┬────────┘  └────────┬─────────┘
        │                    │
        ▼                    ▼
┌─────────────────────────────────────┐
│       TOWN DECISIONS                │
│                                     │
│ IF need healing:                   │
│   → Go to Pokemon Center            │
│   → Use emulator_press_sequence    │
│     "DOWN DOWN DOWN A A A"         │
│                                     │
│ IF need items/money:                │
│   → Go to PokeMart                 │
│   → Buy potions, balls, etc.       │
│                                     │
│ IF done in town:                    │
│   → Exit to route                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       ROUTE DECISIONS               │
│                                     │
│ IF exploring:                       │
│   → Use auto_explore(steps=20)     │
│   → Check for items on ground      │
│   → Look for hidden areas          │
│                                     │
│ IF grinding XP:                     │
│   → Use auto_grind(target_level=X) │
│   → Find area with suitable mobs   │
│                                     │
│ IF stuck:                           │
│   → Use vision to see obstacles    │
│   → Try different direction         │
└─────────────────────────────────────┘
```

---

### Menu Navigation Decision Tree

```
┌─────────────────────────────────────┐
│       NEED TO OPEN MENU?            │
│         (START pressed)             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     MENU STRUCTURE (Pokemon)        │
│                                     │
│   POKÉMON  ─► Party management      │
│   BAG      ─► Items                │
│   POKEDEX  ─► Seen/caught info     │
│   SAVE     ─► Save game            │
│   OPTION   ─► Settings             │
│   EXIT     ─► Close menu           │
└─────────────────────────────────────┘
               │
     ┌─────────┴─────────┐
     │                  │
     ▼                  ▼
┌─────────────┐   ┌────────────────┐
│ Need Heal?  │   │ Need Item?     │
│ → POKEMON   │   │ → BAG          │
│ → Select    │   │ → Select item  │
│   heal mon  │   │ → Use item     │
└─────────────┘   └────────────────┘
```

---

## Example Decision Chains

### Chain 1: Start Pokemon Game

```
1. GET STATE: get_money() → $0
2. GET VISION: get_screen_base64() → see "NEW GAME" text
3. DECIDE: Press START to begin
4. ACT: emulator_press_sequence("START")
5. WAIT: W3
6. ACT: Press A to select "NEW GAME"
7. WAIT: W60 (dialogue)
8. GET VISION: Check player name screen
9. DECIDE: Enter name
10. ACT: Use press_sequence for each letter + START
11. SAVE: save_game_state("start-of-game")
12. LOG: session_set(key="story_progress", value="intro-complete")
```

### Chain 2: Battle Wild Pokemon

```
1. GET STATE: get_party_info() → Charmander Lv5, HP 18/20
2. GET VISION: get_screen_base64() → see Pidgey Lv5
3. DECIDE: 
   - Charmander (Fire) vs Pidgey (Normal/Flying)
   - Type advantage: Fire > Normal
   - HP: 18/20 = 90% - safe
4. ACT: emulator_press_sequence("A") → "Fight"
5. WAIT: W3
6. GET VISION: See move menu (Scratch, Growl, Tail Whip, Ember)
7. DECIDE: Ember does more damage
8. ACT: emulator_press_sequence("DOWN A") → Select Ember
9. WAIT: W30 (battle animation)
10. GET VISION: Check result - Pidgey fainted!
11. SAVE: save_game_state("after-first-battle")
12. LOG: session_set(key="battles_won", value="1")
```

### Chain 3: Explore New Area

```
1. GET STATE: get_map_location() → map_id=38 (Pallet Town)
2. GET VISION: get_screen_base64() → see house ahead
3. DECIDE: Go into house (professor lab)
4. ACT: emulator_press_sequence("RIGHT RIGHT RIGHT")
5. WAIT: W5
6. ACT: emulator_press_sequence("A") → Enter
7. GET VISION: See Oak in lab
8. DECIDE: Talk to Oak (story trigger)
9. ACT: emulator_press_sequence("A") → Talk
10. WAIT: W60 (dialogue)
11. GET STATE: get_money() → Should get starter gift
12. DECIDE: Pick Charmander (best starter)
13. ACT: emulator_press_sequence("RIGHT A")
14. SAVE: save_game_state("received-charmander")
```

### Chain 4: Heal at Pokemon Center

```
1. GET STATE: get_party_info() → Charmander HP 5/20 (critical!)
2. DECIDE: Need healing ASAP
3. GET STATE: get_map_location() → map_id=1 (Viridian City)
4. GET VISION: get_screen_base64() → see Pokemon Center
5. ACT: emulator_press_sequence("RIGHT RIGHT RIGHT")
6. WAIT: W3
7. ACT: emulator_press_sequence("A") → Enter Center
8. WAIT: W10
9. ACT: emulator_press_sequence("A") → Talk to nurse
10. WAIT: W30 (healing animation)
11. GET STATE: get_party_info() → Charmander HP 20/20
12. SAVE: save_game_state("fully-healed")
```

---

## Smart Action Selection

### Based on Game Type

| Game | Key Metrics | Priority Decisions |
|------|-------------|-------------------|
| **Pokemon** | Party HP, Levels, Money, Badges | Battle strategy, Level up, Shop |
| **Zelda** | Health, Items, Keys, Map | Puzzle solving, Exploration |
| **Mario** | Lives, Score, Level | Speedrunning, Collection |
| **Tetris** | Score, Lines, Level | Strategy, Speed |

### Based on Resources

| Resource Level | Decision |
|----------------|----------|
| HP < 20% | Retreat/heal immediately |
| HP 20-50% | Be cautious, consider healing |
| HP > 50% | Safe to continue |
| Money < 100 | Prioritize farming/gold |
| Money > 1000 | Can explore/shop |

### Based on Game Progress

| Progress | Strategy |
|----------|----------|
| Start of game | Follow story, get starter |
| Mid-game | Level up, explore, collect |
| Pre-Elite Four | Grind to appropriate level |
| Elite Four | Optimize team, save often |

---

## Error Recovery Decisions

### Common Error States

```
┌─────────────────────────────────────┐
│       STUCK IN GAME?                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  1. Get current position           │
│     get_player_position()          │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Get screen context             │
│     get_screen_base64()           │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Identify problem              │
│     - Blocked by wall?             │
│     - Menu stuck?                  │
│     - Dialogue waiting?            │
│     - Battle auto-lost?            │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Recovery Actions               │
│                                     │
│  If blocked: try different dir    │
│  If menu stuck: press B to exit   │
│  If dialogue: press A to continue │
│  If fainted: load last save       │
└─────────────────────────────────────┘
```

---

## Session Memory Integration

The agent should store decisions in session for future reference:

```python
# After making a decision
session_set(session_id, "last_action", "attack_with_ember")
session_set(session_id, "battle_history", [...])
session_set(session_id, "visited_locations", [...])
session_set(session_id, "items_used", [...])
```

This allows the agent to:
- Remember what worked before
- Avoid repeating mistakes
- Track long-term goals
- Learn from gameplay patterns

---

## Best Practices

1. **Always save before risky actions** - battles, puzzles, story events
2. **Check game state regularly** - don't rely solely on vision
3. **Use memory reading** - it's faster and more reliable than OCR
4. **Log decisions** - helps with debugging and learning
5. **Plan ahead** - 3-5 steps minimum
6. **Handle errors gracefully** - load saves when stuck
7. **Manage resources** - don't waste items, heal when needed

---

## Related Documentation

- [API_REFERENCE.md](API_REFERENCE.md) - All MCP endpoints
- [EXAMPLES.md](EXAMPLES.md) - Example prompts and sessions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [README.md](../README.md) - Main documentation