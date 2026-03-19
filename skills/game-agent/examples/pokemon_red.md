# Pokemon Red Walkthrough & Strategy Guide

**Status:** ✅ Complete  
**Version:** 1.0  
**Last Updated:** March 19, 2026  
**Game:** Pokemon Red (Game Boy)

---

## Quick Reference

| Detail | Value |
|--------|-------|
| **ROM** | `pokemon-red.gb` |
| **Memory Model** | DMG (original Game Boy) |
| **Save Format** | .state files |
| **Recommended Starter** | Charmander 🔥 |

---

## Memory Addresses (Pokemon Red)

### Player Position

| Address | Description | Example |
|---------|-------------|---------|
| 0xD361 | Player Y position (tile) | 12 |
| 0xD362 | Player X position (tile) | 8 |
| 0xD35E | Current map ID | 38 = Pallet Town |

### Map IDs

| ID | Location | ID | Location |
|----|----------|----|----------|
| 38 | Pallet Town | 6 | Cerulean City |
| 40 | Viridian City | 19 | Vermilion City |
| 2 | Pewter City | 28 | Celadon City |
| 3 | Route 2 | 21 | Fuchsia City |
| 4 | Viridian Forest | 24 | Saffron City |
| 5 | Route 24 | 31 | Cinnabar Island |

### Player Stats

| Address | Description | Format |
|---------|-------------|--------|
| 0xD347 | Money (byte 1) | BCD |
| 0xD348 | Money (byte 2) | BCD |
| 0xD349 | Money (byte 3) | BCD |
| 0xD8F6 | Badge count | 0-8 |
| 0xD163 | Party count | 0-6 |

### Battle State

| Address | Description |
|---------|-------------|
| 0xD011 | Game mode (0=overworld, 3=battle) |
| 0xD057 | Enemy species (battle) |
| 0xD058 | Enemy level (battle) |
| 0xCC26 | Current map ID |

---

## Complete Walkthrough

### Phase 1: Pallet Town (Start)

```
┌────────────────────────────────────────┐
│  📍 PALLET TOWN - START                │
│  Map ID: 38                            │
└────────────────────────────────────────┘

BUTTON SEQUENCE:
1. "START"         - Open menu (or skip)
2. "A"             - Confirm "New Game"
3. "W W W"         - Wait through intro
4. "START"         - Skip "Game Freak" (or A)
5. "A" x 5         - Select language/options
6. "START"         - Confirm

📋 OBJECTIVES:
   □ Choose starter Pokemon
   □ Get Pokedex from Oak
   □ Explore all houses
   □ Collect items
```

**Oak's Lab Sequence:**
```
1. Talk to Oak → "I'd like to tell you something"
2. Walk up to desk → Meet rival
3. Choose starter:
   - LEFT → Squirtle (Water) 🐢
   - UP   → Bulbasaur (Grass) 🌿
   - RIGHT→ Charmander (Fire) 🔥 (RECOMMENDED)

4. Rival battle (optional, can run)
   - Your starter at Lv 5
   - Rival's starter at Lv 5
   - Just fight or run!

5. Get Running Shoes → "Try B button"
6. Get Pokedex
```

**Items in Pallet Town:**
- Town Map (Oak's Lab)
- Potion (House, bottom-right)

---

### Phase 2: Route 1

```
┌────────────────────────────────────────┐
│  🛤️ ROUTE 1 - GRASSROOTS              │
│  Map ID: 40                            │
└────────────────────────────────────────┘

BUTTON SEQUENCE:
1. "RIGHT"         - Exit house
2. "R R R"         - Walk right to route
3. "UP"            - Enter grass

WILDMON ENCOUNTERS:
├── Pidgey (Flying)  - Common, Lv 2-5
└── Rattata (Normal) - Common, Lv 2-5

💡 TIPS:
- Fight wild Pokemon to level up
- Catch Pidgey (Flying beats Bug/Grass)
- Don't rush - grind to Lv 10+!

🎯 GOALS:
   □ Reach Level 10+
   □ Learn Ember (if Charmander)
   □ Navigate to Viridian City
```

---

### Phase 3: Viridian City

```
┌────────────────────────────────────────┐
│  🏙️ VIRIDIAN CITY                      │
│  Map ID: 40                            │
└────────────────────────────────────────┘

BUTTON SEQUENCE:
1. Enter from Route 1 (LEFT)
2. Explore city

IMPORTANT LOCATIONS:
├── Pokemart (west)
│   └── Buy: Potions, Poke Balls
├── Pokemon Center (east)
│   └── Heal: Free HP/PP restore
└── Gym (north) - LOCKED (needs badges)

🛒 SHOPPING LIST:
   Potions x5      - $150
   Poke Balls x10  - $200
   Antidote x3     - $60
   -----
   Total: ~$410

🎯 GOALS:
   □ Heal at Pokemon Center
   □ Stock up at Pokemart
   □ Proceed to Route 2
```

---

### Phase 4: Route 2 & Viridian Forest

```
┌────────────────────────────────────────┐
│  🌲 ROUTE 2 / VIRIDIAN FOREST         │
│  Maps: 3, 4                            │
└────────────────────────────────────────┘

ROUTE 2:
- Enter from north of Viridian City
- Lots of grass, wild Pokemon

VIRIDIAN FOREST (Map ID: 4):
├── Weedle (Bug/Poison) - Lv 5-7
├── Kakuna (Bug/Poison) - Evolves Weedle
├── Pidgey (Flying)
└── Caterpie (Bug) - Good for catching

ITEMS:
- Antidote (in forest)
- Escape Rope (in forest)

💡 TIPS:
- Use Potions as needed
- Catch Caterpie for early Bug type
- Watch for Trainer battles
- Avoid wild Pokemon if low on Potions

🎯 GOALS:
   □ Reach Pewter City
   □ Collect forest items
   □ Level to Lv 14+
```

---

### Phase 5: Pewter City & Brock

```
┌────────────────────────────────────────┐
│  🏛️ PEWTER CITY - ROCK GYM            │
│  Map ID: 2                             │
└────────────────────────────────────────┘

GYM LEADER: BROCK
├── Geodude  (Rock/Ground) Lv 10
└── Onix     (Rock/Ground) Lv 12

TYPE MATCHUP:
- Water > Rock/Ground (SUPER!)
- Grass > Rock/Ground (Not very)
- Fire > Bug/Grass/Ice (SUPER!)

🦆 IF CHARMANDER:
- Ember is SE against Bug
- NOT effective against Rock/Ground
- Use: Scratch + Growl (lower attack)
- Switch to friend if possible!

💡 STRATEGY:
1. Save before gym!
2. Use Growl to lower Geodude's attack
3. Heal when HP low
4. Grind extra levels if struggling

🎯 GOALS:
   □ Beat Brock
   □ Get Boulder Badge (+10 Defense)
   □ Head to Route 3
```

---

### Phase 6: Route 3 & Beyond

```
┌────────────────────────────────────────┐
│  🛤️ ROUTE 3 - SUGAR HILL              │
│  Map ID: ?                             │
└────────────────────────────────────────┘

WILDMON + TRAINERS:
- Rattata, Ekans, Sandshrew
- Many trainer battles

🎯 CONTINUE TO:
- Route 4 → Cerulean City
- Continue gym circuit...

---

## Battle Strategy

### Type Advantage Chart (Quick Reference)

```
ATTACKER →      NORM  FIRE WATR ELEC GRAS ICE FIGT POIS GHOST
─────────────────────────────────────────────────────────────────
NORMAL           1     1    1    1    1    1   1    1    0*
FIRE            1    .5   .5   1    2    2   1    1    1
WATER            1     2   .5   1   .5    1   1    1    1
ELECTRIC         1     1    2   .5   .5    1   1    1    1
GRASS            1    .5    2   1   .5    1   1   .5    1
FIGHTING         2     1    1    1    1    1   1    1    0
POISON           1     1    1    1   .5    1   .5  .5    .5
GROUND          1     1    1    2    1    1   1    2    1
FLYING          1     1    1   .5    2    1  .5    1    1
PSYCHIC          1     1    1    1    1    1   2    2    1
BUG             1    .5    1    1   .5    1  .5   .5    1
ROCK            1     2    1    1    1   .5   1    1    1
DRAGON          1     1    1    1    1    1   1    1    1
GHOST            1     1    1    1    1    1  .5    1    2*
* Normal vs Ghost = 0 (no effect)
* Ghost vs Psychic = 2 (SE)
```

### Battle Actions

| Action | When to Use |
|--------|-------------|
| **Attack** | Default - use strongest SE move |
| **Growl/Leer** | Lower enemy stats before attacking |
| **Tail Whip** | Lower enemy Defense |
| **String Shot** | Lower enemy Speed (escaping!) |
| **Potion** | When HP < 50% |
| **Switch** | When type disadvantage is bad |
| **Run** | When Pokemon will faint otherwise |

---

## Memory Reading Examples

### Get Player Position

```bash
# Get X, Y coordinates
mcporter call pyboy.get_memory address=0xD361 length=2
# Returns: [y, x] tile positions
```

### Get Money

```bash
# Money is BCD encoded at 0xD347-0xD349
mcporter call pyboy.get_memory address=0xD347 length=3
# Example: [0x00, 0x30, 0x00] = $300
```

### Get Map Location

```bash
mcporter call pyboy.get_memory address=0xD35E length=1
# Returns: [0x26] = 38 = Pallet Town
```

### Get Full Player Info

```bash
mcporter call pyboy.get_player_info
```

Returns:
```json
{
  "map_id": 38,
  "map_name": "Pallet Town",
  "position": {"x": 10, "y": 12},
  "money": "$500",
  "badges": 0,
  "party_count": 1
}
```

---

## Save Points Strategy

```
🦆 SAVE BEFORE:
├── Entering any gym
├── Battling rival
├── Exploring dangerous routes
└── Elite 4 challenge

🦆 SAVE AFTER:
├── Beating gym leader
├── Catching legendary
├── Reaching new city
└── Major objectives
```

```bash
# Save game state
mcporter call pyboy.save_game_state save_path="/path/to/saves/pallet-town.state"

# Load game state
mcporter call pyboy.load_game_state save_path="/path/to/saves/pallet-town.state"
```

---

## Button Sequences Reference

### Movement

| Action | Sequence |
|--------|----------|
| Walk up | `UP` |
| Walk down | `DOWN` |
| Walk left | `LEFT` |
| Walk right | `RIGHT` |
| Run (hold B) | `B [direction]` |

### Interactions

| Action | Sequence |
|--------|----------|
| Talk/Confirm | `A` |
| Cancel/Back | `B` |
| Open menu | `START` |
| Select | `SELECT` |

### Navigation

| Action | Sequence |
|--------|----------|
| Move 1 tile | `[direction]` |
| Move 5 tiles | `[direction]5` (hold 5 frames) |
| Move through door | `[direction] W W W` |
| Wait 1 second | `W60` |

---

## Common Issues & Solutions

### Pokemon Keeps Fainting

**Problem:** Your Pokemon faints too easily  
**Solution:**
1. Use Potions liberally (don't hoard!)
2. Run from tough battles
3. Grind more levels
4. Switch to stronger Pokemon

### Can't Beat Brock

**Problem:** Rock types resist Charmander  
**Solution:**
1. Grind to Lv 15+ before gym
2. Use Growl to lower Geodude's attack
3. Have friend Pokemon ready for Onix
4. Or: catch a Mankey (Fighting!)

### Low on Money

**Problem:** Can't afford supplies  
**Solution:**
1. Fight wild Pokemon for money
2. Sell unused items at Pokemart
3. Don't buy too many Potions early
4. Use Pokemon Center (free heals!)

---

## Recommended Party

### Early Game (Pallet - Viridian)

| Pokemon | Level | Notes |
|---------|-------|-------|
| Charmander | 5-10 | Starter, main damage |
| Pidgey | 5 | Flying, catches |
| Rattata | 5 | Early grinder |

### Mid Game (Pewter - Cerulean)

| Pokemon | Level | Notes |
|---------|-------|-------|
| Charmeleon | 16+ | Evolved, strong |
| Pidgeotto | 16+ | Flying, HM slave |
| Raticate | 14+ | Normal filler |
| Butterfree | 12 | Bug for early routes |

### Late Game (Vermilion - Elite 4)

| Pokemon | Level | Notes |
|---------|-------|-------|
| Charizard | 36+ | Final form! |
| Pidgeot | 36+ | Fly HM |
| Alakazam* | 40+ | Psychic (trade) |
| Dragonite* | 45+ | Dragon (trade) |

*Requires trading or in-game events

---

## Automation Script Example

```python
#!/usr/bin/env python3
"""Pokemon Red automation - Pewter Gym"""
import json
import subprocess

def call_mcp(tool, args=None):
    cmd = f"mcporter call pyboy.{tool}"
    if args:
        for k, v in args.items():
            cmd += f" {k}={v}"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return json.loads(result.stdout)

# Load ROM
call_mcp("load_rom", {"rom_path": "/path/to/pokemon-red.gb"})

# Navigate to Pewter
# (simplified - use button sequences)

# Save before gym
call_mcp("save_game_state", {"save_path": "pewter-gym.state"})

# Battle Brock
for move in ["Growl", "Scratch", "Scratch", "Ember"]:
    call_mcp("press_button", {"button": "A"})
    call_mcp("wait_frames", {"frames": 30})
```

---

## See Also

- [game-agent/SKILL.md](../SKILL.md) - Main skill docs
- [openclaw/SKILL.md](../openclaw/SKILL.md) - OpenClaw setup
- [generic.md](./generic.md) - Generic game guide

---

**Author:** Game Agent 🦆  
**Game:** Pokemon Red  
**Goal:** Become the Champion!