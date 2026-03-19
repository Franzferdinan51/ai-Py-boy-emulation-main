#!/usr/bin/env python3
"""
Memory Scanner - Advanced Game State Memory Scanner for Game Boy Pokemon
Scans for HP values, Pokemon stats, items, and other game data

Usage:
    python memory_scanner.py --rom pokemon-red.gb --scan hp
    python memory_scanner.py --rom pokemon-red.gb --scan stats
    python memory_scanner.py --rom pokemon-red.gb --scan items
    python memory_scanner.py --rom pokemon-red.gb --full-scan

Enhanced with:
    - HP value scanning
    - Pokemon stats scanning (attack, defense, speed, etc.)
    - Item location scanning
    - Multi-game support (Red/Blue/Yellow)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not installed. Run: pip install pyboy")
    sys.exit(1)


# Memory addresses for Pokemon Red/Blue/Yellow
# Format: {game: {category: {name: address}}}
GAME_MEMORY_MAP = {
    "pokemon_red": {
        # Player HP
        "player_hp": {
            "current": 0xD6B5,  # 2 bytes (little endian)
            "max": 0xD6BE,      # 2 bytes
        },
        # Party Pokemon (base address 0xD163, 44 bytes each)
        "party": {
            "base": 0xD163,
            "size": 44,
            "max_count": 6,
            "fields": {
                "species": 0,
                "current_hp": 0x1E,  # 2 bytes
                "max_hp": 0x20,      # 2 bytes
                "attack": 0x16,
                "defense": 0x17,
                "speed": 0x19,
                "special": 0x18,
                "level": 0x18,
                "status": 0x1C,       # Status flags
            }
        },
        # Enemy Pokemon (in battle)
        "enemy": {
            "base": 0xD883,
            "current_hp": 0x0C,   # 2 bytes
            "max_hp": 0x10,        # 2 bytes
            "level": 0x16,
            "species": 0,
        },
        # Items
        "inventory": {
            "base": 0xD6E5,
            "item_count": 20,
            "size": 2,  # item_id + quantity
        },
        "key_items": {
            "base": 0xD828,
            "count": 26,
        },
        # Money
        "money": {
            "base": 0xD6F5,  # 3 bytes BCD
            "size": 3,
        },
        # Position
        "position": {
            "x": 0xD062,
            "y": 0xD063,
        },
        # Map
        "map": {
            "id": 0xD35E,
            "bank": 0xD35C,
        },
        # Battle state
        "battle": {
            "status": 0xD057,  # 0 = not in battle
            "type": 0xD058,   # 1 = wild, 2 = trainer
        },
        # Badges
        "badges": {
            "address": 0xD8F6,
            "count": 8,
        },
        # Balls
        "balls": {
            "poke_balls": 0xD67E,
        },
    },
    "pokemon_blue": {
        # Same as Red
        "player_hp": {"current": 0xD6B5, "max": 0xD6BE},
        "party": {"base": 0xD163, "size": 44, "max_count": 6},
        "enemy": {"base": 0xD883},
        "inventory": {"base": 0xD6E5},
        "money": {"base": 0xD6F5},
        "position": {"x": 0xD062, "y": 0xD063},
        "map": {"id": 0xD35E},
        "battle": {"status": 0xD057},
    },
    "pokemon_yellow": {
        # Yellow has slightly different addresses
        "player_hp": {"current": 0xD16C, "max": 0xD175},
        "party": {"base": 0xD11B, "size": 44, "max_count": 6},
        "enemy": {"base": 0xD8B5},
        "inventory": {"base": 0xD700},
        "money": {"base": 0xD72C},
        "position": {"x": 0xD04D, "y": 0xD04E},
        "map": {"id": 0xD367},
        "battle": {"status": 0xD057},
    }
}

# Pokemon species ID to name mapping (Gen 1)
POKEMON_NAMES = {
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur", 4: "Charmander", 5: "Charmeleon",
    6: "Charizard", 7: "Squirtle", 8: "Wartortle", 9: "Blastoise", 10: "Caterpie",
    11: "Metapod", 12: "Butterfree", 13: "Weedle", 14: "Kakuna", 15: "Beedrill",
    16: "Pidgey", 17: "Pidgeotto", 18: "Pidgeot", 19: "Rattata", 20: "Raticate",
    21: "Spearow", 22: "Fearow", 23: "Ekans", 24: "Arbok", 25: "Pikachu",
    26: "Raichu", 27: "Sandshrew", 28: "Sandslash", 29: "Nidoran F", 30: "Nidorina",
    31: "Nidoqueen", 32: "Nidoran M", 33: "Nidorino", 34: "Nidoking", 35: "Clefairy",
    36: "Clefable", 37: "Vulpix", 38: "Ninetales", 39: "Jigglypuff", 40: "Wigglytuff",
    41: "Zubat", 42: "Golbat", 43: "Oddish", 44: "Gloom", 45: "Vileplume",
    46: "Paras", 47: "Parasect", 48: "Venonat", 49: "Diglett", 50: "Dugtrio",
    51: "Meowth", 52: "Persian", 53: "Psyduck", 54: "Golduck", 55: "Mankey",
    56: "Primeape", 57: "Growlithe", 58: "Arcanine", 59: "Poliwag", 60: "Poliwhirl",
    61: "Poliwrath", 62: "Abra", 63: "Kadabra", 64: "Alakazam", 65: "Machop",
    66: "Machoke", 67: "Machamp", 68: "Bellsprout", 69: "Weepinbell", 70: "Victreebel",
    71: "Tentacool", 72: "Tentacruel", 73: "Geodude", 74: "Graveler", 75: "Golem",
    76: "Ponyta", 77: "Rapidash", 78: "Slowpoke", 79: "Slowbro", 80: "Magnemite",
    81: "Magneton", 82: "Farfetch'd", 83: "Doduo", 84: "Dodrio", 85: "Seel",
    86: "Dewgong", 87: "Grimer", 88: "Muk", 89: "Shellder", 90: "Cloyster",
    91: "Gastly", 92: "Haunter", 93: "Gengar", 94: "Onix", 95: "Drowzee",
    96: "Krabby", 97: "Kingler", 98: "Voltorb", 99: "Electrode", 100: "Exeggcute",
    101: "Exeggutor", 102: "Cubone", 103: "Marowak", 104: "Hitmonlee", 105: "Hitmonchan",
    106: "Lickitung", 107: "Koffing", 108: "Weezing", 109: "Rhyhorn", 110: "Rhydon",
    111: "Rhyperior", 112: "Chansey", 113: "Tangela", 114: "Kangaskhan", 115: "Horsea",
    116: "Seadra", 117: "Goldeen", 118: "Seaking", 119: "Staryu", 120: "Starmie",
    121: "Mr. Mime", 122: "Scyther", 123: "Jynx", 124: "Electabuzz", 125: "Magmar",
    126: "Pinsir", 127: "Tauros", 128: "Magikarp", 129: "Gyarados", 130: "Lapras",
    131: "Ditto", 132: "Eevee", 133: "Vaporeon", 134: "Jolteon", 135: "Flareon",
    136: "Porygon", 137: "Omanyte", 138: "Omastar", 139: "Kabuto", 140: "Kabutops",
    141: "Aerodactyl", 142: "Snorlax", 143: "Articuno", 144: "Zapdos", 145: "Moltres",
    146: "Dratini", 147: "Dragonair", 148: "Dragonite", 149: "Mewtwo", 150: "Mew",
}

# Item ID to name mapping (Gen 1)
ITEM_NAMES = {
    0x01: "Master Ball", 0x02: "Ultra Ball", 0x03: "Great Ball", 0x04: "Poke Ball",
    0x05: "Town Map", 0x06: "Bike", 0x07: "Safari Ball", 0x08: "Pokedex",
    0x09: "Moon Stone", 0x0A: "Antidote", 0x0B: "Burn Heal", 0x0C: "Ice Heal",
    0x0D: "Awakening", 0x0E: "Potion", 0x0F: "Super Potion", 0x10: "Hyper Potion",
    0x11: "Max Potion", 0x12: "Revive", 0x13: "Max Revive", 0x14: "Escape Rope",
    0x15: "Repel", 0x16: "Max Repel", 0x17: "Direction Indicator", 0x18: "Old Rod",
    0x19: "Good Rod", 0x1A: "Super Rod", 0x1B: "PP Up", 0x1C: "Ether",
    0x1D: "Max Ether", 0x1E: "Elixir", 0x1F: "Max Elixir", 0x20: "Rare Candy",
    0x21: "PP Max", 0x22: "Unknown 1", 0x23: "Unknown 2", 0x24: "Unknown 3",
    0x25: "HM01", 0x26: "HM02", 0x27: "HM03", 0x28: "HM04", 0x29: "HM05",
    0x2A: "TM01", 0x2B: "TM02", 0x2C: "TM03", 0x2D: "TM04", 0x2E: "TM05",
    0x2F: "TM06", 0x30: "TM07", 0x31: "TM08", 0x32: "TM09", 0x33: "TM10",
    0x34: "TM11", 0x35: "TM12", 0x36: "TM13", 0x37: "TM14", 0x38: "TM15",
    0x39: "TM16", 0x3A: "TM17", 0x3B: "TM18", 0x3C: "TM19", 0x3D: "TM20",
    0x3E: "TM21", 0x3F: "TM22", 0x40: "TM23", 0x41: "TM24", 0x42: "TM25",
    0x43: "TM26", 0x44: "TM27", 0x45: "TM28", 0x46: "TM29", 0x47: "TM30",
    0x48: "TM31", 0x49: "TM32", 0x4A: "TM33", 0x4B: "TM34", 0x4C: "TM35",
    0x4D: "TM36", 0x4E: "TM37", 0x4F: "TM38", 0x50: "TM39", 0x51: "TM40",
    0x52: "TM41", 0x53: "TM42", 0x54: "TM43", 0x55: "TM44", 0x56: "TM45",
    0x57: "TM46", 0x58: "TM47", 0x59: "TM48", 0x5A: "TM49", 0x5B: "TM50",
    0x5C: "Oak's Parcel", 0x5D: "Poke Flute", 0x5E: "Secret Key",
    0x5F: "Bike Voucher", 0x60: "X Accuracy", 0x61: "Leaf Stone", 0x62: "Card Key",
    0x63: "Basement Key", 0x64: "Pass", 0x65: "Protein", 0x66: "Iron",
    0x67: "Carbos", 0x68: "Calcium", 0x69: "Rare Candy", 0x6A: "Dome Fossil",
    0x6B: "Helix Fossil", 0x6C: "Unknown", 0x6D: "Unknown", 0x6E: "S.S. Ticket",
    0x6F: "Mystery Egg", 0x70: "Silver Leaf", 0x71: "Gold Leaf",
    0x72: "Slowpoke Tail", 0x73: "Mystery Berry", 0x74: "Rare Bone",
    0x75: "Green Token", 0x76: "Berry", 0x77: "Gold Berry",
}


class GameMemoryScanner:
    """Scans and parses Game Boy Pokemon memory"""
    
    def __init__(self, emulator: PyBoy, game_type: str = "pokemon_red"):
        self.emulator = emulator
        self.game_type = game_type
        self.memory_map = GAME_MEMORY_MAP.get(game_type, GAME_MEMORY_MAP["pokemon_red"])
    
    def read_hp(self) -> Dict[str, Any]:
        """Scan for HP values"""
        hp_data = {}
        
        # Player current HP (2 bytes, little endian)
        try:
            current = self.emulator.memory[self.memory_map["player_hp"]["current"]]
            current |= self.emulator.memory[self.memory_map["player_hp"]["current"] + 1] << 8
            hp_data["player_current"] = current
        except:
            hp_data["player_current"] = None
        
        # Player max HP
        try:
            max_hp = self.emulator.memory[self.memory_map["player_hp"]["max"]]
            max_hp |= self.emulator.memory[self.memory_map["player_hp"]["max"] + 1] << 8
            hp_data["player_max"] = max_hp
            
            if max_hp > 0:
                hp_data["player_percent"] = round(current / max_hp * 100, 1)
            else:
                hp_data["player_percent"] = 0
        except:
            hp_data["player_max"] = None
        
        # Check for enemy HP (if in battle)
        try:
            enemy_hp = self.emulator.memory[self.memory_map["enemy"]["current_hp"]]
            enemy_hp |= self.emulator.memory[self.memory_map["enemy"]["current_hp"] + 1] << 8
            hp_data["enemy_current"] = enemy_hp
        except:
            hp_data["enemy_current"] = None
        
        try:
            enemy_max = self.emulator.memory[self.memory_map["enemy"]["max_hp"]]
            enemy_max |= self.emulator.memory[self.memory_map["enemy"]["max_hp"] + 1] << 8
            hp_data["enemy_max"] = enemy_max
            
            if enemy_max > 0 and enemy_hp:
                hp_data["enemy_percent"] = round(enemy_hp / enemy_max * 100, 1)
        except:
            hp_data["enemy_max"] = None
        
        return hp_data
    
    def scan_hp_values(self, range_start: int = 0xD000, range_end: int = 0xDFFF) -> List[Dict]:
        """Scan memory range for HP-like values (non-zero 2-byte values)"""
        hp_values = []
        
        for addr in range(range_start, range_end - 1, 2):
            try:
                value = self.emulator.memory[addr] | (self.emulator.memory[addr + 1] << 8)
                # HP values are typically 1-999
                if 1 <= value <= 999:
                    hp_values.append({
                        "address": hex(addr),
                        "value": value,
                        "possible": "player_hp" if 0xD6B0 <= addr <= 0xD6BF else
                                   "party_hp" if 0xD163 <= addr <= 0xD1A0 else
                                   "enemy_hp" if 0xD890 <= addr <= 0xD8A0 else "unknown"
                    })
            except:
                pass
        
        return hp_values[:20]  # Limit results
    
    def read_party_stats(self) -> List[Dict]:
        """Read all party Pokemon stats"""
        party = []
        party_base = self.memory_map["party"]["base"]
        party_size = self.memory_map["party"]["size"]
        max_count = self.memory_map["party"]["max_count"]
        
        for i in range(max_count):
            offset = party_base + (i * party_size)
            
            try:
                species = self.emulator.memory[offset]
                
                # Skip empty slots
                if species == 0 or species == 0xFF:
                    continue
                
                pokemon = {
                    "slot": i + 1,
                    "species_id": species,
                    "species_name": POKEMON_NAMES.get(species, f"Unknown_{species}"),
                }
                
                # Read stats
                try:
                    hp_offset = self.memory_map["party"]["fields"]["current_hp"]
                    pokemon["current_hp"] = self.emulator.memory[offset + hp_offset] | \
                                           (self.emulator.memory[offset + hp_offset + 1] << 8)
                except:
                    pokemon["current_hp"] = 0
                
                try:
                    max_hp_offset = self.memory_map["party"]["fields"]["max_hp"]
                    pokemon["max_hp"] = self.emulator.memory[offset + max_hp_offset] | \
                                        (self.emulator.memory[offset + max_hp_offset + 1] << 8)
                except:
                    pokemon["max_hp"] = 0
                
                if pokemon.get("max_hp", 0) > 0:
                    pokemon["hp_percent"] = round(pokemon["current_hp"] / pokemon["max_hp"] * 100, 1)
                
                # IVs/Stats
                try:
                    stats_offset = self.memory_map["party"]["fields"]
                    pokemon["attack"] = self.emulator.memory[offset + stats_offset["attack"]]
                    pokemon["defense"] = self.emulator.memory[offset + stats_offset["defense"]]
                    pokemon["speed"] = self.emulator.memory[offset + stats_offset["speed"]]
                    pokemon["special"] = self.emulator.memory[offset + stats_offset["special"]]
                except:
                    pass
                
                # Level
                try:
                    level_offset = self.memory_map["party"]["fields"]["level"]
                    pokemon["level"] = self.emulator.memory[offset + level_offset]
                except:
                    pokemon["level"] = 0
                
                # Status
                try:
                    status_offset = self.memory_map["party"]["fields"]["status"]
                    status_byte = self.emulator.memory[offset + status_offset]
                    pokemon["status"] = self._decode_status(status_byte)
                except:
                    pokemon["status"] = "none"
                
                party.append(pokemon)
                
            except (IndexError, KeyError):
                break
        
        return party
    
    def _decode_status(self, status_byte: int) -> str:
        """Decode status byte to readable status"""
        if status_byte == 0:
            return "none"
        
        statuses = []
        if status_byte & 0x01:
            statuses.append("poisoned")
        if status_byte & 0x02:
            statuses.append("sleeping")
        if status_byte & 0x04:
            statuses.append("frozen")
        if status_byte & 0x08:
            statuses.append("paralyzed")
        if status_byte & 0x10:
            statuses.append("burned")
        
        return ",".join(statuses) if statuses else "unknown"
    
    def read_inventory(self) -> List[Dict]:
        """Read player inventory"""
        items = []
        inv_base = self.memory_map["inventory"]["base"]
        
        try:
            for i in range(self.memory_map["inventory"]["item_count"]):
                offset = inv_base + (i * 2)
                item_id = self.emulator.memory[offset]
                quantity = self.emulator.memory[offset + 1]
                
                if item_id == 0 or item_id == 0xFF:
                    continue
                
                items.append({
                    "slot": i + 1,
                    "item_id": item_id,
                    "item_hex": hex(item_id),
                    "item_name": ITEM_NAMES.get(item_id, f"Unknown_{item_id}"),
                    "quantity": quantity
                })
        except:
            pass
        
        return items
    
    def read_money(self) -> Dict[str, Any]:
        """Read player money (BCD format)"""
        money_data = {}
        
        try:
            money_bytes = [
                self.emulator.memory[self.memory_map["money"]["base"]],
                self.emulator.memory[self.memory_map["money"]["base"] + 1],
                self.emulator.memory[self.memory_map["money"]["base"] + 2]
            ]
            
            # Convert BCD to decimal
            money = 0
            for byte in money_bytes:
                high_digit = (byte >> 4) & 0x0F
                low_digit = byte & 0x0F
                money = money * 100 + high_digit * 10 + low_digit
            
            money_data["money"] = money
            money_data["formatted"] = f"${money:,}"
            money_data["raw_bytes"] = [hex(b) for b in money_bytes]
        except:
            money_data["money"] = 0
            money_data["error"] = "Failed to read money"
        
        return money_data
    
    def read_position(self) -> Dict[str, Any]:
        """Read player position"""
        try:
            x = self.emulator.memory[self.memory_map["position"]["x"]]
            y = self.emulator.memory[self.memory_map["position"]["y"]]
            return {"x": x, "y": y, "formatted": f"({x}, {y})"}
        except:
            return {"error": "Failed to read position"}
    
    def read_map(self) -> Dict[str, Any]:
        """Read current map"""
        try:
            map_id = self.emulator.memory[self.memory_map["map"]["id"]]
            return {"map_id": map_id, "map_hex": hex(map_id)}
        except:
            return {"error": "Failed to read map"}
    
    def read_battle_state(self) -> Dict[str, Any]:
        """Read current battle state"""
        battle = {}
        
        try:
            status = self.emulator.memory[self.memory_map["battle"]["status"]]
            battle["in_battle"] = status != 0
            battle["status_code"] = status
            battle["battle_type"] = "wild" if status == 1 else "trainer" if status == 2 else "none"
        except:
            battle["in_battle"] = False
        
        return battle
    
    def read_badges(self) -> List[str]:
        """Read obtained badges"""
        badges = []
        
        try:
            badge_byte = self.emulator.memory[self.memory_map["badges"]["address"]]
            
            badge_names = ["Boulder", "Cascade", "Thunder", "Rainbow",
                          "Soul", "Marsh", "Volcano", "Earth"]
            
            for i, name in enumerate(badge_names):
                if badge_byte & (1 << i):
                    badges.append(name)
        except:
            pass
        
        return badges
    
    def read_balls(self) -> Dict[str, int]:
        """Read ball counts"""
        balls = {}
        
        try:
            balls["poke_balls"] = self.emulator.memory[self.memory_map["balls"]["poke_balls"]]
        except:
            balls["poke_balls"] = 0
        
        return balls
    
    def full_scan(self) -> Dict[str, Any]:
        """Perform full memory scan"""
        return {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "game_type": self.game_type,
            "hp": self.read_hp(),
            "party": self.read_party_stats(),
            "inventory": self.read_inventory(),
            "money": self.read_money(),
            "position": self.read_position(),
            "map": self.read_map(),
            "battle": self.read_battle_state(),
            "badges": self.read_badges(),
            "balls": self.read_balls(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Memory Scanner for Pokemon Games"
    )
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument(
        "--scan",
        choices=["hp", "stats", "items", "all", "battle"],
        default="all",
        help="What to scan for"
    )
    parser.add_argument(
        "--game",
        choices=["pokemon_red", "pokemon_blue", "pokemon_yellow"],
        default="pokemon_red",
        help="Game type"
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Complete memory scan"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.rom).exists():
        print(f"ERROR: ROM not found: {args.rom}")
        sys.exit(1)
    
    # Initialize emulator
    print(f"Loading ROM: {args.rom}")
    emulator = PyBoy(args.rom, window="null")
    
    # Create scanner
    scanner = GameMemoryScanner(emulator, args.game)
    
    results = {}
    
    if args.full_scan:
        results = scanner.full_scan()
    elif args.scan == "hp":
        results = {
            "hp": scanner.read_hp(),
            "party": scanner.read_party_stats(),
        }
    elif args.scan == "stats":
        results = {
            "party": scanner.read_party_stats(),
            "money": scanner.read_money(),
            "badges": scanner.read_badges(),
        }
    elif args.scan == "items":
        results = {
            "inventory": scanner.read_inventory(),
            "balls": scanner.read_balls(),
        }
    elif args.scan == "battle":
        results = {
            "battle": scanner.read_battle_state(),
            "hp": scanner.read_hp(),
        }
    elif args.scan == "all":
        results = scanner.full_scan()
    
    # Output results
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