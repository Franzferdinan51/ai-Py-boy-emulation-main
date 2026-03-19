#!/usr/bin/env python3
"""
Memory Scanner - Find memory values in Game Boy RAM
Used by AI agents to discover and track game state

Usage:
    python memory_scan.py --address 0xD000 --length 16
    python memory_scan.py --scan "pokemon" --rom pokemon-red.gb
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not installed. Run: pip install pyboy")
    sys.exit(1)


def scan_memory_range(emulator, start: int, length: int) -> dict:
    """Scan a range of memory addresses"""
    results = []
    
    for addr in range(start, min(start + length, 0x10000)):
        try:
            value = emulator.memory[addr]
            if value != 0:  # Only show non-zero values
                results.append({
                    "address": hex(addr),
                    "address_int": addr,
                    "value": value,
                    "value_hex": hex(value),
                    "value_binary": bin(value)
                })
        except:
            pass
    
    return {
        "start": hex(start),
        "length": length,
        "non_zero_count": len(results),
        "values": results
    }


def scan_for_value(emulator, target_value: int, start: int = 0xC000, end: int = 0xE000) -> dict:
    """Find all addresses containing a specific value"""
    matches = []
    
    for addr in range(start, end):
        try:
            value = emulator.memory[addr]
            if value == target_value:
                matches.append({
                    "address": hex(addr),
                    "address_int": addr,
                    "value": value
                })
        except:
            pass
    
    return {
        "target_value": target_value,
        "target_hex": hex(target_value),
        "scan_range": f"{hex(start)}-{hex(end)}",
        "matches": matches,
        "match_count": len(matches)
    }


def scan_for_pattern(emulator, pattern: bytes, start: int = 0xC000, end: int = 0xE000) -> dict:
    """Find a byte pattern in memory"""
    matches = []
    pattern_len = len(pattern)
    
    for addr in range(start, end - pattern_len):
        try:
            found = True
            for i in range(pattern_len):
                if emulator.memory[addr + i] != pattern[i]:
                    found = False
                    break
            
            if found:
                matches.append({
                    "address": hex(addr),
                    "address_int": addr,
                    "context": [hex(emulator.memory[addr + j]) for j in range(min(pattern_len, 8))]
                })
        except:
            pass
    
    return {
        "pattern": pattern.hex(),
        "pattern_length": pattern_len,
        "scan_range": f"{hex(start)}-{hex(end)}",
        "matches": matches,
        "match_count": len(matches)
    }


def dump_sprite_data(emulator, start: int = 0x8000, length: int = 128) -> dict:
    """Dump sprite/Tile data area (for visual analysis)"""
    results = []
    
    for addr in range(start, start + length):
        try:
            value = emulator.memory[addr]
            results.append({
                "address": hex(addr),
                "value": value,
                "binary": bin(value).replace("0b", "").zfill(8)
            })
        except:
            pass
    
    return {
        "area": "sprite_data",
        "start": hex(start),
        "length": length,
        "data": results[:32]  # Limit output
    }


def find_changed_addresses(emulator, previous_state: dict, start: int = 0xC000, end: int = 0xD000) -> dict:
    """Find memory addresses that changed since last scan"""
    changes = []
    
    for addr in range(start, end):
        try:
            current_value = emulator.memory[addr]
            prev_value = previous_state.get(addr)
            
            if prev_value is not None and current_value != prev_value:
                changes.append({
                    "address": hex(addr),
                    "previous": prev_value,
                    "current": current_value,
                    "difference": current_value - prev_value
                })
        except:
            pass
    
    return {
        "changes": changes,
        "change_count": len(changes)
    }


def main():
    parser = argparse.ArgumentParser(description="Game Boy Memory Scanner for Agents")
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument("--address", type=lambda x: int(x, 0), help="Start address (hex)")
    parser.add_argument("--length", type=int, default=16, help="Length to scan")
    parser.add_argument("--find-value", type=lambda x: int(x, 0), help="Find specific value")
    parser.add_argument("--find-pattern", help="Find byte pattern (hex string)")
    parser.add_argument("--dump-sprites", action="store_true", help="Dump sprite area")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.rom).exists():
        print(f"ERROR: ROM not found: {args.rom}")
        sys.exit(1)
    
    # Initialize emulator
    print(f"Loading ROM: {args.rom}")
    emulator = PyBoy(args.rom, window="null")
    
    results = {}
    
    if args.address is not None:
        print(f"Scanning memory at {hex(args.address)}...")
        results = scan_memory_range(emulator, args.address, args.length)
    
    elif args.find_value is not None:
        print(f"Searching for value {hex(args.find_value)}...")
        results = scan_for_value(emulator, args.find_value)
    
    elif args.find_pattern:
        print(f"Searching for pattern {args.find_pattern}...")
        pattern = bytes.fromhex(args.find_pattern)
        results = scan_for_pattern(emulator, pattern)
    
    elif args.dump_sprites:
        print("Dumping sprite data area...")
        results = dump_sprite_data(emulator)
    
    else:
        print("Scanning common Pokemon addresses...")
        # Scan common game addresses
        common_addresses = [0xD062, 0xD063, 0xD6F5, 0xD6F6, 0xD6F7, 0xD057, 0xD8F6]
        all_results = {}
        for addr in common_addresses:
            try:
                value = emulator.memory[addr]
                all_results[hex(addr)] = {
                    "value": value,
                    "hex": hex(value)
                }
            except:
                pass
        results = {"common_addresses": all_results}
    
    # Output results
    import json
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