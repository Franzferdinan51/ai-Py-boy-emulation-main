#!/usr/bin/env python3
"""
PyBoy Save/Load State Verification Script

This script verifies that save/load state functionality correctly restores
game state, not just endpoint success.

Tests:
1. Screen hash comparison before/after load
2. Memory value comparison for Pokemon Red
3. Gameplay transition verification
4. State roundtrip integrity

Usage:
    python verify_save_load.py [--rom path/to/rom.gb]
"""

import sys
import os
import io
import hashlib
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Set SDL environment for headless operation
os.environ['SDL_WINDOW_HIDDEN'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'disk'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'ai-game-server' / 'src'))

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("ERROR: PyBoy not available. Install with 'pip install pyboy'")
    sys.exit(1)


# Pokemon Red Memory Map (from disassembly project)
POKEMON_RED_MEMORY = {
    'player_x': 0xD362,
    'player_y': 0xD361,
    'map_id': 0xD35E,
    'party_count': 0xD163,
    'money_low': 0xD347,
    'money_mid': 0xD348,
    'money_high': 0xD349,
    'badges': 0xD6E6,
    # Additional state indicators
    'wCurMapTileset': 0xD367,
    'wYBlockCoord': 0xD364,
    'wXBlockCoord': 0xD365,
    'wCurMapWidth': 0xD369,
    'wCurMapHeight': 0xD36A,
}

MAP_NAMES = {
    0: "Pallet Town",
    38: "Player's House 1F",
    39: "Player's House 2F",
    41: "Professor Oak's Lab",
    54: "Route 1",
}


class SaveLoadVerifier:
    """Verifies PyBoy save/load state functionality"""

    def __init__(self, rom_path: str):
        self.rom_path = rom_path
        self.pyboy = None
        self.results = {
            'rom_path': rom_path,
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_passed': False,
            'warnings': [],
            'errors': []
        }

    def initialize(self) -> bool:
        """Initialize PyBoy with the ROM"""
        print(f"\n🎮 Initializing PyBoy with ROM: {os.path.basename(self.rom_path)}")

        if not os.path.exists(self.rom_path):
            self.results['errors'].append(f"ROM not found: {self.rom_path}")
            return False

        try:
            self.pyboy = PyBoy(
                self.rom_path,
                window="headless",
                scale=1,
                sound_emulated=False,
                sound_volume=0
            )
            self.pyboy.set_emulation_speed(0)

            # Warm up emulator
            print("   Warming up emulator...")
            for _ in range(180):
                self.pyboy.tick(1, False)

            print("   ✅ PyBoy initialized successfully")
            return True

        except Exception as e:
            self.results['errors'].append(f"Failed to initialize PyBoy: {e}")
            return False

    def cleanup(self):
        """Clean up PyBoy resources"""
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None

    def get_screen_hash(self) -> str:
        """Get MD5 hash of current screen"""
        try:
            screen = self.pyboy.screen.ndarray
            if screen is None or screen.size == 0:
                return "ERROR_EMPTY_SCREEN"
            # Convert to RGB if RGBA
            if len(screen.shape) == 3 and screen.shape[2] == 4:
                screen = screen[:, :, :3]
            return hashlib.md5(screen.tobytes()).hexdigest()
        except Exception as e:
            return f"ERROR_{e}"

    def get_memory_values(self) -> dict:
        """Read key memory addresses"""
        values = {}
        try:
            for name, addr in POKEMON_RED_MEMORY.items():
                try:
                    values[name] = self.pyboy.memory[addr]
                except Exception:
                    values[name] = None

            # Read money as combined value (BCD encoded)
            try:
                money = (self.pyboy.memory[POKEMON_RED_MEMORY['money_high']] << 16) | \
                        (self.pyboy.memory[POKEMON_RED_MEMORY['money_mid']] << 8) | \
                        self.pyboy.memory[POKEMON_RED_MEMORY['money_low']]
                values['money'] = money
            except Exception:
                values['money'] = None

        except Exception as e:
            self.results['warnings'].append(f"Memory read error: {e}")

        return values

    def save_state_to_bytes(self) -> bytes:
        """Save state to bytes"""
        try:
            state_buffer = io.BytesIO()
            self.pyboy.save_state(state_buffer)
            return state_buffer.getvalue()
        except Exception as e:
            self.results['errors'].append(f"Save state failed: {e}")
            return b''

    def load_state_from_bytes(self, state_data: bytes) -> bool:
        """Load state from bytes"""
        try:
            state_buffer = io.BytesIO(state_data)
            self.pyboy.load_state(state_buffer)
            return True
        except Exception as e:
            self.results['errors'].append(f"Load state failed: {e}")
            return False

    def press_button(self, button: str, frames: int = 30):
        """Press a button for specified frames"""
        button_map = {
            'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
            'A': 'a', 'B': 'b', 'START': 'start', 'SELECT': 'select'
        }
        if button in button_map:
            self.pyboy.button(button_map[button], frames)
        # Tick one extra frame to process release
        self.pyboy.tick(1, True)

    def tick_frames(self, count: int, render: bool = True):
        """Advance emulator frames"""
        for _ in range(count):
            self.pyboy.tick(1, render)

    def test_position_change(self) -> dict:
        """Test: Position change is correctly saved and restored"""
        test_name = "Position Change Verification"
        print(f"\n📍 Test: {test_name}")
        print("   " + "-" * 50)

        result = {
            'name': test_name,
            'passed': False,
            'details': {}
        }

        try:
            # Get initial position
            initial_mem = self.get_memory_values()
            initial_x = initial_mem.get('player_x')
            initial_y = initial_mem.get('player_y')
            initial_map = initial_mem.get('map_id')

            print(f"   Initial position: ({initial_x}, {initial_y}) on map {initial_map}")

            # Save state at initial position
            state_at_initial = self.save_state_to_bytes()
            if not state_at_initial:
                result['error'] = "Failed to save state at initial position"
                return result

            print(f"   Saved state A ({len(state_at_initial)} bytes)")

            # Try to move the player
            print("   Attempting to move player...")
            self.press_button('RIGHT', 15)
            self.tick_frames(30)
            self.press_button('DOWN', 15)
            self.tick_frames(30)

            # Check new position
            moved_mem = self.get_memory_values()
            moved_x = moved_mem.get('player_x')
            moved_y = moved_mem.get('player_y')
            moved_map = moved_mem.get('map_id')

            print(f"   After movement: ({moved_x}, {moved_y}) on map {moved_map}")

            # Check if position actually changed
            position_changed = (moved_x != initial_x or moved_y != initial_y)
            result['details']['position_changed'] = position_changed
            result['details']['initial_pos'] = (initial_x, initial_y)
            result['details']['moved_pos'] = (moved_x, moved_y)

            if not position_changed:
                print("   ⚠️  Position did not change - may be in menu or blocked")
                self.results['warnings'].append("Player position unchanged after movement - may need menu handling")

            # Save state at moved position
            state_at_moved = self.save_state_to_bytes()
            if not state_at_moved:
                result['error'] = "Failed to save state at moved position"
                return result

            print(f"   Saved state B ({len(state_at_moved)} bytes)")

            # Move again to create a different state
            self.press_button('LEFT', 15)
            self.tick_frames(30)

            final_mem_before_load = self.get_memory_values()
            print(f"   Before load: ({final_mem_before_load.get('player_x')}, {final_mem_before_load.get('player_y')})")

            # Now load state from moved position (state B)
            print("   Loading state B...")
            if not self.load_state_from_bytes(state_at_moved):
                result['error'] = "Failed to load state B"
                return result

            # Verify position matches moved position
            self.tick_frames(1, True)  # One tick to ensure state is applied
            loaded_mem = self.get_memory_values()
            loaded_x = loaded_mem.get('player_x')
            loaded_y = loaded_mem.get('player_y')
            loaded_map = loaded_mem.get('map_id')

            print(f"   After load: ({loaded_x}, {loaded_y}) on map {loaded_map}")

            # Verify match
            position_match = (loaded_x == moved_x and loaded_y == moved_y and loaded_map == moved_map)
            result['details']['loaded_pos'] = (loaded_x, loaded_y)
            result['details']['position_match'] = position_match

            if position_match:
                print("   ✅ Position correctly restored!")
                result['passed'] = True
            else:
                print(f"   ❌ Position mismatch! Expected ({moved_x}, {moved_y}), got ({loaded_x}, {loaded_y})")
                result['error'] = "Position not restored correctly"

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Exception: {e}")

        return result

    def test_screen_hash(self) -> dict:
        """Test: Screen content is correctly restored after load"""
        test_name = "Screen Hash Verification"
        print(f"\n🖼️  Test: {test_name}")
        print("   " + "-" * 50)

        result = {
            'name': test_name,
            'passed': False,
            'details': {}
        }

        try:
            # Capture initial screen
            self.tick_frames(1, True)
            initial_hash = self.get_screen_hash()
            print(f"   Initial screen hash: {initial_hash[:16]}...")

            # Save state
            state_a = self.save_state_to_bytes()
            if not state_a:
                result['error'] = "Failed to save state"
                return result

            print(f"   Saved state ({len(state_a)} bytes)")

            # Make changes (press buttons)
            print("   Pressing buttons to change screen...")
            self.press_button('A', 10)
            self.tick_frames(30)
            self.press_button('B', 10)
            self.tick_frames(30)

            # Capture changed screen
            changed_hash = self.get_screen_hash()
            print(f"   Changed screen hash: {changed_hash[:16]}...")

            screen_changed = (changed_hash != initial_hash)
            result['details']['screen_changed'] = screen_changed
            result['details']['initial_hash'] = initial_hash
            result['details']['changed_hash'] = changed_hash

            if not screen_changed:
                print("   ⚠️  Screen did not change - game may be in a static state")
                self.results['warnings'].append("Screen unchanged after button presses - may be in dialog/menu")

            # Load state
            print("   Loading saved state...")
            if not self.load_state_from_bytes(state_a):
                result['error'] = "Failed to load state"
                return result

            # Capture restored screen
            self.tick_frames(1, True)
            restored_hash = self.get_screen_hash()
            print(f"   Restored screen hash: {restored_hash[:16]}...")

            # Verify hash match
            hash_match = (restored_hash == initial_hash)
            result['details']['restored_hash'] = restored_hash
            result['details']['hash_match'] = hash_match

            if hash_match:
                print("   ✅ Screen correctly restored!")
                result['passed'] = True
            else:
                print("   ❌ Screen hash mismatch!")
                result['error'] = "Screen content not restored correctly"

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Exception: {e}")

        return result

    def test_memory_state(self) -> dict:
        """Test: Memory values are correctly restored"""
        test_name = "Memory State Verification"
        print(f"\n🧠 Test: {test_name}")
        print("   " + "-" * 50)

        result = {
            'name': test_name,
            'passed': False,
            'details': {}
        }

        try:
            # Read initial memory
            initial_mem = self.get_memory_values()
            print(f"   Initial memory: X={initial_mem.get('player_x')}, Y={initial_mem.get('player_y')}, Map={initial_mem.get('map_id')}")

            # Save state
            state_a = self.save_state_to_bytes()
            if not state_a:
                result['error'] = "Failed to save state"
                return result

            # Make changes
            print("   Making changes...")
            self.press_button('RIGHT', 20)
            self.tick_frames(60)

            # Read changed memory
            changed_mem = self.get_memory_values()
            print(f"   Changed memory: X={changed_mem.get('player_x')}, Y={changed_mem.get('player_y')}, Map={changed_mem.get('map_id')}")

            # Load state
            print("   Loading saved state...")
            if not self.load_state_from_bytes(state_a):
                result['error'] = "Failed to load state"
                return result

            self.tick_frames(1, True)

            # Read restored memory
            restored_mem = self.get_memory_values()
            print(f"   Restored memory: X={restored_mem.get('player_x')}, Y={restored_mem.get('player_y')}, Map={restored_mem.get('map_id')}")

            # Compare all values
            matches = {}
            mismatches = []
            for key in initial_mem:
                if initial_mem[key] is not None and restored_mem[key] is not None:
                    match = initial_mem[key] == restored_mem[key]
                    matches[key] = match
                    if not match:
                        mismatches.append(f"{key}: {initial_mem[key]} -> {restored_mem[key]}")

            result['details']['initial_memory'] = initial_mem
            result['details']['changed_memory'] = changed_mem
            result['details']['restored_memory'] = restored_mem
            result['details']['matches'] = matches
            result['details']['mismatches'] = mismatches

            if mismatches:
                print(f"   ❌ Memory mismatches: {mismatches}")
                result['error'] = f"Memory values not restored: {mismatches}"
            else:
                print("   ✅ All memory values correctly restored!")
                result['passed'] = True

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Exception: {e}")

        return result

    def test_state_size_consistency(self) -> dict:
        """Test: State size is consistent across multiple saves"""
        test_name = "State Size Consistency"
        print(f"\n📦 Test: {test_name}")
        print("   " + "-" * 50)

        result = {
            'name': test_name,
            'passed': False,
            'details': {}
        }

        try:
            sizes = []

            # Save multiple times
            for i in range(3):
                state = self.save_state_to_bytes()
                if state:
                    sizes.append(len(state))
                    print(f"   Save {i+1}: {len(state)} bytes")
                self.tick_frames(30)

            if len(sizes) >= 2:
                # Check all sizes are the same
                all_same = all(s == sizes[0] for s in sizes)
                result['details']['sizes'] = sizes
                result['details']['all_same'] = all_same

                if all_same:
                    print(f"   ✅ All saves produce consistent size: {sizes[0]} bytes")
                    result['passed'] = True
                else:
                    print(f"   ❌ Inconsistent sizes: {sizes}")
                    result['error'] = "State size varies between saves"
            else:
                result['error'] = "Failed to save enough states"

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Exception: {e}")

        return result

    def test_load_invalid_state(self) -> dict:
        """Test: Loading invalid state is handled gracefully"""
        test_name = "Invalid State Handling"
        print(f"\n⚠️  Test: {test_name}")
        print("   " + "-" * 50)

        result = {
            'name': test_name,
            'passed': False,
            'details': {}
        }

        try:
            # Try to load empty state
            print("   Testing empty state...")
            try:
                empty_loaded = self.load_state_from_bytes(b'')
                result['details']['empty_state_loaded'] = empty_loaded
                print(f"   Empty state load returned: {empty_loaded}")
            except Exception as e:
                print(f"   Empty state raised exception (expected): {e}")
                result['details']['empty_state_exception'] = str(e)

            # Try to load corrupted state
            print("   Testing corrupted state...")
            corrupted = b'\x00' * 100 + b'\xFF' * 100
            try:
                corrupted_loaded = self.load_state_from_bytes(corrupted)
                result['details']['corrupted_state_loaded'] = corrupted_loaded
                print(f"   Corrupted state load returned: {corrupted_loaded}")
            except Exception as e:
                print(f"   Corrupted state raised exception: {e}")
                result['details']['corrupted_state_exception'] = str(e)

            # Test passes if emulator doesn't crash
            print("   ✅ Emulator survived invalid states")
            result['passed'] = True

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Exception: {e}")

        return result

    def run_all_tests(self) -> dict:
        """Run all verification tests"""
        print("\n" + "=" * 60)
        print("🧪 PyBoy Save/Load State Verification")
        print("=" * 60)

        if not self.initialize():
            print("\n❌ Initialization failed")
            return self.results

        try:
            # Run tests
            tests = [
                self.test_state_size_consistency(),
                self.test_screen_hash(),
                self.test_memory_state(),
                self.test_position_change(),
                self.test_load_invalid_state(),
            ]

            self.results['tests'] = tests

            # Calculate overall result
            passed_count = sum(1 for t in tests if t.get('passed'))
            total_count = len(tests)
            self.results['passed_count'] = passed_count
            self.results['total_count'] = total_count

            # Overall pass requires at least screen and memory tests
            critical_tests = ['Screen Hash Verification', 'Memory State Verification']
            critical_passed = all(
                any(t['name'] == name and t.get('passed') for t in tests)
                for name in critical_tests
            )

            self.results['overall_passed'] = critical_passed

            # Summary
            print("\n" + "=" * 60)
            print("📊 SUMMARY")
            print("=" * 60)

            for test in tests:
                status = "✅ PASS" if test.get('passed') else "❌ FAIL"
                print(f"  {status}: {test['name']}")
                if test.get('error'):
                    print(f"          Error: {test['error']}")

            print("\n" + "-" * 60)
            print(f"Total: {passed_count}/{total_count} tests passed")
            print(f"Overall: {'✅ PASSED' if self.results['overall_passed'] else '❌ FAILED'}")
            print("-" * 60)

            if self.results['warnings']:
                print("\n⚠️  Warnings:")
                for w in self.results['warnings']:
                    print(f"  - {w}")

            if self.results['errors']:
                print("\n❌ Errors:")
                for e in self.results['errors']:
                    print(f"  - {e}")

        finally:
            self.cleanup()

        return self.results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Verify PyBoy save/load state functionality')
    parser.add_argument('--rom', type=str, help='Path to ROM file (default: Pokemon Red)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    args = parser.parse_args()

    # Default ROM path
    if args.rom:
        rom_path = args.rom
    else:
        # Try to find Pokemon Red
        possible_paths = [
            '/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb',
            os.path.expanduser('~/roms/Pokemon - Red Version.gb'),
            os.path.expanduser('~/Desktop/ROMS/Pokemon - Red Version.gb'),
        ]

        rom_path = None
        for path in possible_paths:
            if os.path.exists(path):
                rom_path = path
                break

        if not rom_path:
            print("ERROR: No ROM file found. Please specify with --rom")
            print("Searched paths:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)

    # Run verification
    verifier = SaveLoadVerifier(rom_path)
    results = verifier.run_all_tests()

    # Output results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {args.output}")

    # Exit code
    sys.exit(0 if results['overall_passed'] else 1)


if __name__ == '__main__':
    main()