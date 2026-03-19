#!/usr/bin/env python3
"""
Detailed PyBoy Save/Load Debug Test

Investigates the cycle 3 failure in save/load verification.
"""

import sys
import os
import io
import hashlib
import time

# Set SDL environment for headless operation
os.environ['SDL_WINDOW_HIDDEN'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'disk'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
    from pyboy import PyBoy
except ImportError:
    print("ERROR: PyBoy not available")
    sys.exit(1)

# Pokemon Red Memory Map
MEMORY = {
    'player_x': 0xD362,
    'player_y': 0xD361,
    'map_id': 0xD35E,
}


def get_memory_values(pyboy) -> dict:
    """Read key memory addresses"""
    values = {}
    for name, addr in MEMORY.items():
        try:
            values[name] = pyboy.memory[addr]
        except Exception:
            values[name] = None
    return values


def get_screen_hash(pyboy) -> str:
    """Get MD5 hash of current screen"""
    try:
        screen = pyboy.screen.ndarray
        if screen is None or screen.size == 0:
            return "ERROR_EMPTY_SCREEN"
        if len(screen.shape) == 3 and screen.shape[2] == 4:
            screen = screen[:, :, :3]
        return hashlib.md5(screen.tobytes()).hexdigest()
    except Exception as e:
        return f"ERROR_{e}"


def press_button(pyboy, button: str, frames: int = 15):
    """Press a button"""
    button_map = {
        'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
        'A': 'a', 'B': 'b', 'START': 'start', 'SELECT': 'select'
    }
    if button in button_map:
        pyboy.button(button_map[button], frames)
    pyboy.tick(1, True)


def tick_frames(pyboy, count: int, render: bool = True):
    """Advance emulator frames"""
    for _ in range(count):
        pyboy.tick(1, render)


def main():
    print("\n" + "=" * 60)
    print("🔍 PyBoy Save/Load Debug Test")
    print("=" * 60)

    # Paths
    rom_path = "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb"
    save_state_path = "/Users/duckets/.openclaw/workspace/mcp-pyboy/saves/duckbot_route1.state"

    # Initialize PyBoy
    print("\n🎮 Initializing PyBoy...")
    pyboy = PyBoy(
        rom_path,
        window="null",
        scale=1,
        sound_emulated=False,
        sound_volume=0
    )
    pyboy.set_emulation_speed(0)

    # Warm up
    for _ in range(60):
        pyboy.tick(1, False)

    # Load gameplay save state
    with open(save_state_path, 'rb') as f:
        state_data = f.read()
    pyboy.load_state(io.BytesIO(state_data))
    tick_frames(pyboy, 10, True)

    # Get baseline state
    baseline_hash = get_screen_hash(pyboy)
    baseline_mem = get_memory_values(pyboy)
    print(f"   Baseline: ({baseline_mem.get('player_x')}, {baseline_mem.get('player_y')}) hash={baseline_hash[:16]}...")

    # TEST: Multiple cycles with detailed tracking
    print("\n" + "-" * 50)
    print("TEST: Multiple save/load cycles (detailed)")
    print("-" * 50)

    # First, save the baseline state
    baseline_state_buf = io.BytesIO()
    pyboy.save_state(baseline_state_buf)
    baseline_state = baseline_state_buf.getvalue()

    for i in range(5):
        print(f"\n--- Cycle {i+1} ---")
        
        # Save current state
        state_buf = io.BytesIO()
        pyboy.save_state(state_buf)
        saved_state = state_buf.getvalue()
        saved_hash = get_screen_hash(pyboy)
        saved_mem = get_memory_values(pyboy)
        print(f"   Saved: ({saved_mem.get('player_x')}, {saved_mem.get('player_y')}) hash={saved_hash[:16]}...")

        # Make changes
        press_button(pyboy, 'DOWN', 10)
        tick_frames(pyboy, 20)

        changed_hash = get_screen_hash(pyboy)
        changed_mem = get_memory_values(pyboy)
        print(f"   After change: ({changed_mem.get('player_x')}, {changed_mem.get('player_y')}) hash={changed_hash[:16]}...")

        # Load saved state
        load_buf = io.BytesIO(saved_state)
        pyboy.load_state(load_buf)
        tick_frames(pyboy, 5, True)

        loaded_hash = get_screen_hash(pyboy)
        loaded_mem = get_memory_values(pyboy)
        print(f"   After load: ({loaded_mem.get('player_x')}, {loaded_mem.get('player_y')}) hash={loaded_hash[:16]}...")

        # Verify
        if loaded_hash == saved_hash:
            print(f"   ✅ Cycle {i+1} PASSED (hash match)")
        else:
            print(f"   ❌ Cycle {i+1} FAILED (hash mismatch)")
            print(f"      Expected: {saved_hash[:16]}...")
            print(f"      Got:      {loaded_hash[:16]}...")
            
            # Try again with more ticks
            print("   Retrying with more ticks...")
            pyboy.load_state(io.BytesIO(saved_state))
            tick_frames(pyboy, 30, True)
            retry_hash = get_screen_hash(pyboy)
            print(f"   After 30 ticks: {retry_hash[:16]}...")
            if retry_hash == saved_hash:
                print("   ✅ Retry with more ticks succeeded")
            else:
                print("   ❌ Still mismatch after more ticks")

    # Reload baseline and check drift
    print("\n" + "-" * 50)
    print("TEST: State drift check")
    print("-" * 50)
    
    pyboy.load_state(io.BytesIO(baseline_state))
    tick_frames(pyboy, 5, True)
    final_hash = get_screen_hash(pyboy)
    final_mem = get_memory_values(pyboy)
    
    print(f"   Baseline hash: {baseline_hash[:16]}...")
    print(f"   Final hash:    {final_hash[:16]}...")
    print(f"   Match: {'✅' if final_hash == baseline_hash else '❌'}")

    # Cleanup
    pyboy.stop()
    print("\n✅ Debug test complete")


if __name__ == '__main__':
    main()