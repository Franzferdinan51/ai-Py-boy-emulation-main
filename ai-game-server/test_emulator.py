#!/usr/bin/env python3
"""
Quick test script for Py-Boy Emulator
Tests basic functionality without MCP overhead
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pyboy import PyBoy
from pyboy.utils import WindowEvent

def test_rom(rom_path):
    """Test loading and running a ROM"""
    print(f"\n🎮 Testing ROM: {rom_path}")
    print("=" * 50)
    
    try:
        # Initialize emulator
        print("Loading ROM...")
        emulator = PyBoy(rom_path, window="null")
        print("✅ ROM loaded successfully!")
        
        # Get initial frame
        print("\nCapturing initial frame...")
        screen = emulator.screen
        if screen is not None:
            img = screen.image
            print(f"✅ Screen captured: {img.size}")
        else:
            print("⚠️  No screen buffer")
        
        # Test button presses
        print("\nTesting button presses...")
        buttons = ['A', 'B', 'START', 'SELECT', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for button in buttons:
            button_map = {
                'A': WindowEvent.PRESS_BUTTON_A,
                'B': WindowEvent.PRESS_BUTTON_B,
                'START': WindowEvent.PRESS_BUTTON_START,
                'SELECT': WindowEvent.PRESS_BUTTON_SELECT,
                'UP': WindowEvent.PRESS_ARROW_UP,
                'DOWN': WindowEvent.PRESS_ARROW_DOWN,
                'LEFT': WindowEvent.PRESS_ARROW_LEFT,
                'RIGHT': WindowEvent.PRESS_ARROW_RIGHT,
            }
            
            release_map = {
                'A': WindowEvent.RELEASE_BUTTON_A,
                'B': WindowEvent.RELEASE_BUTTON_B,
                'START': WindowEvent.RELEASE_BUTTON_START,
                'SELECT': WindowEvent.RELEASE_BUTTON_SELECT,
                'UP': WindowEvent.RELEASE_ARROW_UP,
                'DOWN': WindowEvent.RELEASE_ARROW_DOWN,
                'LEFT': WindowEvent.RELEASE_ARROW_LEFT,
                'RIGHT': WindowEvent.RELEASE_ARROW_RIGHT,
            }
            
            emulator.send_input(button_map[button])
            emulator.tick()
            emulator.send_input(release_map[button])
            print(f"  ✅ {button} pressed")
        
        # Advance a few frames
        print("\nAdvancing 10 frames...")
        for i in range(10):
            emulator.tick()
        print("✅ Frames advanced")
        
        # Get final frame
        print("\nCapturing final frame...")
        screen = emulator.screen
        if screen is not None:
            img = screen.image
            print(f"✅ Final screen: {img.size}")
            
            # Save screenshot
            screenshot_path = "/home/duckets/.openclaw/workspace/screenshots/emulator-test.png"
            img.save(screenshot_path, format='PNG')
            print(f"✅ Screenshot saved: {screenshot_path}")
        else:
            print("⚠️  No screen buffer")
        
        # Cleanup
        print("\nStopping emulator...")
        emulator.stop()
        print("✅ Emulator stopped")
        
        print("\n" + "=" * 50)
        print("🎉 TEST PASSED - Emulator working correctly!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    roms_dir = Path("/home/duckets/roms")
    
    # Find GB files
    gb_files = list(roms_dir.glob("*.gb"))
    
    if not gb_files:
        print("❌ No GB ROM files found in /home/duckets/roms/")
        sys.exit(1)
    
    print(f"Found {len(gb_files)} ROM file(s):")
    for gb in gb_files:
        print(f"  - {gb.name}")
    
    # Test each ROM (skip corrupted ones)
    results = []
    for gb_file in gb_files:
        # Skip files that aren't valid ROM sizes
        file_size = gb_file.stat().st_size
        if file_size % 16384 != 0:
            print(f"\n⚠️  Skipping {gb_file.name} - invalid ROM size ({file_size} bytes)")
            continue
        
        result = test_rom(str(gb_file))
        results.append((gb_file.name, result))
    
    # Summary
    print("\n\n📊 TEST SUMMARY")
    print("=" * 50)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)
