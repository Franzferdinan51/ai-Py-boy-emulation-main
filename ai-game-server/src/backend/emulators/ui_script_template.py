import sys
import os
import time

print("=== PYBOY UI SCRIPT STARTING ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"ROM path: {{ROM_PATH}}")

# Add PyBoy path to Python path
sys.path.insert(0, {{PYBOY_PATH}})
sys.path.insert(0, {{PROJECT_PATH}})

try:
    print("Importing PyBoy...")
    from pyboy import PyBoy
    print("PyBoy imported successfully")

    print(f"Loading ROM: {{ROM_PATH}}")
    rom_path = {{ROM_PATH}}
    if not os.path.exists(rom_path):
        print(f"ERROR: ROM file not found: {rom_path}")
        sys.exit(1)

    pyboy = PyBoy(rom_path, window="SDL2", scale=2, sound_emulated=False, debug=False)
    print("PyBoy initialized successfully")
    pyboy.set_emulation_speed(1)
    print("Emulation speed set to 1")

    frame_count = 0
    print("Starting UI loop...")
    # Keep the UI running
    while True:
        try:
            pyboy.tick(1, True)
            frame_count += 1
            if frame_count % 60 == 0:  # Log every 2 seconds at 30fps
                print(f"UI frame: {frame_count}")
        except Exception as tick_error:
            print(f"Tick error: {tick_error}")
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("UI interrupted by user")
            break

except KeyboardInterrupt:
    print("UI interrupted by user")
except Exception as e:
    print(f"UI process error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("UI process ending")
