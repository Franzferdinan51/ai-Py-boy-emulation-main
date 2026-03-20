"""
PyBoy emulator implementation with performance optimizations

SDL_WINDOW_HIDDEN=1 - Creates invisible window for screen rendering
SDL_AUDIODRIVER=disk - Disables audio for performance
SDL_VIDEODRIVER=dummy - Forces dummy video driver (no actual display)
                        CRITICAL for macOS: prevents SDL2 from creating menus
                        on background threads, which causes crashes.
"""
import os
# CRITICAL: Set SDL2 environment variables BEFORE any SDL2 import
# These must be set before PyBoy imports SDL2 to prevent macOS thread crashes
os.environ['SDL_WINDOW_HIDDEN'] = '1'
os.environ['SDL_AUDIODRIVER'] = 'disk'
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Prevents SDL2 menu creation on macOS

import numpy as np
from typing import List, Tuple, Optional, Any
import io
import sys
import logging
import subprocess
import threading
import multiprocessing
import time
import hashlib

# Performance optimization imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Multi-processing support
try:
    import multiprocessing as mp
    from multiprocessing import Queue, Process, Event, Manager
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Try to import PyBoy
try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("PyBoy not available. Install with 'pip install pyboy'")

from .emulator_interface import EmulatorInterface

# UI process manager for separate PyBoy process
UI_MANAGER_AVAILABLE = True

logger = logging.getLogger(__name__)


class PyBoyEmulator(EmulatorInterface):
    """PyBoy emulator implementation using official API patterns"""

    def __init__(self):
        self.pyboy = None
        self.rom_path = None
        self.initialized = False
        self.game_title = ""
        self.auto_launch_ui = True
        self.ui_launched = False
        self.game_wrapper = None
        self.ui_process = None
        self.ui_thread = None
        self._emulator_lock = threading.RLock()

        # Performance optimization attributes - cache DISABLED by default to save RAM
        self._screen_cache = {}
        self._screen_cache_enabled = False  # Disabled to reduce memory usage
        self._last_screen_hash = None
        self._frame_counter = 0
        self._fps_tracker = []
        self._last_fps_time = time.time()
        self._performance_stats = {
            'screen_captures': 0,
            'cache_hits': 0,
            'conversion_time': 0,
            'avg_fps': 0
        }

    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file into the PyBoy emulator using official API"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy is not available. Please install it with 'pip install pyboy'")

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        try:
            with self._emulator_lock:
                # Initialize PyBoy using the official API pattern
                logger.info(f"Initializing PyBoy with ROM: {os.path.basename(rom_path)}")

                # Use SDL2 window with hidden display for proper screen rendering
                # PyBoy requires actual SDL2 window to populate screen buffer
                # SDL_WINDOW_HIDDEN makes window invisible while still rendering
                os.environ['SDL_WINDOW_HIDDEN'] = '1'
                os.environ['SDL_AUDIODRIVER'] = 'disk'
                
                self.pyboy = PyBoy(
                    rom_path,
                    window="headless",
                    scale=2,
                    sound_emulated=False,
                    sound_volume=0
                )

                # Set emulation speed to unlimited for AI training
                self.pyboy.set_emulation_speed(0)

                # Initialize game wrapper if available
                self.game_wrapper = self.pyboy.game_wrapper

                # Store basic info
                self.rom_path = rom_path
                self.initialized = True
                self.game_title = self.pyboy.cartridge_title

                # Log successful initialization
                logger.info(f"PyBoy initialized successfully")
                logger.info(f"Game title: {self.game_title}")
                logger.info(f"Window type: null")
                logger.info(f"Emulation speed: unlimited (0)")

                # Warm up the emulator long enough to populate a meaningful framebuffer.
                # A single frame often stays blank/white during Game Boy boot.
                self.pyboy.tick(180, True)

            # Launch UI in separate process if requested
            if self.auto_launch_ui:
                self._launch_ui_process()

            return True

        except Exception as e:
            logger.error(f"Error loading ROM: {e}")
            return False

    def _validate_rom_path(self, rom_path: str) -> bool:
        """Validate ROM path for security"""
        if not rom_path or not isinstance(rom_path, str):
            return False

        # Normalize path to prevent directory traversal
        try:
            normalized_path = os.path.normpath(os.path.abspath(rom_path))
        except (OSError, ValueError):
            return False

        # Check if path is within allowed directories
        allowed_dirs = [
            os.path.abspath(os.path.dirname(self.rom_path)) if self.rom_path else "",
            os.path.abspath("C:\\Users\\Ryan\\Desktop\\ROMS\\PyGB")
        ]

        if not any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_dirs if allowed_dir):
            return False

        # Check file extension
        if not normalized_path.lower().endswith(('.gb', '.gbc', '.rom')):
            return False

        # Check if file exists and is a file
        if not os.path.isfile(normalized_path):
            return False

        # Check file size (max 16MB for Game Boy ROMs)
        try:
            file_size = os.path.getsize(normalized_path)
            if file_size > 16 * 1024 * 1024:  # 16MB
                return False
        except OSError:
            return False

        return True

    def _launch_ui_process(self):
        """Launch PyBoy UI in a separate process using secure approach"""
        if not self.rom_path or not os.path.exists(self.rom_path):
            logger.error("Cannot launch UI - no ROM loaded")
            return

        # Validate ROM path for security
        if not self._validate_rom_path(self.rom_path):
            logger.error(f"Invalid ROM path: {self.rom_path}")
            return

        try:
            logger.info("=== LAUNCHING PYBOY UI PROCESS ===")
            logger.info(f"ROM path: {self.rom_path}")
            logger.info(f"ROM exists: {os.path.exists(self.rom_path)}")
            logger.info(f"Working directory: {os.getcwd()}")

            # Use secure template-based approach instead of dynamic script generation
            # Load the UI script from a predefined template file
            template_path = os.path.join(os.path.dirname(__file__), "ui_script_template.py")

            # If template doesn't exist, create a secure hardcoded version
            if not os.path.exists(template_path):
                self._create_ui_script_template(template_path)

            # Create a temporary script file with validated parameters
            import tempfile
            import shutil

            # Create secure temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                script_path = temp_script.name

                # Read template and substitute validated parameters
                with open(template_path, 'r') as template_file:
                    template_content = template_file.read()

                # Safe parameter substitution
                safe_rom_path = repr(self.rom_path)  # Properly escape the path
                safe_pyboy_path = repr("C:\\Users\\Ryan\\Desktop\\ROMS\\PyGB\\PyBoy")
                safe_project_path = repr("C:\\Users\\Ryan\\Desktop\\ROMS\\PyGB")

                # Substitute parameters safely
                script_content = template_content.replace('{{ROM_PATH}}', safe_rom_path)
                script_content = script_content.replace('{{PYBOY_PATH}}', safe_pyboy_path)
                script_content = script_content.replace('{{PROJECT_PATH}}', safe_project_path)

                temp_script.write(script_content)
                temp_script.flush()

                # Set secure file permissions
                os.chmod(script_path, 0o600)  # Read/write for owner only

            logger.info(f"UI script written to: {script_path}")

            # Launch the UI process with enhanced security
            logger.info("Starting UI subprocess...")

            # Use secure subprocess execution
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                env={
                    **os.environ,
                    'PYTHONUNBUFFERED': '1'
                }
            )
            if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP'):
                popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

            self.ui_process = subprocess.Popen([
                sys.executable, script_path
            ], **popen_kwargs)

            # Wait a moment and check if process started
            time.sleep(1)

            if self.ui_process.poll() is None:
                self.ui_launched = True
                logger.info(f"=== UI PROCESS LAUNCHED SUCCESSFULLY ===")
                logger.info(f"UI process PID: {self.ui_process.pid}")

                # Start a thread to monitor UI process output
                def monitor_ui_process():
                    try:
                        stdout, stderr = self.ui_process.communicate(timeout=5)
                        logger.info(f"UI process stdout: {stdout}")
                        if stderr:
                            logger.error(f"UI process stderr: {stderr}")
                    except subprocess.TimeoutExpired:
                        logger.info("UI process is running (timeout reading output)")
                    except Exception as monitor_e:
                        logger.error(f"Error monitoring UI process: {monitor_e}")

                monitor_thread = threading.Thread(target=monitor_ui_process)
                monitor_thread.daemon = True
                monitor_thread.start()

            else:
                # Process already terminated
                stdout, stderr = self.ui_process.communicate()
                logger.error(f"=== UI PROCESS FAILED TO START ===")
                logger.error(f"Exit code: {self.ui_process.returncode}")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                self.ui_launched = False

            # Clean up the script file after a delay
            def cleanup_script():
                time.sleep(5)
                try:
                    if os.path.exists(script_path):
                        os.remove(script_path)
                        logger.info(f"Cleaned up UI script: {script_path}")
                except Exception as cleanup_e:
                    logger.error(f"Failed to clean up script: {cleanup_e}")

            cleanup_thread = threading.Thread(target=cleanup_script)
            cleanup_thread.daemon = True
            cleanup_thread.start()

        except Exception as e:
            logger.error(f"=== FAILED TO LAUNCH UI PROCESS ===", exc_info=True)
            logger.error(f"Error: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.ui_launched = False

    def _create_ui_script_template(self, template_path: str):
        """Create a secure UI script template"""
        template_content = '''import sys
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

    # Sound settings from environment or defaults
    sound_enabled = os.environ.get('PYBOY_SOUND_ENABLED', 'true').lower() == 'true'
    sound_volume = int(os.environ.get('PYBOY_SOUND_VOLUME', '50'))
    sound_output = os.environ.get('PYBOY_SOUND_OUTPUT', 'false').lower() == 'true'
    
    # Configure SDL audio driver
    if sound_output:
        # Use system default for actual audio output
        if 'SDL_AUDIODRIVER' in os.environ:
            del os.environ['SDL_AUDIODRIVER']
        print("Sound output: enabled (system audio)")
    else:
        # Use dummy driver for silent operation
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        print("Sound output: disabled (silent mode)")
    
    print(f"Sound emulation: {'enabled' if sound_enabled else 'disabled'}")
    print(f"Sound volume: {sound_volume}%")
    
    pyboy = PyBoy(rom_path, window="headless", scale=2, sound_emulated=sound_enabled, sound_volume=sound_volume, debug=False)
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
'''

        # Write template with secure permissions
        with open(template_path, 'w') as f:
            f.write(template_content)
        os.chmod(template_path, 0o644)  # Read/write for owner, read for others
        logger.info(f"Created UI script template: {template_path}")

    def step(self, action: str, frames: int = 1) -> bool:
        """Execute an action for a number of frames using official PyBoy API"""
        if not self.initialized or self.pyboy is None:
            return False

        # Map actions to PyBoy buttons
        action_map = {
            'UP': 'up',
            'DOWN': 'down',
            'LEFT': 'left',
            'RIGHT': 'right',
            'A': 'a',
            'B': 'b',
            'START': 'start',
            'SELECT': 'select'
        }

        action = str(action or 'NOOP').upper()

        try:
            frame_count = max(int(frames), 1)
        except (TypeError, ValueError):
            frame_count = 1

        try:
            with self._emulator_lock:
                if action in action_map:
                    # Hold the button for the requested frames and advance one extra tick
                    # so PyBoy processes the release before the next action arrives.
                    button = action_map[action]
                    total_frames = frame_count + 1
                    self.pyboy.button(button, frame_count)
                else:
                    total_frames = frame_count

                for i in range(total_frames):
                    render_this_frame = (i == total_frames - 1)
                    self.pyboy.tick(1, render_this_frame)

            return True
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return False

    def advance_idle_frames(self, frames: int = 1, render_last: bool = False) -> bool:
        """Advance the emulator without injecting input."""
        if not self.initialized or self.pyboy is None:
            return False

        try:
            frame_count = max(int(frames), 1)
        except (TypeError, ValueError):
            frame_count = 1

        try:
            with self._emulator_lock:
                for i in range(frame_count):
                    render_this_frame = render_last and i == frame_count - 1
                    self.pyboy.tick(1, render_this_frame)
            return True
        except Exception as e:
            logger.error(f"Error advancing idle frames: {e}")
            return False

    def get_screen(self) -> np.ndarray:
        """Get the current screen as a numpy array - simple direct capture (no tick)"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, returning black screen")
            return np.zeros((144, 160, 3), dtype=np.uint8)

        try:
            with self._emulator_lock:
                # Direct screen capture - works in headless mode
                # Don't tick here - caller should tick before calling
                screen = self.pyboy.screen.ndarray
                
                # Convert RGBA to RGB if needed
                if screen.shape[2] == 4:
                    screen = screen[:, :, :3]
                
                return screen
        except Exception as e:
            logger.error(f"Error getting screen from PyBoy: {e}")
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def _calculate_screen_hash(self, screen_array: np.ndarray) -> Optional[str]:
        """Calculate a fast hash of the screen array for caching"""
        try:
            # Use a small sample of the screen for faster hashing
            if screen_array.size > 1000:
                # Sample every 4th pixel for hashing
                sample = screen_array[::4, ::4].tobytes()
            else:
                sample = screen_array.tobytes()

            return hashlib.md5(sample).hexdigest()
        except Exception:
            return None

    def _update_fps_counter(self, process_time: float):
        """Update FPS tracking counter"""
        current_time = time.time()
        self._performance_stats['conversion_time'] += process_time

        # Track FPS every second
        if current_time - self._last_fps_time >= 1.0:
            if len(self._fps_tracker) > 0:
                avg_fps = len(self._fps_tracker) / (current_time - self._last_fps_time)
                self._performance_stats['avg_fps'] = avg_fps
            self._fps_tracker = []
            self._last_fps_time = current_time

        self._fps_tracker.append(current_time)
        self._frame_counter += 1

    def get_performance_stats(self) -> dict:
        """Get current performance statistics"""
        return {
            **self._performance_stats,
            'cache_enabled': self._screen_cache_enabled,
            'cache_size': len(self._screen_cache),
            'frame_count': self._frame_counter,
            'cv2_available': CV2_AVAILABLE,
            'pil_available': PIL_AVAILABLE
        }

    def set_screen_caching(self, enabled: bool):
        """Enable or disable screen caching"""
        self._screen_cache_enabled = enabled
        if not enabled:
            self._screen_cache.clear()
        logger.info(f"Screen caching {'enabled' if enabled else 'disabled'}")

    def clear_screen_cache(self):
        """Clear the screen cache"""
        self._screen_cache.clear()
        self._last_screen_hash = None
        logger.info("Screen cache cleared")

    def get_screen_bytes(self) -> bytes:
        """Get the current screen as bytes with optimized conversion"""
        try:
            start_time = time.time()

            # Get screen as numpy array first (already cached)
            screen_array = self.get_screen()

            # Use OpenCV for faster conversion if available
            if CV2_AVAILABLE:
                # OpenCV is much faster for image encoding
                success, img_buffer = cv2.imencode('.jpg', screen_array, [cv2.IMWRITE_JPEG_QUALITY, 75, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                if success:
                    img_bytes = img_buffer.tobytes()
                else:
                    raise RuntimeError("OpenCV encoding failed")
            elif PIL_AVAILABLE:
                # Fallback to PIL with optimized settings
                img_buffer = io.BytesIO()
                Image.fromarray(screen_array).save(img_buffer, format='JPEG', quality=75, optimize=False, progressive=False)
                img_bytes = img_buffer.getvalue()
            else:
                # Ultimate fallback - raw bytes
                img_bytes = screen_array.tobytes()

            conversion_time = time.time() - start_time
            self._performance_stats['conversion_time'] += conversion_time

            logger.debug(f"Screen converted to bytes: {len(img_bytes)} bytes in {conversion_time:.3f}s")
            return img_bytes

        except Exception as e:
            logger.error(f"Error converting screen to bytes: {e}")
            # Return minimal valid JPEG data as fallback
            return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xd9'

    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Read memory from the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, returning zeros")
            return b'\x00' * size

        # Validate memory address range (Game Boy has 64KB address space)
        if address < 0 or address > 0xFFFF:
            logger.error(f"Invalid memory address: {hex(address)}")
            return b'\x00' * size

        if size < 1 or (address + size) > 0x10000:
            logger.error(f"Invalid memory size or range: address={hex(address)}, size={size}")
            return b'\x00' * size

        try:
            with self._emulator_lock:
                if size == 1:
                    return bytes([self.pyboy.memory[address]])
                else:
                    return bytes(self.pyboy.memory[address:address + size])
        except Exception as e:
            logger.error(f"Error reading memory at {hex(address)}: {e}")
            return b'\x00' * size

    def set_memory(self, address: int, value: bytes) -> bool:
        """Write memory to the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot write memory")
            return False

        # Validate memory address range
        if address < 0 or address > 0xFFFF:
            logger.error(f"Invalid memory address: {hex(address)}")
            return False

        if len(value) < 1 or (address + len(value)) > 0x10000:
            logger.error(f"Invalid memory write range: address={hex(address)}, size={len(value)}")
            return False

        try:
            with self._emulator_lock:
                if len(value) == 1:
                    self.pyboy.memory[address] = value[0]
                else:
                    self.pyboy.memory[address:address + len(value)] = list(value)
            logger.debug(f"Memory written to {hex(address)}: {value.hex()}")
            return True
        except Exception as e:
            logger.error(f"Error writing memory at {hex(address)}: {e}")
            return False

    def reset(self) -> bool:
        """Reset the emulator using simplified approach"""
        if not self.initialized or self.pyboy is None:
            return False

        try:
            with self._emulator_lock:
                # Stop the current emulator
                self.pyboy.stop()

                # Re-initialize PyBoy with simplified configuration
                self.pyboy = PyBoy(
                    self.rom_path,
                    window="headless" if self.auto_launch_ui else "null",
                    scale=2,
                    sound_emulated=True,
                    sound_volume=50
                )

                # Set emulation speed to unlimited for AI training
                self.pyboy.set_emulation_speed(0)

                # Re-initialize game wrapper
                self.game_wrapper = self.pyboy.game_wrapper

                # Start with one tick
                self.pyboy.tick(1, self.auto_launch_ui)

            logger.info("Emulator reset successfully")
            return True

        except Exception as e:
            logger.error(f"Error resetting emulator: {e}")
            return False

    def save_state(self) -> bytes:
        """Save the current state of the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot save state")
            return b''

        try:
            # Save state to bytes using PyBoy's save_state method
            state_buffer = io.BytesIO()
            with self._emulator_lock:
                self.pyboy.save_state(state_buffer)
            state_data = state_buffer.getvalue()
            logger.info(f"State saved successfully: {len(state_data)} bytes")
            return state_data
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return b''

    def load_state(self, state: bytes) -> bool:
        """Load a saved state into the emulator"""
        if not self.initialized or self.pyboy is None:
            logger.warning("PyBoy not initialized, cannot load state")
            return False

        if not state:
            logger.warning("Empty state data provided")
            return False

        try:
            # Load state from bytes using PyBoy's load_state method
            state_buffer = io.BytesIO(state)
            with self._emulator_lock:
                self.pyboy.load_state(state_buffer)
            logger.info(f"State loaded successfully: {len(state)} bytes")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def get_info(self) -> dict:
        """Get information about the current game state"""
        if not self.initialized or self.pyboy is None:
            return {"error": "PyBoy not initialized"}

        try:
            with self._emulator_lock:
                info = {
                    "rom_title": self.pyboy.cartridge_title,
                    "frame_count": self.pyboy.frame_count,
                    "initialized": self.initialized,
                    "game_title": self.game_title,
                    "rom_path": self.rom_path,
                    "emulation_speed": "unlimited"  # We set this to 0 for AI training
                }

                # Add screen information if available
                try:
                    screen_shape = self.pyboy.screen.ndarray.shape
                    info["screen_size"] = screen_shape
                    info["screen_format"] = "RGBA" if len(screen_shape) == 3 and screen_shape[2] == 4 else "RGB"
                except Exception as screen_e:
                    logger.warning(f"Could not get screen info: {screen_e}")
                    info["screen_size"] = "unknown"
                    info["screen_format"] = "unknown"

            return info
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return {"error": str(e), "initialized": self.initialized}

    def get_game_state_analysis(self) -> dict:
        """Get a detailed analysis of the current game state"""
        if not self.initialized or self.pyboy is None:
            return {}

        try:
            # Get basic info
            info = self.get_info()

            # Get screen analysis
            screen = self.get_screen()

            # Get memory regions of interest (this would be game-specific)
            # For now, we'll just get some general memory values
            memory_analysis = {}

            # Add game-specific analysis based on the game title
            game_specific = self._get_game_specific_analysis()

            return {
                "basic_info": info,
                "screen_analysis": {
                    "shape": screen.shape,
                    "mean_color": screen.mean(axis=(0,1)).tolist(),
                    "unique_colors": len(np.unique(screen.reshape(-1, screen.shape[2]), axis=0))
                },
                "memory_analysis": memory_analysis,
                "game_specific": game_specific
            }
        except Exception as e:
            print(f"Error getting game state analysis: {e}")
            return {}

    def _get_game_specific_analysis(self) -> dict:
        """Get game-specific analysis based on the game title"""
        if not self.initialized or self.pyboy is None:
            return {}

        game_title = self.game_title.lower()

        # Placeholder for game-specific analysis
        # In a real implementation, this would have specific logic for each game
        if "pokemon" in game_title:
            return self._get_pokemon_analysis()
        elif "tetris" in game_title:
            return self._get_tetris_analysis()
        elif "mario" in game_title:
            return self._get_mario_analysis()
        else:
            return {"game_type": "unknown", "analysis": "No specific analysis available for this game"}

    # =========================================
    # Pokemon Red/Blue Memory Addresses (Gen 1)
    # =========================================
    POKEMON_MEMORY = {
        # Player position
        "player_x": 0xD062,
        "player_y": 0xD063,
        "map_id": 0xD35E,
        
        # Party Pokemon (base address 0xD163, 44 bytes each)
        "party_count": 0xD163,
        "party_base": 0xD16C,  # First Pokemon species
        "party_size": 44,
        "party_species_offset": 0x00,
        "party_hp_offset": 0x1E,  # 2 bytes little-endian
        "party_max_hp_offset": 0x20,
        "party_level_offset": 0x18,
        "party_status_offset": 0x1C,
        "party_type1_offset": 0x12,
        "party_type2_offset": 0x13,
        
        # Money (3 bytes BCD)
        "money": 0xD6F5,
        
        # Inventory
        "inventory_base": 0xD6E5,
        "inventory_count": 0xD6E5,  # First byte is count
        
        # Battle state
        "battle_status": 0xD057,
        "battle_type": 0xD058,
        
        # Enemy Pokemon
        "enemy_base": 0xD883,
        "enemy_species_offset": 0x00,
        "enemy_hp_offset": 0x0C,
        "enemy_max_hp_offset": 0x10,
        "enemy_level_offset": 0x16,
        
        # Badges
        "badges": 0xD8F6,
    }
    
    # Pokemon species ID to name (Gen 1)
    POKEMON_NAMES = {
        0: None,  # Empty slot
        1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur", 4: "Charmander", 5: "Charmeleon",
        6: "Charizard", 7: "Squirtle", 8: "Wartortle", 9: "Blastoise", 10: "Caterpie",
        11: "Metapod", 12: "Butterfree", 13: "Weedle", 14: "Kakuna", 15: "Beedrill",
        16: "Pidgey", 17: "Pidgeotto", 18: "Pidgeot", 19: "Rattata", 20: "Raticate",
        21: "Spearow", 22: "Fearow", 23: "Ekans", 24: "Arbok", 25: "Pikachu",
        26: "Raichu", 27: "Sandshrew", 28: "Sandslash", 29: "Nidoran♀", 30: "Nidorina",
        31: "Nidoqueen", 32: "Nidoran♂", 33: "Nidorino", 34: "Nidoking", 35: "Clefairy",
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
        96: "Hypno", 97: "Krabby", 98: "Kingler", 99: "Voltorb", 100: "Electrode",
        101: "Exeggcute", 102: "Exeggutor", 103: "Cubone", 104: "Marowak", 105: "Hitmonlee",
        106: "Hitmonchan", 107: "Lickitung", 108: "Koffing", 109: "Weezing", 110: "Rhyhorn",
        111: "Rhydon", 112: "Chansey", 113: "Tangela", 114: "Kangaskhan", 115: "Horsea",
        116: "Seadra", 117: "Goldeen", 118: "Seaking", 119: "Staryu", 120: "Starmie",
        121: "Mr. Mime", 122: "Scyther", 123: "Jynx", 124: "Electabuzz", 125: "Magmar",
        126: "Pinsir", 127: "Tauros", 128: "Magikarp", 129: "Gyarados", 130: "Lapras",
        131: "Ditto", 132: "Eevee", 133: "Vaporeon", 134: "Jolteon", 135: "Flareon",
        136: "Porygon", 137: "Omanyte", 138: "Omastar", 139: "Kabuto", 140: "Kabutops",
        141: "Aerodactyl", 142: "Snorlax", 143: "Articuno", 144: "Zapdos", 145: "Moltres",
        146: "Dratini", 147: "Dragonair", 148: "Dragonite", 149: "Mewtwo", 150: "Mew",
    }
    
    # Item ID to name (Gen 1)
    ITEM_NAMES = {
        0: None,
        1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poké Ball",
        5: "Town Map", 6: "Bicycle", 7: "?????", 8: "Safari Ball",
        9: "Moon Stone", 10: "Antidote", 11: "Burn Heal", 12: "Ice Heal",
        13: "Awakening", 14: "Parlyz Heal", 15: "Full Restore", 16: "Max Potion",
        17: "Hyper Potion", 18: "Super Potion", 19: "Potion", 20: "Escape Rope",
        21: "Repel", 22: "Max Repel", 23: "Super Repel", 25: "Fire Stone",
        26: "Thunder Stone", 27: "Water Stone", 28: "HP Up", 29: "Protein",
        30: "Iron", 31: "Carbos", 32: "Calcium", 33: "Rare Candy",
        34: "X Accuracy", 35: "X Attack", 36: "X Defend", 37: "X Speed",
        38: "X Special", 39: "Poké Doll", 40: "Fresh Water", 41: "Soda Pop",
        42: "Lemonade", 43: "S.S. Ticket", 44: "Gold Teeth", 45: "X Accuracy",
        46: "Poké Flute", 47: "Secret Key", 48: "Bike Voucher", 49: "Card Key",
        50: "Lift Key", 52: "Exp. All", 53: "Old Amber", 54: "Helix Fossil",
        55: "Dome Fossil", 56: "Silph Scope", 57: "Town Map", 58: "Coin Case",
        59: "Itemfinder", 60: "Silph Scope", 61: "Poké Flute",
        64: "HM01 (Cut)", 65: "HM02 (Fly)", 66: "HM03 (Surf)", 67: "HM04 (Strength)", 68: "HM05 (Flash)",
    }
    
    # Type names
    TYPE_NAMES = {
        0: "Normal", 1: "Fighting", 2: "Flying", 3: "Poison", 4: "Ground",
        5: "Rock", 6: "Bird", 7: "Bug", 8: "Ghost",
        20: "Fire", 21: "Water", 22: "Grass", 23: "Electric", 24: "Psychic",
        25: "Ice", 26: "Dragon",
    }
    
    def _read_byte(self, address: int) -> int:
        """Read a single byte from memory safely"""
        if not self.initialized or self.pyboy is None:
            return 0
        try:
            with self._emulator_lock:
                return self.pyboy.memory[address]
        except Exception:
            return 0
    
    def _read_word(self, address: int) -> int:
        """Read a 2-byte word (little-endian) from memory safely"""
        if not self.initialized or self.pyboy is None:
            return 0
        try:
            with self._emulator_lock:
                low = self.pyboy.memory[address]
                high = self.pyboy.memory[address + 1]
                return low | (high << 8)
        except Exception:
            return 0
    
    def _read_bcd(self, address: int, size: int = 3) -> int:
        """Read BCD-encoded value (for money)"""
        try:
            value = 0
            for i in range(size):
                byte = self._read_byte(address + i)
                high = (byte >> 4) & 0x0F
                low = byte & 0x0F
                value = value * 100 + high * 10 + low
            return value
        except Exception:
            return 0
    
    def get_party_info(self) -> List[dict]:
        """Get detailed party Pokemon information"""
        if not self.initialized or self.pyboy is None:
            return []
        
        party = []
        try:
            party_count = self._read_byte(self.POKEMON_MEMORY["party_count"])
            party_count = min(party_count, 6)  # Max 6 Pokemon
            
            for i in range(party_count):
                # Party Pokemon are stored starting at 0xD16C (species list)
                # Then actual data starts at different addresses
                # Simplified: read from species list
                species_addr = 0xD16C + i
                species_id = self._read_byte(species_addr)
                
                if species_id == 0 or species_id == 0xFF:
                    continue
                
                # Pokemon data structure is complex in Gen 1
                # We'll use simplified addresses
                # Box Pokemon start at 0xD175, party stats at different locations
                
                # For now, use the simplified approach
                # Party stats base: each Pokemon has 44 bytes
                # But the actual layout is different
                
                # Let's use the battle Pokemon structure as reference
                # First Pokemon HP is at 0xD16C + offset
                
                # Actually, let's use the simpler approach from memory_scanner.py
                # which uses D163 as party count and D16C as species
                
                pokemon = {
                    "slot": i + 1,
                    "species_id": species_id,
                    "species_name": self.POKEMON_NAMES.get(species_id, f"Unknown ({species_id})"),
                }
                
                # Try to read HP from D16A + (i * 44) + 0x1E (current HP)
                # This is party Pokemon data
                base = 0xD16A + (i * 44)
                hp = self._read_word(base + 0x1E)
                max_hp = self._read_word(base + 0x20)
                level = self._read_byte(base + 0x18)
                status = self._read_byte(base + 0x1C)
                
                pokemon["hp"] = hp
                pokemon["max_hp"] = max_hp if max_hp > 0 else 1
                pokemon["level"] = level
                pokemon["status"] = status
                pokemon["status_text"] = self._decode_status(status)
                pokemon["hp_percent"] = round((hp / max_hp) * 100, 1) if max_hp > 0 else 0
                
                # Types
                type1 = self._read_byte(base + 0x12)
                type2 = self._read_byte(base + 0x13)
                pokemon["type1"] = self.TYPE_NAMES.get(type1, f"Type {type1}")
                pokemon["type2"] = self.TYPE_NAMES.get(type2) if type2 != type1 else None
                
                # Moves (simplified - just count)
                pokemon["moves"] = []  # Would need more complex parsing
                
                party.append(pokemon)
                
        except Exception as e:
            logger.debug(f"Error reading party: {e}")
        
        return party
    
    def get_inventory_info(self) -> dict:
        """Get player inventory and money"""
        if not self.initialized or self.pyboy is None:
            return {"money": 0, "items": []}
        
        result = {"money": 0, "items": []}
        
        try:
            # Read money (3 bytes BCD)
            result["money"] = self._read_bcd(self.POKEMON_MEMORY["money"], 3)
            
            # Read items
            items = []
            item_count = self._read_byte(self.POKEMON_MEMORY["inventory_count"])
            item_count = min(item_count, 20)  # Max 20 items
            
            for i in range(item_count):
                # Item slots: item_id (1 byte) + quantity (1 byte)
                addr = self.POKEMON_MEMORY["inventory_base"] + 1 + (i * 2)
                item_id = self._read_byte(addr)
                quantity = self._read_byte(addr + 1)
                
                if item_id == 0 or item_id == 0xFF:
                    continue
                
                items.append({
                    "slot": i + 1,
                    "id": item_id,
                    "name": self.ITEM_NAMES.get(item_id, f"Item {item_id}"),
                    "quantity": quantity,
                })
            
            result["items"] = items
            
        except Exception as e:
            logger.debug(f"Error reading inventory: {e}")
        
        return result
    
    def get_position(self) -> dict:
        """Get player position and map info"""
        if not self.initialized or self.pyboy is None:
            return {"x": 0, "y": 0, "map_id": 0, "map_name": "unknown"}
        
        try:
            x = self._read_byte(self.POKEMON_MEMORY["player_x"])
            y = self._read_byte(self.POKEMON_MEMORY["player_y"])
            map_id = self._read_byte(self.POKEMON_MEMORY["map_id"])
            
            # Known map names (partial list)
            map_names = {
                0x00: "Pallet Town",
                0x01: "Viridian City",
                0x02: "Pewter City",
                0x03: "Cerulean City",
                0x04: "Lavender Town",
                0x05: "Vermilion City",
                0x06: "Celadon City",
                0x07: "Fuchsia City",
                0x08: "Cinnabar Island",
                0x09: "Indigo Plateau",
                0x0A: "Saffron City",
                0x24: "Oak's Lab",
                0x25: "Player's House 1F",
                0x26: "Player's House 2F",
                0x27: "Rival's House",
                0x38: "Route 1",
                0x39: "Route 2",
                0x3A: "Route 3",
                0x3B: "Route 4",
            }
            
            return {
                "x": x,
                "y": y,
                "map_id": map_id,
                "map_name": map_names.get(map_id, f"Map {map_id}"),
            }
        except Exception as e:
            return {"x": 0, "y": 0, "map_id": 0, "map_name": "unknown", "error": str(e)}
    
    def get_battle_info(self) -> dict:
        """Get current battle state"""
        if not self.initialized or self.pyboy is None:
            return {"in_battle": False, "battle_type": "none"}
        
        try:
            battle_status = self._read_byte(self.POKEMON_MEMORY["battle_status"])
            battle_type = self._read_byte(self.POKEMON_MEMORY["battle_type"])
            
            in_battle = battle_status != 0
            
            battle_types = {
                0: "none",
                1: "wild",
                2: "trainer",
            }
            
            result = {
                "in_battle": in_battle,
                "battle_status": battle_status,
                "battle_type": battle_types.get(battle_type, "unknown"),
            }
            
            if in_battle:
                # Read enemy Pokemon info
                enemy_base = self.POKEMON_MEMORY["enemy_base"]
                enemy_species = self._read_byte(enemy_base + self.POKEMON_MEMORY["enemy_species_offset"])
                enemy_hp = self._read_word(enemy_base + self.POKEMON_MEMORY["enemy_hp_offset"])
                enemy_max_hp = self._read_word(enemy_base + self.POKEMON_MEMORY["enemy_max_hp_offset"])
                enemy_level = self._read_byte(enemy_base + self.POKEMON_MEMORY["enemy_level_offset"])
                
                result["enemy"] = {
                    "species_id": enemy_species,
                    "species_name": self.POKEMON_NAMES.get(enemy_species, f"Pokemon {enemy_species}"),
                    "level": enemy_level,
                    "hp": enemy_hp,
                    "max_hp": enemy_max_hp,
                    "hp_percent": round((enemy_hp / enemy_max_hp) * 100, 1) if enemy_max_hp > 0 else 0,
                }
            
            return result
            
        except Exception as e:
            return {"in_battle": False, "battle_type": "none", "error": str(e)}
    
    def _decode_status(self, status_byte: int) -> str:
        """Decode status byte to readable text"""
        if status_byte == 0:
            return "OK"
        
        statuses = []
        # Gen 1 status flags (simplified)
        if status_byte & 0x07:  # Sleep counter (0-7 turns)
            statuses.append("sleep")
        if status_byte & 0x08:
            statuses.append("poison")
        if status_byte & 0x10:
            statuses.append("burn")
        if status_byte & 0x20:
            statuses.append("freeze")
        if status_byte & 0x40:
            statuses.append("paralyze")
        
        return ", ".join(statuses) if statuses else "OK"
    
    def _get_pokemon_analysis(self) -> dict:
        """Get Pokemon-specific game analysis"""
        try:
            position = self.get_position()
            party = self.get_party_info()
            battle = self.get_battle_info()
            inventory = self.get_inventory_info()
            
            return {
                "game_type": "pokemon",
                "player_position": position,
                "current_party": party,
                "battle_status": battle,
                "money": inventory.get("money", 0),
            }
        except Exception as e:
            return {
                "game_type": "pokemon",
                "player_position": "unknown",
                "current_party": [],
                "battle_status": "unknown",
                "error": str(e),
            }

    def _get_tetris_analysis(self) -> dict:
        """Get Tetris-specific game analysis"""
        # Placeholder for Tetris-specific analysis
        return {
            "game_type": "tetris",
            "current_piece": "unknown",
            "next_piece": "unknown",
            "lines_cleared": 0
        }

    def _get_mario_analysis(self) -> dict:
        """Get Mario-specific game analysis"""
        # Placeholder for Mario-specific analysis
        return {
            "game_type": "mario",
            "player_position": "unknown",
            "lives": 0,
            "coins": 0
        }

    # UI Management Methods (Simplified)
    def set_auto_launch_ui(self, enabled: bool):
        """Enable or disable automatic UI launching"""
        self.auto_launch_ui = enabled
        logger.info(f"Auto-launch UI set to: {enabled}")

    def get_auto_launch_ui(self) -> bool:
        """Get current auto-launch UI setting"""
        return self.auto_launch_ui

    def get_ui_status(self) -> dict:
        """Get UI status (simplified)"""
        return {
            "running": self.auto_launch_ui and self.initialized,
            "ready": self.initialized,
            "rom_path": self.rom_path,
            "window_type": "SDL2" if self.auto_launch_ui else "null",
            "auto_launch_enabled": self.auto_launch_ui,
            "ui_launched": self.ui_launched
        }

    def cleanup(self) -> bool:
        """Clean up resources and stop the emulator"""
        try:
            logger.info("Cleaning up PyBoy emulator resources")

            # Stop PyBoy emulator
            with self._emulator_lock:
                if self.pyboy is not None:
                    logger.info("Stopping PyBoy emulator")
                    self.pyboy.stop()
                    self.pyboy = None

            # Reset state
            self.initialized = False
            self.rom_path = None
            self.game_title = ""
            self.ui_launched = False
            self.game_wrapper = None

            logger.info("PyBoy emulator cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Error during PyBoy cleanup: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the emulator is currently running"""
        return self.initialized and self.pyboy is not None

    def get_frame_count(self) -> int:
        """Get the current frame count"""
        if not self.initialized or self.pyboy is None:
            return 0
        try:
            with self._emulator_lock:
                return self.pyboy.frame_count
        except Exception as e:
            logger.error(f"Error getting frame count: {e}")
            return 0

    def set_emulation_speed(self, speed: int) -> bool:
        """Set the emulation speed (0 = unlimited, 1 = normal, >1 = faster)"""
        if not self.initialized or self.pyboy is None:
            return False
        try:
            self.pyboy.set_emulation_speed(speed)
            logger.info(f"Emulation speed set to: {speed}")
            return True
        except Exception as e:
            logger.error(f"Error setting emulation speed: {e}")
            return False

    def run_game_loop(self, max_frames: int = 1000, render: bool = True) -> bool:
        """
        Run a standard game loop based on official PyBoy examples
        This method follows the pattern shown in PyBoy documentation

        Args:
            max_frames: Maximum number of frames to run
            render: Whether to render frames (for UI display)

        Returns:
            bool: True if loop completed successfully
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot run game loop")
            return False

        try:
            logger.info(f"Starting game loop: max_frames={max_frames}, render={render}")

            frame_count = 0
            while frame_count < max_frames:
                # Progress the emulator by one frame
                # Based on official example: while pyboy.tick(): pass
                if not self.pyboy.tick(1, render):
                    logger.info("Game loop ended naturally (tick returned False)")
                    break

                frame_count += 1

                # Log progress every 60 frames (1 second at 60fps)
                if frame_count % 60 == 0:
                    logger.debug(f"Game loop progress: {frame_count}/{max_frames} frames")

            logger.info(f"Game loop completed: {frame_count} frames processed")
            return True

        except Exception as e:
            logger.error(f"Error in game loop: {e}")
            return False

    def run_ai_training_loop(self, episodes: int = 10, max_frames_per_episode: int = 1000) -> dict:
        """
        Run an AI training loop with proper episode management
        This is optimized for AI training with minimal rendering

        Args:
            episodes: Number of training episodes to run
            max_frames_per_episode: Maximum frames per episode

        Returns:
            dict: Training statistics
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot run training loop")
            return {"error": "PyBoy not initialized"}

        try:
            logger.info(f"Starting AI training loop: episodes={episodes}, max_frames={max_frames_per_episode}")

            training_stats = {
                "episodes_completed": 0,
                "total_frames": 0,
                "episode_lengths": [],
                "success": True
            }

            for episode in range(episodes):
                logger.info(f"Starting episode {episode + 1}/{episodes}")

                # Reset emulator for new episode
                if episode > 0:
                    self.reset()

                episode_frames = 0
                episode_reward = 0  # Placeholder for reward tracking

                # Run episode with minimal rendering for performance
                while episode_frames < max_frames_per_episode:
                    # For AI training, we typically don't render every frame
                    # Only render on last frame for any needed screen capture
                    render_this_frame = (episode_frames == max_frames_per_episode - 1)

                    if not self.pyboy.tick(1, render_this_frame):
                        logger.info(f"Episode {episode + 1} ended naturally")
                        break

                    episode_frames += 1

                    # Here you would typically:
                    # 1. Get screen state
                    # 2. AI makes decision
                    # 3. Execute action
                    # 4. Calculate reward
                    # For now, just tick with NOOP

                    # Log progress
                    if episode_frames % 300 == 0:  # Every 5 seconds at 60fps
                        logger.debug(f"Episode {episode + 1}: {episode_frames}/{max_frames_per_episode} frames")

                # Record episode stats
                training_stats["episodes_completed"] += 1
                training_stats["total_frames"] += episode_frames
                training_stats["episode_lengths"].append(episode_frames)

                logger.info(f"Episode {episode + 1} completed: {episode_frames} frames")

            logger.info(f"Training loop completed: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            return {"error": str(e), "success": False}

    def capture_screenshot(self, filepath: str = None) -> bool:
        """
        Capture a screenshot using PyBoy's official screen API
        Based on the official example: pyboy.screen.image.save()

        Args:
            filepath: Path to save screenshot. If None, uses temp file.

        Returns:
            bool: True if screenshot captured successfully
        """
        if not self.initialized or self.pyboy is None:
            logger.error("PyBoy not initialized, cannot capture screenshot")
            return False

        try:
            # Ensure we have a rendered frame
            self.pyboy.tick(1, True)

            # Use PyBoy's screen.image property (PIL Image)
            screen_image = self.pyboy.screen.image

            if screen_image is None:
                logger.error("Screen image is None")
                return False

            # Generate filename if not provided
            if filepath is None:
                import tempfile
                filepath = tempfile.mktemp(suffix=".png")

            # Save the screenshot
            screen_image.save(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return False


class PyBoyEmulatorMP(EmulatorInterface):
    """
    Multi-process version of PyBoy emulator for improved performance and isolation
    """
    def __init__(self):
        self.pyboy_process = None
        self.command_queue = None
        self.result_queue = None
        self.stop_event = None
        self.initialized = False
        self.rom_path = None
        self.game_title = ""
        self.frame_count = 0

        # Performance attributes
        self._last_command_time = 0
        self._command_times = []

    def load_rom(self, rom_path: str) -> bool:
        """Load a ROM file in a separate process"""
        if not MP_AVAILABLE:
            logger.warning("Multi-processing not available, falling back to single process")
            # Fall back to regular PyBoyEmulator
            fallback_emulator = PyBoyEmulator()
            success = fallback_emulator.load_rom(rom_path)
            if success:
                # Transfer state to this instance
                self.pyboy_process = fallback_emulator
                self.initialized = True
                self.rom_path = rom_path
                self.game_title = fallback_emulator.game_title
            return success

        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy is not available. Please install it with 'pip install pyboy'")

        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")

        try:
            logger.info(f"Initializing multi-process PyBoy with ROM: {os.path.basename(rom_path)}")

            # Create communication channels
            self.command_queue = Queue()
            self.result_queue = Queue()
            self.stop_event = Event()

            # Start PyBoy in separate process
            self.pyboy_process = Process(
                target=self._pyboy_worker,
                args=(rom_path, self.command_queue, self.result_queue, self.stop_event),
                daemon=True
            )
            self.pyboy_process.start()

            # Wait for initialization
            try:
                result = self.result_queue.get(timeout=10.0)
                if result.get('status') == 'initialized':
                    self.initialized = True
                    self.rom_path = rom_path
                    self.game_title = result.get('game_title', '')
                    logger.info(f"Multi-process PyBoy initialized successfully")
                    return True
                else:
                    logger.error(f"Initialization failed: {result.get('error', 'Unknown error')}")
                    return False
            except Exception as e:
                logger.error(f"Timeout waiting for PyBoy initialization: {e}")
                self._cleanup_process()
                return False

        except Exception as e:
            logger.error(f"Error loading ROM in multi-process mode: {e}")
            return False

    def _pyboy_worker(self, rom_path: str, command_queue: Queue, result_queue: Queue, stop_event: Event):
        """Worker function that runs PyBoy in a separate process"""
        import os
        try:
            # Initialize PyBoy in worker process with headless mode
            os.environ['SDL_WINDOW_HIDDEN'] = '1'
            # Use dummy audio driver for silent operation in multi-process mode
            # (multi-process mode is typically for headless/server use)
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            
            # Sound settings from environment or defaults
            sound_enabled = os.environ.get('PYBOY_SOUND_ENABLED', 'true').lower() == 'true'
            sound_volume = int(os.environ.get('PYBOY_SOUND_VOLUME', '50'))

            pyboy = PyBoy(
                rom_path,
                window="headless",
                scale=2,
                sound_emulated=sound_enabled,
                sound_volume=sound_volume
            )
            pyboy.set_emulation_speed(0)

            # Signal successful initialization
            result_queue.put({
                'status': 'initialized',
                'game_title': pyboy.cartridge_title
            })

            # Process commands
            while not stop_event.is_set():
                try:
                    # Wait for command with timeout
                    try:
                        command = command_queue.get(timeout=0.1)
                    except:
                        continue

                    cmd_type = command.get('type')
                    cmd_id = command.get('id')

                    if cmd_type == 'step':
                        action = command.get('action', 'NOOP')
                        frames = command.get('frames', 1)
                        success = self._execute_step(pyboy, action, frames)
                        result_queue.put({'id': cmd_id, 'success': success})

                    elif cmd_type == 'get_screen':
                        screen_array = self._get_screen_array(pyboy)
                        result_queue.put({'id': cmd_id, 'screen': screen_array})

                    elif cmd_type == 'get_screen_bytes':
                        screen_bytes = self._get_screen_bytes_worker(pyboy)
                        result_queue.put({'id': cmd_id, 'bytes': screen_bytes})

                    elif cmd_type == 'get_memory':
                        address = command.get('address')
                        size = command.get('size', 1)
                        memory_data = self._get_memory_worker(pyboy, address, size)
                        result_queue.put({'id': cmd_id, 'memory': memory_data})

                    elif cmd_type == 'set_memory':
                        address = command.get('address')
                        value = command.get('value')
                        success = self._set_memory_worker(pyboy, address, value)
                        result_queue.put({'id': cmd_id, 'success': success})

                    elif cmd_type == 'reset':
                        success = self._reset_worker(pyboy, rom_path)
                        result_queue.put({'id': cmd_id, 'success': success})

                    elif cmd_type == 'get_info':
                        info = self._get_info_worker(pyboy)
                        result_queue.put({'id': cmd_id, 'info': info})

                    elif cmd_type == 'get_frame_count':
                        frame_count = pyboy.frame_count if pyboy else 0
                        result_queue.put({'id': cmd_id, 'frame_count': frame_count})

                    elif cmd_type == 'stop':
                        break

                except Exception as e:
                    logger.error(f"Error processing command in worker: {e}")
                    result_queue.put({'id': cmd_id, 'error': str(e)})

        except Exception as e:
            logger.error(f"Critical error in PyBoy worker: {e}")
            result_queue.put({'status': 'error', 'error': str(e)})

        finally:
            # Clean up PyBoy
            if 'pyboy' in locals() and pyboy:
                pyboy.stop()

    def _execute_step(self, pyboy, action: str, frames: int) -> bool:
        """Execute action in worker process"""
        action_map = {
            'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
            'A': 'a', 'B': 'b', 'START': 'start', 'SELECT': 'select'
        }

        action = str(action or 'NOOP').upper()

        try:
            frame_count = max(int(frames), 1)
        except (TypeError, ValueError):
            frame_count = 1

        try:
            if action in action_map:
                button = action_map[action]
                total_frames = frame_count + 1
                pyboy.button(button, frame_count)
            else:
                total_frames = frame_count

            for i in range(total_frames):
                pyboy.tick(1, i == total_frames - 1)
            return True
        except Exception:
            return False

    def _get_screen_array(self, pyboy) -> np.ndarray:
        """Get screen array in worker process"""
        try:
            screen_array = pyboy.screen.ndarray
            if screen_array is None or screen_array.size == 0:
                return np.zeros((144, 160, 3), dtype=np.uint8)

            if len(screen_array.shape) == 3 and screen_array.shape[2] == 4:
                screen_array = screen_array[:, :, :3]

            if screen_array.dtype != np.uint8:
                screen_array = screen_array.astype(np.uint8)

            return screen_array
        except Exception:
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def _get_screen_bytes_worker(self, pyboy) -> bytes:
        """Get screen bytes in worker process"""
        try:
            screen_array = self._get_screen_array(pyboy)
            if CV2_AVAILABLE:
                success, img_buffer = cv2.imencode('.jpg', screen_array, [
                    cv2.IMWRITE_JPEG_QUALITY, 75,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ])
                if success:
                    return img_buffer.tobytes()

            # Fallback to raw bytes
            return screen_array.tobytes()
        except Exception:
            return b''

    def _get_memory_worker(self, pyboy, address: int, size: int) -> bytes:
        """Get memory in worker process"""
        try:
            if size == 1:
                return bytes([pyboy.memory[address]])
            else:
                return bytes(pyboy.memory[address:address + size])
        except Exception:
            return b'\x00' * size

    def _set_memory_worker(self, pyboy, address: int, value: bytes) -> bool:
        """Set memory in worker process"""
        try:
            if len(value) == 1:
                pyboy.memory[address] = value[0]
            else:
                pyboy.memory[address:address + len(value)] = list(value)
            return True
        except Exception:
            return False

    def _reset_worker(self, pyboy, rom_path: str) -> bool:
        """Reset emulator in worker process"""
        try:
            pyboy.stop()
            # Sound settings from environment or defaults
            sound_enabled = os.environ.get('PYBOY_SOUND_ENABLED', 'true').lower() == 'true'
            sound_volume = int(os.environ.get('PYBOY_SOUND_VOLUME', '50'))
            
            pyboy = PyBoy(
                rom_path,
                window="null",
                scale=2,
                sound_emulated=sound_enabled,
                sound_volume=sound_volume
            )
            pyboy.set_emulation_speed(0)
            pyboy.tick(1, False)
            return True
        except Exception:
            return False

    def _get_info_worker(self, pyboy) -> dict:
        """Get info in worker process"""
        try:
            return {
                "rom_title": pyboy.cartridge_title,
                "frame_count": pyboy.frame_count,
                "initialized": True
            }
        except Exception:
            return {"error": "Failed to get info"}

    def _send_command(self, command_type: str, **kwargs) -> dict:
        """Send command to worker process and get result"""
        if not self.initialized or not self.pyboy_process:
            return {"error": "Emulator not initialized"}

        cmd_id = f"cmd_{int(time.time() * 1000000)}"
        command = {'type': command_type, 'id': cmd_id, **kwargs}

        try:
            self.command_queue.put(command)
            result = self.result_queue.get(timeout=5.0)

            if result.get('id') == cmd_id:
                return result
            else:
                return {"error": "Command ID mismatch"}

        except Exception as e:
            logger.error(f"Command {command_type} failed: {e}")
            return {"error": str(e)}

    def step(self, action: str, frames: int = 1) -> bool:
        """Execute action via multi-process communication"""
        start_time = time.time()
        result = self._send_command('step', action=action, frames=frames)
        self._track_command_time(time.time() - start_time)
        return result.get('success', False)

    def get_screen(self) -> np.ndarray:
        """Get screen via multi-process communication"""
        start_time = time.time()
        result = self._send_command('get_screen')
        screen_data = result.get('screen')
        self._track_command_time(time.time() - start_time)

        if isinstance(screen_data, np.ndarray):
            return screen_data
        else:
            return np.zeros((144, 160, 3), dtype=np.uint8)

    def get_screen_bytes(self) -> bytes:
        """Get screen bytes via multi-process communication"""
        start_time = time.time()
        result = self._send_command('get_screen_bytes')
        bytes_data = result.get('bytes', b'')
        self._track_command_time(time.time() - start_time)
        return bytes_data

    def get_memory(self, address: int, size: int = 1) -> bytes:
        """Get memory via multi-process communication"""
        result = self._send_command('get_memory', address=address, size=size)
        return result.get('memory', b'\x00' * size)

    def set_memory(self, address: int, value: bytes) -> bool:
        """Set memory via multi-process communication"""
        result = self._send_command('set_memory', address=address, value=value)
        return result.get('success', False)

    def reset(self) -> bool:
        """Reset via multi-process communication"""
        result = self._send_command('reset')
        return result.get('success', False)

    def get_info(self) -> dict:
        """Get info via multi-process communication"""
        result = self._send_command('get_info')
        return result.get('info', {"error": "Failed to get info"})

    def get_frame_count(self) -> int:
        """Get frame count via multi-process communication"""
        result = self._send_command('get_frame_count')
        return result.get('frame_count', 0)

    def _track_command_time(self, command_time: float):
        """Track command execution time for performance monitoring"""
        self._command_times.append(command_time)
        if len(self._command_times) > 100:
            self._command_times.pop(0)

    def _cleanup_process(self):
        """Clean up the worker process"""
        if self.stop_event:
            self.stop_event.set()

        if self.pyboy_process and self.pyboy_process.is_alive():
            try:
                self.pyboy_process.join(timeout=2.0)
                if self.pyboy_process.is_alive():
                    self.pyboy_process.terminate()
            except Exception as e:
                logger.error(f"Error cleaning up process: {e}")

        self.pyboy_process = None
        self.command_queue = None
        self.result_queue = None
        self.stop_event = None

    def cleanup(self) -> bool:
        """Clean up multi-process resources"""
        try:
            logger.info("Cleaning up multi-process PyBoy emulator")
            self._cleanup_process()
            self.initialized = False
            self.rom_path = None
            self.game_title = ""
            logger.info("Multi-process PyBoy cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Error during multi-process cleanup: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the emulator is running"""
        return self.initialized and self.pyboy_process and self.pyboy_process.is_alive()

    def get_performance_stats(self) -> dict:
        """Get performance statistics for multi-process mode"""
        if not self._command_times:
            return {"mode": "multi-process", "status": "no_data"}

        avg_command_time = sum(self._command_times) / len(self._command_times)
        return {
            "mode": "multi-process",
            "avg_command_time_ms": round(avg_command_time * 1000, 2),
            "command_count": len(self._command_times),
            "process_alive": self.pyboy_process.is_alive() if self.pyboy_process else False,
            "queue_sizes": {
                "command_queue": self.command_queue.qsize() if self.command_queue else 0,
                "result_queue": self.result_queue.qsize() if self.result_queue else 0
            }
        }
