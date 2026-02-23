#!/usr/bin/env python3
"""
OpenClaw Game Agent - Adapted from ClaudePlayer
Model-agnostic: Works with ANY OpenClaw model (Gemini, Qwen, GLM, local, etc.)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️  PyBoy not installed")

from PIL import Image
from io import BytesIO
import base64

# Model provider configurations
MODEL_PROVIDERS = {
    # Google Gemini
    'gemini-3-flash': {
        'model': 'google-gemini-cli/gemini-3-flash-preview',
        'vision': True,
        'cost': 'unlimited'
    },
    'gemini-3-pro': {
        'model': 'google-gemini-cli/gemini-3-pro-preview',
        'vision': True,
        'cost': '~100/day'
    },
    
    # Alibaba Qwen
    'qwen-3.5-plus': {
        'model': 'bailian/qwen3.5-plus',
        'vision': True,
        'cost': '18K/month'
    },
    'qwen-vl': {
        'model': 'bailian/qwen-vl-max',
        'vision': True,
        'cost': 'API credits'
    },
    
    # GLM (Z.ai)
    'glm-5': {
        'model': 'zai/glm-5',
        'vision': False,
        'cost': 'API credits'
    },
    'glm-4.7': {
        'model': 'zai/glm-4.7',
        'vision': False,
        'cost': 'API credits'
    },
    
    # MiniMax
    'minimax': {
        'model': 'minimax-portal/MiniMax-M2.5',
        'vision': False,
        'cost': 'free'
    },
    
    # Local LM Studio
    'lmstudio-jan': {
        'model': 'lmstudio/jan-v3-4b',
        'vision': False,
        'cost': 'free (local)'
    },
    'lmstudio-glm': {
        'model': 'lmstudio/glm-4.7-flash',
        'vision': False,
        'cost': 'free (local)'
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openclaw-game-agent")


class OpenClawGameAgent:
    """
    AI-powered Game Boy agent using OpenClaw models
    Model-agnostic: Works with ANY OpenClaw model
    
    Supported Models:
    - Gemini 3 Flash/Pro (vision capable)
    - Qwen 3.5 Plus, Qwen-VL (vision capable)
    - GLM-5, GLM-4.7
    - MiniMax M2.5
    - Local LM Studio models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rom_path = config.get('ROM_PATH', '')
        self.emulation_mode = config.get('EMULATION_MODE', 'turn_based')
        self.log_file = config.get('LOG_FILE', 'game_agent.log')
        self.max_history = config.get('MAX_HISTORY_MESSAGES', 30)
        self.max_screenshots = config.get('MAX_SCREENSHOTS', 5)
        self.custom_instructions = config.get('CUSTOM_INSTRUCTIONS', '')
        
        # Model selection (can be any OpenClaw model)
        model_config = config.get('MODEL_DEFAULTS', {})
        self.model_name = model_config.get('MODEL', 'google-gemini-cli/gemini-3-flash-preview')
        self.model_info = self._get_model_info(self.model_name)
        
        # Game state
        self.emulator: Optional[PyBoy] = None
        self.frame_count = 0
        self.memory: List[str] = []
        self.history: List[Dict] = []
        self.current_goal: str = ""
        self.game_name: str = ""
        
        # Setup
        self._setup_logging()
        self._init_emulator()
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model capabilities by name or alias"""
        # Check aliases first
        for alias, info in MODEL_PROVIDERS.items():
            if alias in model_name.lower() or info['model'] == model_name:
                return {
                    'alias': alias,
                    'full_name': info['model'],
                    'vision': info['vision'],
                    'cost': info['cost']
                }
        
        # Default: assume vision-capable for unknown models
        return {
            'alias': 'custom',
            'full_name': model_name,
            'vision': True,  # Assume vision support
            'cost': 'unknown'
        }
    
    def _setup_logging(self):
        """Setup file logging"""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    def _init_emulator(self):
        """Initialize PyBoy emulator"""
        if not PYBOY_AVAILABLE:
            raise RuntimeError("PyBoy not installed")
        
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"ROM not found: {self.rom_path}")
        
        logger.info(f"Loading ROM: {self.rom_path}")
        self.emulator = PyBoy(self.rom_path, window="null")
        self.game_name = Path(self.rom_path).stem
        logger.info(f"Game loaded: {self.game_name}")
    
    def get_frame(self) -> Dict[str, Any]:
        """Capture current frame as base64"""
        if self.emulator is None:
            return {'error': 'Emulator not initialized'}
        
        screen = self.emulator.screen
        if screen is None:
            return {'error': 'No screen buffer'}
        
        img = screen.image
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'base64': img_base64,
            'width': img.width,
            'height': img.height,
            'frame': self.frame_count
        }
    
    def send_inputs(self, inputs: str) -> bool:
        """
        Send button inputs to emulator
        
        Input notation:
        - A: Press A once
        - A2: Hold A for 2 ticks
        - AB: Press A and B together
        - W: Wait 1 tick
        - R2 A U3: Right 2 ticks, A once, Up 3 ticks
        """
        if self.emulator is None:
            return False
        
        button_map = {
            'A': WindowEvent.PRESS_BUTTON_A,
            'B': WindowEvent.PRESS_BUTTON_B,
            'U': WindowEvent.PRESS_ARROW_UP,
            'D': WindowEvent.PRESS_ARROW_DOWN,
            'L': WindowEvent.PRESS_ARROW_LEFT,
            'R': WindowEvent.PRESS_ARROW_RIGHT,
            'S': WindowEvent.PRESS_BUTTON_START,
            'X': WindowEvent.PRESS_BUTTON_SELECT,
        }
        
        release_map = {
            'A': WindowEvent.RELEASE_BUTTON_A,
            'B': WindowEvent.RELEASE_BUTTON_B,
            'U': WindowEvent.RELEASE_ARROW_UP,
            'D': WindowEvent.RELEASE_ARROW_DOWN,
            'L': WindowEvent.RELEASE_ARROW_LEFT,
            'R': WindowEvent.RELEASE_ARROW_RIGHT,
            'S': WindowEvent.RELEASE_BUTTON_START,
            'X': WindowEvent.RELEASE_BUTTON_SELECT,
        }
        
        try:
            # Parse input string
            tokens = inputs.strip().split()
            
            for token in tokens:
                if token.upper() == 'W':
                    # Wait
                    self.emulator.tick()
                    self.frame_count += 1
                    continue
                
                # Check for button + count (e.g., A2, R3)
                button = token[0].upper()
                count = int(token[1:]) if len(token) > 1 and token[1:].isdigit() else 1
                
                if button not in button_map:
                    logger.warning(f"Unknown button: {button}")
                    continue
                
                # Press and hold for count ticks
                for _ in range(count):
                    self.emulator.send_input(button_map[button])
                    self.emulator.tick()
                    self.emulator.send_input(release_map[button])
                    self.frame_count += 1
            
            logger.info(f"Inputs sent: {inputs}")
            return True
            
        except Exception as e:
            logger.error(f"Input failed: {e}")
            return False
    
    def add_to_memory(self, item: str):
        """Add item to memory"""
        self.memory.append(item)
        logger.info(f"Memory added: {item}")
        
        # Trim if too long
        if len(self.memory) > self.max_history:
            self.memory.pop(0)
    
    def remove_from_memory(self, index: int) -> bool:
        """Remove item from memory by index"""
        if 0 <= index < len(self.memory):
            removed = self.memory.pop(index)
            logger.info(f"Memory removed: {removed}")
            return True
        return False
    
    def update_memory_item(self, index: int, new_item: str) -> bool:
        """Update memory item at index"""
        if 0 <= index < len(self.memory):
            old = self.memory[index]
            self.memory[index] = new_item
            logger.info(f"Memory updated: {old} -> {new_item}")
            return True
        return False
    
    def set_current_goal(self, goal: str):
        """Set current gameplay objective"""
        self.current_goal = goal
        logger.info(f"Goal set: {goal}")
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state"""
        return {
            'game': self.game_name,
            'frame': self.frame_count,
            'goal': self.current_goal,
            'memory': self.memory,
            'emulator_initialized': self.emulator is not None
        }
    
    def save_screenshot(self, output_dir: str = None) -> Optional[str]:
        """Save current frame to file"""
        if self.emulator is None:
            return None
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "frames" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_data = self.get_frame()
        if 'error' in frame_data:
            return None
        
        # Decode and save
        img_data = base64.b64decode(frame_data['base64'])
        output_path = output_dir / f"frame_{self.frame_count:06d}.png"
        
        with open(output_path, 'wb') as f:
            f.write(img_data)
        
        logger.info(f"Screenshot saved: {output_path}")
        return str(output_path)
    
    def tick(self, render: bool = False) -> bool:
        """Advance emulator by one frame"""
        if self.emulator is None:
            return False
        
        self.emulator.tick(render)
        self.frame_count += 1
        return True
    
    def stop(self):
        """Stop emulator"""
        if self.emulator:
            self.emulator.stop()
            logger.info("Emulator stopped")
    
    def get_ai_decision(self, use_vision: bool = True) -> str:
        """
        Get AI decision using configured model
        
        Args:
            use_vision: Whether to send screenshot (if model supports it)
        
        Returns:
            Button inputs from AI (e.g., "A B START")
        """
        try:
            # Try OpenAI-compatible endpoint (OpenClaw Gateway)
            from openai import OpenAI
            
            client = OpenAI(
                base_url="http://localhost:18789/v1",
                api_key="not-needed"  # OpenClaw doesn't require key
            )
            
            # Build messages
            messages = [{
                "role": "system",
                "content": self.generate_prompt()
            }]
            
            # Add vision if supported and requested
            if use_vision and self.model_info['vision']:
                frame = self.get_frame()
                if 'error' not in frame:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What buttons should I press next?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame['base64']}"}}
                        ]
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": "What buttons should I press next? Respond with button notation (e.g., 'A B2 START')."
                })
            
            # Get AI response
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            decision = response.choices[0].message.content.strip()
            logger.info(f"AI decision ({self.model_name}): {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"AI decision failed: {e}")
            return "W"  # Default: wait
    
    def generate_prompt(self) -> str:
        """Generate prompt for AI model"""
        base_prompt = f"""You are playing {self.game_name} on Game Boy.

**Current Model:** {self.model_name} ({self.model_info['alias']})
**Vision:** {'✅ Enabled' if self.model_info['vision'] else '❌ Text-only'}
**Cost:** {self.model_info['cost']}

**Current Goal:** {self.current_goal or "Explore and progress through the game"}

**Memory:**
{chr(10).join(f"- {m}" for m in self.memory) if self.memory else "No memories yet"}

**Available buttons:** U (Up), D (Down), L (Left), R (Right), A, B, S (Start), X (Select), W (Wait)

**Input notation:**
- A: Press A once
- A2: Hold A for 2 ticks
- AB: Press A and B together
- W: Wait 1 tick
- R2 A U3: Right 2 ticks, A once, Up 3 ticks

Analyze the game screen and decide what buttons to press next. Be specific and strategic.

**Response format:** Just the button inputs (e.g., "A B START" or "R2 A U3"). No explanation needed."""

        if self.custom_instructions:
            base_prompt += f"\n\n**Custom Instructions:** {self.custom_instructions}"
        
        return base_prompt
    
    def run_turn(self, ai_response: str = None, use_vision: bool = None) -> bool:
        """
        Run a single turn
        
        Args:
            ai_response: Pre-computed button inputs (optional)
            use_vision: Whether to use vision (defaults to model capability)
        
        Returns:
            True if successful
        """
        # Get AI decision if not provided
        if ai_response is None:
            if use_vision is None:
                use_vision = self.model_info['vision']
            ai_response = self.get_ai_decision(use_vision=use_vision)
        
        logger.info(f"AI decision ({self.model_name}): {ai_response}")
        
        # Parse AI response for button inputs
        import re
        button_pattern = r'\b([UDLRABSX])(\d*)\b'
        matches = re.findall(button_pattern, ai_response.upper())
        
        if not matches:
            logger.warning(f"No valid button inputs found: {ai_response}")
            return False
        
        # Format inputs
        inputs = ' '.join(f"{btn}{count}" if count else btn for btn, count in matches)
        
        # Save screenshot before action
        self.save_screenshot()
        
        # Send inputs
        success = self.send_inputs(inputs)
        
        # Save screenshot after action
        self.save_screenshot()
        
        return success
    
    def run_auto(self, max_turns: int = 10, use_vision: bool = None) -> List[Dict]:
        """
        Run multiple turns automatically
        
        Args:
            max_turns: Maximum number of turns to run
            use_vision: Whether to use vision (defaults to model capability)
        
        Returns:
            List of turn results
        """
        results = []
        
        print(f"\n🎮 Starting auto-play with {self.model_name}")
        print(f"   Vision: {'✅ Yes' if (use_vision if use_vision is not None else self.model_info['vision']) else '❌ No'}")
        print(f"   Max turns: {max_turns}")
        print()
        
        for i in range(max_turns):
            print(f"Turn {i+1}/{max_turns}...", end=" ", flush=True)
            
            success = self.run_turn(use_vision=use_vision)
            
            state = {
                'turn': i + 1,
                'success': success,
                'frame': self.frame_count,
                'memory_count': len(self.memory)
            }
            results.append(state)
            
            print(f"{'✅' if success else '❌'} Frame: {self.frame_count}")
        
        return results


def main():
    """Test the agent with different models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenClaw Game Agent - Model-Agnostic")
    parser.add_argument("--rom", default="/home/duckets/roms/Pokemon - Red Version (USA, Europe) (SGB Enhanced).gb",
                       help="ROM file path")
    parser.add_argument("--model", default="google-gemini-cli/gemini-3-flash-preview",
                       help="Model to use (any OpenClaw model)")
    parser.add_argument("--turns", type=int, default=5,
                       help="Number of auto turns to run")
    parser.add_argument("--no-vision", action="store_true",
                       help="Disable vision (for text-only models)")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        print("\n📦 Available Models for OpenClaw Game Agent")
        print("=" * 60)
        print(f"{'Alias':<20} {'Model':<40} {'Vision':<8} {'Cost':<15}")
        print("=" * 60)
        for alias, info in MODEL_PROVIDERS.items():
            vision = "✅" if info['vision'] else "❌"
            print(f"{alias:<20} {info['model']:<40} {vision:<8} {info['cost']:<15}")
        print("=" * 60)
        print("\nUsage:")
        print("  python openclaw_agent.py --model gemini-3-flash")
        print("  python openclaw_agent.py --model qwen-3.5-plus")
        print("  python openclaw_agent.py --model lmstudio-jan --no-vision")
        return
    
    # Create config
    config = {
        'ROM_PATH': args.rom,
        'EMULATION_MODE': 'turn_based',
        'LOG_FILE': '/home/duckets/.openclaw/workspace/logs/openclaw-game-agent.log',
        'MAX_HISTORY_MESSAGES': 30,
        'MAX_SCREENSHOTS': 5,
        'CUSTOM_INSTRUCTIONS': 'Explore and progress through the game',
        'MODEL_DEFAULTS': {
            'MODEL': args.model
        }
    }
    
    agent = OpenClawGameAgent(config)
    
    print(f"\n🎮 Game Loaded: {agent.game_name}")
    print(f"🤖 Model: {agent.model_name}")
    print(f"   Alias: {agent.model_info['alias']}")
    print(f"   Vision: {'✅ Yes' if agent.model_info['vision'] else '❌ No'}")
    print(f"   Cost: {agent.model_info['cost']}")
    print()
    
    # Run auto turns
    use_vision = not args.no_vision
    results = agent.run_auto(max_turns=args.turns, use_vision=use_vision)
    
    # Summary
    print(f"\n📊 Results:")
    successful = sum(1 for r in results if r['success'])
    print(f"   Successful turns: {successful}/{len(results)}")
    print(f"   Final frame: {results[-1]['frame'] if results else 0}")
    
    # Cleanup
    agent.stop()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
