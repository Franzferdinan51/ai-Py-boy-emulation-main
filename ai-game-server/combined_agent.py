#!/usr/bin/env python3
"""
Combined Game Agent - Unified Interface
Merges MCP Server + ClaudePlayer AI Agent into single interface

Features:
- MCP tools for low-level control (load ROM, buttons, frames)
- AI decision-making for autonomous gameplay
- Works with ANY OpenClaw model (9+ supported)
- Vision + Memory + Auto-play
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from io import BytesIO
import base64

# PyBoy
try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️  PyBoy not installed")

# Vision
from PIL import Image

# MCP Server
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP library not installed")

# AI Client (OpenClaw Gateway)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("combined-agent")


# ============================================================================
# Model Configurations (9+ models supported)
# ============================================================================

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


# ============================================================================
# Combined Game Agent
# ============================================================================

class CombinedGameAgent:
    """
    Unified Game Agent combining:
    - MCP tools (low-level control)
    - AI decision-making (autonomous gameplay)
    - Multi-model support (9+ models)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rom_path = config.get('ROM_PATH', '')
        self.model_name = config.get('MODEL', 'google-gemini-cli/gemini-3-flash-preview')
        self.custom_instructions = config.get('CUSTOM_INSTRUCTIONS', '')
        
        # Model info
        self.model_info = self._get_model_info(self.model_name)
        
        # Game state
        self.emulator: Optional[PyBoy] = None
        self.frame_count = 0
        self.memory: List[str] = []
        self.current_goal: str = ""
        self.game_name: str = ""
        
        # MCP server
        self.mcp_server: Optional[Server] = None
        
        # AI client
        self.ai_client: Optional[OpenAI] = None
        
        # Setup
        self._setup_logging()
        self._init_emulator()
        self._init_ai_client()
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model capabilities"""
        for alias, info in MODEL_PROVIDERS.items():
            if alias in model_name.lower() or info['model'] == model_name:
                return {
                    'alias': alias,
                    'full_name': info['model'],
                    'vision': info['vision'],
                    'cost': info['cost']
                }
        
        return {
            'alias': 'custom',
            'full_name': model_name,
            'vision': True,
            'cost': 'unknown'
        }
    
    def _setup_logging(self):
        """Setup file logging"""
        log_file = self.config.get('LOG_FILE', '/home/duckets/.openclaw/workspace/logs/combined-agent.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
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
        logger.info(f"✅ Game loaded: {self.game_name}")
    
    def _init_ai_client(self):
        """Initialize AI client (OpenClaw Gateway)"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI client not available - AI features disabled")
            return
        
        try:
            self.ai_client = OpenAI(
                base_url="http://localhost:18789/v1",
                api_key="not-needed"
            )
            logger.info(f"✅ AI client initialized: {self.model_name}")
        except Exception as e:
            logger.warning(f"AI client init failed: {e}")
            self.ai_client = None
    
    # ========================================================================
    # MCP Tools (Low-Level Control)
    # ========================================================================
    
    def emulator_load_rom(self, rom_path: str) -> Dict:
        """Load a ROM file"""
        try:
            if self.emulator:
                self.emulator.stop()
            
            self.emulator = PyBoy(rom_path, window="null")
            self.game_name = Path(rom_path).stem
            self.frame_count = 0
            
            logger.info(f"ROM loaded: {rom_path}")
            return {"success": True, "rom": rom_path, "game": self.game_name}
        except Exception as e:
            logger.error(f"Load ROM failed: {e}")
            return {"success": False, "error": str(e)}
    
    def emulator_press_button(self, button: str) -> Dict:
        """Press a controller button"""
        if self.emulator is None:
            return {"success": False, "error": "Emulator not initialized"}
        
        button_map = {
            'A': WindowEvent.PRESS_BUTTON_A,
            'B': WindowEvent.PRESS_BUTTON_B,
            'UP': WindowEvent.PRESS_ARROW_UP,
            'DOWN': WindowEvent.PRESS_ARROW_DOWN,
            'LEFT': WindowEvent.PRESS_ARROW_LEFT,
            'RIGHT': WindowEvent.PRESS_ARROW_RIGHT,
            'START': WindowEvent.PRESS_BUTTON_START,
            'SELECT': WindowEvent.PRESS_BUTTON_SELECT,
        }
        
        release_map = {
            'A': WindowEvent.RELEASE_BUTTON_A,
            'B': WindowEvent.RELEASE_BUTTON_B,
            'UP': WindowEvent.RELEASE_ARROW_UP,
            'DOWN': WindowEvent.RELEASE_ARROW_DOWN,
            'LEFT': WindowEvent.RELEASE_ARROW_LEFT,
            'RIGHT': WindowEvent.RELEASE_ARROW_RIGHT,
            'START': WindowEvent.RELEASE_BUTTON_START,
            'SELECT': WindowEvent.RELEASE_BUTTON_SELECT,
        }
        
        try:
            button_upper = button.upper()
            if button_upper not in button_map:
                return {"success": False, "error": f"Unknown button: {button}"}
            
            self.emulator.send_input(button_map[button_upper])
            self.emulator.tick()
            self.emulator.send_input(release_map[button_upper])
            self.frame_count += 1
            
            logger.info(f"Button pressed: {button}")
            return {"success": True, "button": button, "frame": self.frame_count}
        except Exception as e:
            logger.error(f"Button press failed: {e}")
            return {"success": False, "error": str(e)}
    
    def emulator_get_frame(self) -> Dict:
        """Get current frame as base64"""
        if self.emulator is None:
            return {"success": False, "error": "Emulator not initialized"}
        
        try:
            screen = self.emulator.screen
            if screen is None:
                return {"success": False, "error": "No screen buffer"}
            
            img = screen.image
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                "success": True,
                "base64": img_base64,
                "width": img.width,
                "height": img.height,
                "frame": self.frame_count
            }
        except Exception as e:
            logger.error(f"Get frame failed: {e}")
            return {"success": False, "error": str(e)}
    
    def emulator_get_state(self) -> Dict:
        """Get emulator state"""
        return {
            "success": True,
            "initialized": self.emulator is not None,
            "game": self.game_name,
            "frame": self.frame_count,
            "rom_path": self.rom_path
        }
    
    def emulator_tick(self, frames: int = 1) -> Dict:
        """Advance emulation"""
        if self.emulator is None:
            return {"success": False, "error": "Emulator not initialized"}
        
        try:
            for _ in range(frames):
                self.emulator.tick()
                self.frame_count += 1
            
            return {"success": True, "frames": frames, "new_frame": self.frame_count}
        except Exception as e:
            logger.error(f"Tick failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # AI Agent Features (High-Level Control)
    # ========================================================================
    
    def get_ai_decision(self, use_vision: bool = True) -> str:
        """Get AI decision using configured model"""
        if self.ai_client is None:
            logger.warning("AI client not available - using default")
            return "W"
        
        try:
            # Build prompt
            system_prompt = self._generate_prompt()
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add vision if supported
            if use_vision and self.model_info['vision']:
                frame = self.emulator_get_frame()
                if frame.get('success'):
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
                    "content": "What buttons should I press next? Respond with button notation (e.g., 'A B START')."
                })
            
            # Get AI response
            response = self.ai_client.chat.completions.create(
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
            return "W"
    
    def _generate_prompt(self) -> str:
        """Generate AI prompt"""
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
    
    def add_to_memory(self, item: str) -> Dict:
        """Add item to memory"""
        self.memory.append(item)
        logger.info(f"Memory added: {item}")
        
        # Trim if too long
        max_memory = self.config.get('MAX_MEMORY', 30)
        if len(self.memory) > max_memory:
            self.memory.pop(0)
        
        return {"success": True, "memory_count": len(self.memory)}
    
    def set_current_goal(self, goal: str) -> Dict:
        """Set current gameplay objective"""
        self.current_goal = goal
        logger.info(f"Goal set: {goal}")
        return {"success": True, "goal": goal}
    
    def save_screenshot(self, output_dir: str = None) -> Dict:
        """Save current frame to file"""
        if self.emulator is None:
            return {"success": False, "error": "Emulator not initialized"}
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "frames" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame = self.emulator_get_frame()
        if not frame.get('success'):
            return frame
        
        # Decode and save
        img_data = base64.b64decode(frame['base64'])
        output_path = output_dir / f"frame_{self.frame_count:06d}.png"
        
        with open(output_path, 'wb') as f:
            f.write(img_data)
        
        logger.info(f"Screenshot saved: {output_path}")
        return {"success": True, "path": str(output_path)}
    
    def run_auto(self, max_turns: int = 10, use_vision: bool = None) -> List[Dict]:
        """Run multiple turns automatically"""
        if use_vision is None:
            use_vision = self.model_info['vision']
        
        results = []
        logger.info(f"Starting auto-play: {max_turns} turns, vision={use_vision}")
        
        for i in range(max_turns):
            # Get AI decision
            decision = self.get_ai_decision(use_vision=use_vision)
            
            # Parse and execute
            success = self._parse_and_execute(decision)
            
            state = {
                'turn': i + 1,
                'decision': decision,
                'success': success,
                'frame': self.frame_count
            }
            results.append(state)
            
            logger.info(f"Turn {i+1}: {decision} -> {'✅' if success else '❌'}")
        
        return results
    
    def _parse_and_execute(self, ai_response: str) -> bool:
        """Parse AI response and execute button inputs"""
        import re
        button_pattern = r'\b([UDLRABSX])(\d*)\b'
        matches = re.findall(button_pattern, ai_response.upper())
        
        if not matches:
            logger.warning(f"No valid inputs: {ai_response}")
            return False
        
        # Execute each button
        for button, count in matches:
            count = int(count) if count else 1
            for _ in range(count):
                result = self.emulator_press_button(button)
                if not result.get('success'):
                    return False
        
        return True
    
    def stop(self):
        """Stop emulator"""
        if self.emulator:
            self.emulator.stop()
            logger.info("Emulator stopped")


# ============================================================================
# MCP Server Integration
# ============================================================================

def create_mcp_server(agent: CombinedGameAgent) -> Server:
    """Create MCP server with combined agent tools"""
    server = Server("combined-game-agent")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="agent_load_rom",
                description="Load a Game Boy ROM file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "rom_path": {"type": "string", "description": "Path to ROM file"}
                    },
                    "required": ["rom_path"]
                }
            ),
            Tool(
                name="agent_press_button",
                description="Press a controller button (A, B, UP, DOWN, LEFT, RIGHT, START, SELECT)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "button": {"type": "string", "enum": ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]}
                    },
                    "required": ["button"]
                }
            ),
            Tool(
                name="agent_get_frame",
                description="Get current screen as base64 PNG",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="agent_get_state",
                description="Get emulator state",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="agent_get_ai_decision",
                description="Get AI decision for next move (uses vision if available)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "use_vision": {"type": "boolean", "default": True}
                    }
                }
            ),
            Tool(
                name="agent_add_memory",
                description="Add item to game memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Memory item to store"}
                    },
                    "required": ["item"]
                }
            ),
            Tool(
                name="agent_set_goal",
                description="Set current gameplay objective",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "Gameplay objective"}
                    },
                    "required": ["goal"]
                }
            ),
            Tool(
                name="agent_save_screenshot",
                description="Save current frame to file",
                inputSchema={"type": "object", "properties": {}}
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "agent_load_rom":
                result = agent.emulator_load_rom(arguments.get("rom_path"))
            elif name == "agent_press_button":
                result = agent.emulator_press_button(arguments.get("button"))
            elif name == "agent_get_frame":
                result = agent.emulator_get_frame()
            elif name == "agent_get_state":
                result = agent.emulator_get_state()
            elif name == "agent_get_ai_decision":
                decision = agent.get_ai_decision(arguments.get("use_vision", True))
                result = {"success": True, "decision": decision}
            elif name == "agent_add_memory":
                result = agent.add_to_memory(arguments.get("item"))
            elif name == "agent_set_goal":
                result = agent.set_current_goal(arguments.get("goal"))
            elif name == "agent_save_screenshot":
                result = agent.save_screenshot()
            else:
                return [TextContent(type="text", text=f"Error: Unknown tool: {name}")]
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return server


# ============================================================================
# Main Entry Points
# ============================================================================

def run_mcp_mode(config: Dict[str, Any]):
    """Run as MCP server"""
    agent = CombinedGameAgent(config)
    server = create_mcp_server(agent)
    
    logger.info("Starting MCP server mode...")
    
    import asyncio
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    
    asyncio.run(run())


def run_standalone(config: Dict[str, Any]):
    """Run standalone with auto-play"""
    agent = CombinedGameAgent(config)
    
    print(f"\n🎮 Game Loaded: {agent.game_name}")
    print(f"🤖 Model: {agent.model_name}")
    print(f"   Vision: {'✅ Yes' if agent.model_info['vision'] else '❌ No'}")
    print(f"   Cost: {agent.model_info['cost']}")
    print()
    
    # Run auto-play
    max_turns = config.get('MAX_TURNS', 10)
    use_vision = config.get('USE_VISION', agent.model_info['vision'])
    
    print(f"🎯 Starting auto-play: {max_turns} turns")
    results = agent.run_auto(max_turns=max_turns, use_vision=use_vision)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 Results: {successful}/{max_turns} successful turns")
    
    agent.stop()
    print("\n✅ Complete!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Game Agent - MCP + AI")
    parser.add_argument("--rom", required=True, help="ROM file path")
    parser.add_argument("--model", default="google-gemini-cli/gemini-3-flash-preview", help="Model to use")
    parser.add_argument("--goal", default="", help="Gameplay objective")
    parser.add_argument("--mcp", action="store_true", help="Run as MCP server")
    parser.add_argument("--turns", type=int, default=10, help="Auto-play turns")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\n📦 Available Models")
        print("=" * 70)
        print(f"{'Alias':<20} {'Model':<40} {'Vision':<8} {'Cost':<15}")
        print("=" * 70)
        for alias, info in MODEL_PROVIDERS.items():
            vision = "✅" if info['vision'] else "❌"
            print(f"{alias:<20} {info['model']:<40} {vision:<8} {info['cost']:<15}")
        print("=" * 70)
        return
    
    # Create config
    config = {
        'ROM_PATH': args.rom,
        'MODEL': args.model,
        'CUSTOM_INSTRUCTIONS': args.goal,
        'MAX_TURNS': args.turns,
        'USE_VISION': not args.no_vision,
        'LOG_FILE': '/home/duckets/.openclaw/workspace/logs/combined-agent.log'
    }
    
    # Run
    if args.mcp:
        run_mcp_mode(config)
    else:
        run_standalone(config)


if __name__ == "__main__":
    main()
