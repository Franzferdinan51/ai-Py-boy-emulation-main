#!/usr/bin/env python3
"""
Vision Analysis Module for Pokemon Red GameBoy Emulation
Uses Kimi-K2.5 (Bailian) for AI-powered screen understanding
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import io

# Configuration
KIMI_API_URL = "https://api.bailian.com/v1/chat/completions"
KIMI_MODEL = "kimi-k2.5"

class GameState(Enum):
    """Possible game states detectable from screen"""
    OVERWORLD = "overworld"
    BATTLE = "battle"
    MENU = "menu"
    DIALOG = "dialog"
    INVENTORY = "inventory"
    POKEMON_PARTY = "pokemon_party"
    POKEDEX = "pokedex"
    START_MENU = "start_menu"
    SHOP = "shop"
    PC = "pc"
    TRADE = "trade"
    TITLE_SCREEN = "title_screen"
    UNKNOWN = "unknown"

class BattleState(Enum):
    """Battle-specific states"""
    NOT_IN_BATTLE = "not_in_battle"
    PLAYER_TURN = "player_turn"
    ENEMY_TURN = "enemy_turn"
    MOVE_SELECTION = "move_selection"
    ITEM_SELECTION = "item_selection"
    POKEMON_SELECTION = "pokemon_selection"
    RUN_ATTEMPT = "run_attempt"
    BATTLE_WON = "battle_won"
    BATTLE_LOST = "battle_lost"
    WILD_ENCOUNTER = "wild_encounter"
    TRAINER_BATTLE = "trainer_battle"

@dataclass
class DetectedSprite:
    """Represents a detected sprite on screen"""
    sprite_type: str  # "pokemon", "npc", "item", "player"
    name: str
    position: Tuple[int, int]  # (x, y) center coordinates
    bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    description: str = ""

@dataclass
class DetectedText:
    """Represents OCR'd text from screen"""
    text: str
    position: Tuple[int, int]
    text_type: str  # "dialog", "menu", "battle_text", "ui", "unknown"
    confidence: float

@dataclass
class VisionAnalysisResult:
    """Complete vision analysis result"""
    game_state: GameState
    battle_state: BattleState
    sprites: List[DetectedSprite]
    texts: List[DetectedText]
    current_location: str
    player_position: Optional[Tuple[int, int]]
    menu_open: bool
    in_battle: bool
    battle_info: Optional[Dict[str, Any]]
    recommendations: List[str]
    raw_analysis: str

class PokemonVisionAnalyzer:
    """
    AI-powered vision analyzer for Pokemon Red gameplay
    Uses Kimi-K2.5 for understanding game screens
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BAILIAN_API_KEY")
        if not self.api_key:
            raise ValueError("Bailian API key required. Set BAILIAN_API_KEY env var.")
        
        self.api_url = KIMI_API_URL
        self.model = KIMI_MODEL
        self.last_analysis: Optional[VisionAnalysisResult] = None
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _call_kimi_vision(self, image_base64: str, prompt: str) -> str:
        """Call Kimi-K2.5 vision API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def analyze_screen(self, image_path: str) -> VisionAnalysisResult:
        """
        Perform comprehensive vision analysis of game screen
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            VisionAnalysisResult with complete analysis
        """
        image_base64 = self._encode_image(image_path)
        
        # Comprehensive analysis prompt
        prompt = """Analyze this Pokemon Red GameBoy screen image. Provide a detailed JSON response with the following structure:

{
    "game_state": "overworld|battle|menu|dialog|inventory|pokemon_party|pokedex|start_menu|shop|pc|trade|title_screen|unknown",
    "battle_state": "not_in_battle|player_turn|enemy_turn|move_selection|item_selection|pokemon_selection|run_attempt|battle_won|battle_lost|wild_encounter|trainer_battle",
    "current_location": "name of location (Pallet Town, Route 1, etc.)",
    "player_position": {"x": int, "y": int} or null,
    "menu_open": true/false,
    "in_battle": true/false,
    "sprites": [
        {
            "type": "pokemon|npc|item|player",
            "name": "sprite name or description",
            "position": {"x": int, "y": int},
            "bounds": {"x1": int, "y1": int, "x2": int, "y2": int},
            "confidence": 0.0-1.0,
            "description": "additional details"
        }
    ],
    "texts": [
        {
            "text": "detected text",
            "position": {"x": int, "y": int},
            "type": "dialog|menu|battle_text|ui|unknown",
            "confidence": 0.0-1.0
        }
    ],
    "battle_info": {
        "player_pokemon": "name or null",
        "player_hp_percent": int or null,
        "player_level": int or null,
        "enemy_pokemon": "name or null",
        "enemy_hp_percent": int or null,
        "enemy_level": int or null,
        "enemy_trainer": "trainer name or null (for trainer battles)"
    } or null,
    "recommendations": ["list of recommended actions based on current state"]
}

Be thorough and accurate. For sprites, estimate pixel coordinates based on the 160x144 GameBoy screen resolution. For text, transcribe exactly what's visible."""

        response = self._call_kimi_vision(image_base64, prompt)
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            # Fallback: try to parse the whole response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                data = self._fallback_parse(response)
        
        # Build result object
        result = self._build_result(data, response)
        self.last_analysis = result
        return result
    
    def _fallback_parse(self, response: str) -> Dict:
        """Fallback parsing when JSON extraction fails"""
        return {
            "game_state": "unknown",
            "battle_state": "not_in_battle",
            "current_location": "unknown",
            "player_position": None,
            "menu_open": False,
            "in_battle": False,
            "sprites": [],
            "texts": [],
            "battle_info": None,
            "recommendations": ["Could not parse screen - manual review needed"],
            "parse_error": True,
            "raw_response_preview": response[:500]
        }
    
    def _build_result(self, data: Dict, raw_response: str) -> VisionAnalysisResult:
        """Build VisionAnalysisResult from parsed data"""
        
        # Parse game state
        game_state_str = data.get("game_state", "unknown").lower()
        try:
            game_state = GameState(game_state_str)
        except ValueError:
            game_state = GameState.UNKNOWN
        
        # Parse battle state
        battle_state_str = data.get("battle_state", "not_in_battle").lower()
        try:
            battle_state = BattleState(battle_state_str)
        except ValueError:
            battle_state = BattleState.NOT_IN_BATTLE
        
        # Parse sprites
        sprites = []
        for sprite_data in data.get("sprites", []):
            try:
                pos = sprite_data.get("position", {})
                bounds = sprite_data.get("bounds", {})
                sprites.append(DetectedSprite(
                    sprite_type=sprite_data.get("type", "unknown"),
                    name=sprite_data.get("name", "unknown"),
                    position=(pos.get("x", 0), pos.get("y", 0)),
                    bounds=(
                        bounds.get("x1", 0), bounds.get("y1", 0),
                        bounds.get("x2", 0), bounds.get("y2", 0)
                    ),
                    confidence=sprite_data.get("confidence", 0.5),
                    description=sprite_data.get("description", "")
                ))
            except Exception:
                continue
        
        # Parse texts
        texts = []
        for text_data in data.get("texts", []):
            try:
                pos = text_data.get("position", {})
                texts.append(DetectedText(
                    text=text_data.get("text", ""),
                    position=(pos.get("x", 0), pos.get("y", 0)),
                    text_type=text_data.get("type", "unknown"),
                    confidence=text_data.get("confidence", 0.5)
                ))
            except Exception:
                continue
        
        # Parse player position
        player_pos = data.get("player_position")
        if player_pos and isinstance(player_pos, dict):
            player_position = (player_pos.get("x"), player_pos.get("y"))
        else:
            player_position = None
        
        return VisionAnalysisResult(
            game_state=game_state,
            battle_state=battle_state,
            sprites=sprites,
            texts=texts,
            current_location=data.get("current_location", "unknown"),
            player_position=player_position,
            menu_open=data.get("menu_open", False),
            in_battle=data.get("in_battle", False),
            battle_info=data.get("battle_info"),
            recommendations=data.get("recommendations", []),
            raw_analysis=raw_response
        )
    
    def detect_sprites(self, image_path: str) -> List[DetectedSprite]:
        """
        Quick sprite detection - focused analysis
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            List of detected sprites
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Focus ONLY on detecting sprites in this Pokemon Red screen.
Identify all visible sprites: Pokemon, NPCs, items, and the player character.

Return JSON:
{
    "sprites": [
        {
            "type": "pokemon|npc|item|player",
            "name": "specific name or description",
            "position": {"x": int, "y": int},
            "bounds": {"x1": int, "y1": int, "x2": int, "y2": int},
            "confidence": 0.0-1.0
        }
    ]
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
        except:
            return []
        
        sprites = []
        for sprite_data in data.get("sprites", []):
            try:
                pos = sprite_data.get("position", {})
                bounds = sprite_data.get("bounds", {})
                sprites.append(DetectedSprite(
                    sprite_type=sprite_data.get("type", "unknown"),
                    name=sprite_data.get("name", "unknown"),
                    position=(pos.get("x", 0), pos.get("y", 0)),
                    bounds=(
                        bounds.get("x1", 0), bounds.get("y1", 0),
                        bounds.get("x2", 0), bounds.get("y2", 0)
                    ),
                    confidence=sprite_data.get("confidence", 0.5),
                    description=sprite_data.get("description", "")
                ))
            except Exception:
                continue
        
        return sprites
    
    def ocr_text(self, image_path: str) -> List[DetectedText]:
        """
        Extract all text from screen using OCR
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            List of detected text elements
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Extract ALL text visible in this Pokemon Red screen image.
Include dialog boxes, menus, battle text, UI elements, and any other text.

Return JSON:
{
    "texts": [
        {
            "text": "the exact text",
            "position": {"x": int, "y": int},
            "type": "dialog|menu|battle_text|ui|unknown",
            "confidence": 0.0-1.0
        }
    ]
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
        except:
            return []
        
        texts = []
        for text_data in data.get("texts", []):
            try:
                pos = text_data.get("position", {})
                texts.append(DetectedText(
                    text=text_data.get("text", ""),
                    position=(pos.get("x", 0), pos.get("y", 0)),
                    text_type=text_data.get("type", "unknown"),
                    confidence=text_data.get("confidence", 0.5)
                ))
            except Exception:
                continue
        
        return texts
    
    def detect_menu_state(self, image_path: str) -> Dict[str, Any]:
        """
        Detect if a menu is open and what type
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            Dict with menu state information
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Analyze this Pokemon Red screen to detect menu state.
Is a menu open? What type? What options are visible?

Return JSON:
{
    "menu_open": true/false,
    "menu_type": "start_menu|battle_menu|inventory|pokemon_party|shop|pc|dialog|none",
    "menu_options": ["list", "of", "visible", "options"],
    "selected_option": "currently selected option or null",
    "cursor_position": {"x": int, "y": int} or null
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except:
            return {
                "menu_open": False,
                "menu_type": "none",
                "menu_options": [],
                "selected_option": None,
                "cursor_position": None
            }
    
    def detect_battle_state(self, image_path: str) -> Dict[str, Any]:
        """
        Detect battle state and information
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            Dict with battle information
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Analyze this Pokemon Red screen for battle information.
Is there an active battle? What Pokemon are involved? What are their stats?

Return JSON:
{
    "in_battle": true/false,
    "battle_type": "wild|trainer|none",
    "player_pokemon": {
        "name": "Pokemon name",
        "level": int,
        "hp_current": int or null,
        "hp_max": int or null,
        "hp_percent": int,
        "status": "normal|poison|paralyze|sleep|burn|freeze|null"
    } or null,
    "enemy_pokemon": {
        "name": "Pokemon name",
        "level": int,
        "hp_percent": int,
        "status": "normal|poison|paralyze|sleep|burn|freeze|null"
    } or null,
    "enemy_trainer": "trainer name or null",
    "battle_phase": "intro|player_turn|enemy_turn|move_selection|item_selection|pokemon_selection|victory|defeat",
    "available_moves": ["move1", "move2", ...] or null,
    "battle_message": "current battle text or null"
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except:
            return {
                "in_battle": False,
                "battle_type": "none",
                "player_pokemon": None,
                "enemy_pokemon": None,
                "enemy_trainer": None,
                "battle_phase": "none",
                "available_moves": None,
                "battle_message": None
            }
    
    def detect_map_transition(self, image_path: str, previous_analysis: Optional[VisionAnalysisResult] = None) -> Dict[str, Any]:
        """
        Detect if player is transitioning between maps
        
        Args:
            image_path: Path to current screenshot
            previous_analysis: Previous analysis result for comparison
            
        Returns:
            Dict with transition information
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Analyze this Pokemon Red screen for map/location information.
What is the current location? Is there a transition happening (black screen, warp, etc.)?

Return JSON:
{
    "current_location": "location name (Pallet Town, Route 1, etc.)",
    "location_type": "town|route|cave|building|gym|center|mart|house|indoor|outdoor",
    "transition_detected": true/false,
    "transition_type": "door|warp|edge|stairs|cave|none",
    "screen_type": "normal|black|white|flash|none",
    "visible_exits": ["north", "south", "east", "west"]
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            # Compare with previous if provided
            if previous_analysis:
                result["location_changed"] = result.get("current_location") != previous_analysis.current_location
            else:
                result["location_changed"] = False
            
            return result
        except:
            return {
                "current_location": "unknown",
                "location_type": "unknown",
                "transition_detected": False,
                "transition_type": "none",
                "screen_type": "normal",
                "visible_exits": [],
                "location_changed": False
            }
    
    def get_recommendations(self, image_path: str) -> List[str]:
        """
        Get AI recommendations for current game state
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            List of recommended actions
        """
        image_base64 = self._encode_image(image_path)
        
        prompt = """Analyze this Pokemon Red screen and provide strategic recommendations.
What should the player do next? Consider:
- Current game state (overworld, battle, menu)
- Pokemon health and status
- Available items
- Progress toward goals
- Risk assessment

Return JSON:
{
    "recommendations": [
        "First recommended action with reasoning",
        "Second option",
        "Third option"
    ],
    "priority": "high|medium|low",
    "reasoning": "brief explanation of recommendation"
}"""
        
        response = self._call_kimi_vision(image_base64, prompt)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            return data.get("recommendations", [])
        except:
            return ["Could not generate recommendations"]
    
    def export_to_json(self, result: VisionAnalysisResult, output_path: str):
        """Export analysis result to JSON file"""
        data = {
            "game_state": result.game_state.value,
            "battle_state": result.battle_state.value,
            "current_location": result.current_location,
            "player_position": result.player_position,
            "menu_open": result.menu_open,
            "in_battle": result.in_battle,
            "sprites": [
                {
                    "type": s.sprite_type,
                    "name": s.name,
                    "position": s.position,
                    "bounds": s.bounds,
                    "confidence": s.confidence,
                    "description": s.description
                }
                for s in result.sprites
            ],
            "texts": [
                {
                    "text": t.text,
                    "position": t.position,
                    "type": t.text_type,
                    "confidence": t.confidence
                }
                for t in result.texts
            ],
            "battle_info": result.battle_info,
            "recommendations": result.recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_analysis(self, result: VisionAnalysisResult):
        """Print formatted analysis to console"""
        print("\n" + "="*60)
        print("POKEMON RED VISION ANALYSIS")
        print("="*60)
        print(f"\n📍 Location: {result.current_location}")
        print(f"🎮 Game State: {result.game_state.value}")
        print(f"⚔️  Battle State: {result.battle_state.value}")
        print(f"📋 Menu Open: {result.menu_open}")
        print(f"🥊 In Battle: {result.in_battle}")
        
        if result.player_position:
            print(f"🚶 Player Position: {result.player_position}")
        
        if result.sprites:
            print(f"\n🎨 Detected Sprites ({len(result.sprites)}):")
            for sprite in result.sprites:
                print(f"  • {sprite.name} ({sprite.sprite_type}) at {sprite.position}")
        
        if result.texts:
            print(f"\n📝 Detected Text ({len(result.texts)}):")
            for text in result.texts[:5]:  # Limit to first 5
                print(f"  • [{text.text_type}] \"{text.text[:50]}...\" at {text.position}")
        
        if result.battle_info:
            print(f"\n⚔️  Battle Info:")
            print(f"  Player: {result.battle_info.get('player_pokemon')}")
            print(f"  Enemy: {result.battle_info.get('enemy_pokemon')}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("="*60 + "\n")


def main():
    """CLI interface for vision analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pokemon Red Vision Analysis Tool")
    parser.add_argument("image", help="Path to screenshot image")
    parser.add_argument("--api-key", help="Bailian API key (or set BAILIAN_API_KEY env var)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--mode", "-m", choices=["full", "sprites", "text", "menu", "battle", "map"],
                       default="full", help="Analysis mode")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = PokemonVisionAnalyzer(api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Run analysis based on mode
    if args.mode == "full":
        result = analyzer.analyze_screen(args.image)
        analyzer.print_analysis(result)
        
        if args.output:
            analyzer.export_to_json(result, args.output)
            print(f"Analysis exported to: {args.output}")
    
    elif args.mode == "sprites":
        sprites = analyzer.detect_sprites(args.image)
        print(f"\nDetected {len(sprites)} sprites:")
        for sprite in sprites:
            print(f"  • {sprite.name} ({sprite.sprite_type}) at {sprite.position}")
    
    elif args.mode == "text":
        texts = analyzer.ocr_text(args.image)
        print(f"\nDetected {len(texts)} text elements:")
        for text in texts:
            print(f"  • [{text.text_type}] \"{text.text}\" at {text.position}")
    
    elif args.mode == "menu":
        menu_state = analyzer.detect_menu_state(args.image)
        print(f"\nMenu State:")
        print(json.dumps(menu_state, indent=2))
    
    elif args.mode == "battle":
        battle_state = analyzer.detect_battle_state(args.image)
        print(f"\nBattle State:")
        print(json.dumps(battle_state, indent=2))
    
    elif args.mode == "map":
        map_info = analyzer.detect_map_transition(args.image)
        print(f"\nMap Information:")
        print(json.dumps(map_info, indent=2))


if __name__ == "__main__":
    main()