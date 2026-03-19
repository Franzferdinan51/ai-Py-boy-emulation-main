"""
Dual Model Provider - Orchestrates vision and planning models independently

This provider implements the dual-model architecture:
1. Vision Model: Analyzes screenshots and extracts game state/context
2. Planning Model: Makes decisions based on vision model's analysis

Flow: Screenshot -> Vision Model -> Planning Model -> Action
"""
import logging
import base64
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VisionModelType(Enum):
    """Available vision models"""
    KIMI_K25 = "kimi-k2.5"
    QWEN_VL = "qwen-vl-plus"
    GLM_4V = "glm-4v-flash"
    MINIMAX_M27 = "MiniMax-M2.7"  # Also has vision capability


class PlanningModelType(Enum):
    """Available planning/thinking models"""
    GLM_5 = "glm-5"
    QWEN_35_PLUS = "qwen3.5-plus"
    MINIMAX_M27 = "MiniMax-M2.7"
    MINIMAX_M25 = "MiniMax-M2.5"


@dataclass
class VisionAnalysis:
    """Result from vision model analysis"""
    game_state: str  # Current game state description
    player_position: Optional[str] = None
    nearby_entities: List[str] = None
    ui_elements: List[str] = None
    relevant_objects: List[str] = None
    danger_level: str = "low"  # low, medium, high
    opportunities: List[str] = None
    raw_description: str = ""
    
    def __post_init__(self):
        if self.nearby_entities is None:
            self.nearby_entities = []
        if self.ui_elements is None:
            self.ui_elements = []
        if self.relevant_objects is None:
            self.relevant_objects = []
        if self.opportunities is None:
            self.opportunities = []


@dataclass
class PlanningResult:
    """Result from planning model"""
    action: str
    reasoning: str
    confidence: float = 0.8
    alternative_actions: List[str] = None
    expected_outcome: str = ""
    
    def __post_init__(self):
        if self.alternative_actions is None:
            self.alternative_actions = []


class DualModelProvider:
    """
    Orchestrates independent vision and planning models.
    
    Architecture:
    - Vision Model: Specialized in image understanding, screen analysis
    - Planning Model: Specialized in reasoning, decision making
    
    The vision model provides context, the planning model makes decisions.
    """
    
    # Default model configurations
    DEFAULT_VISION_MODEL = VisionModelType.KIMI_K25.value
    DEFAULT_PLANNING_MODEL = PlanningModelType.GLM_5.value
    
    # Model capabilities mapping
    VISION_CAPABLE_MODELS = {
        "kimi-k2.5": "bailian/kimi-k2.5",
        "qwen-vl-plus": "bailian/qwen-vl-plus",
        "glm-4v-flash": "bailian/glm-4v-flash",
        "MiniMax-M2.7": "bailian/MiniMax-M2.7",
    }
    
    PLANNING_MODELS = {
        "glm-5": "bailian/glm-5",
        "qwen3.5-plus": "bailian/qwen3.5-plus",
        "MiniMax-M2.7": "bailian/MiniMax-M2.7",
        "MiniMax-M2.5": "bailian/MiniMax-M2.5",
    }
    
    def __init__(
        self,
        openclaw_endpoint: str = "http://localhost:18789",
        vision_model: str = None,
        planning_model: str = None,
    ):
        """
        Initialize dual-model provider.
        
        Args:
            openclaw_endpoint: OpenClaw Gateway endpoint
            vision_model: Vision model identifier (default: kimi-k2.5)
            planning_model: Planning model identifier (default: glm-5)
        """
        self.openclaw_endpoint = openclaw_endpoint.rstrip('/')
        self.vision_model = vision_model or self.DEFAULT_VISION_MODEL
        self.planning_model = planning_model or self.DEFAULT_PLANNING_MODEL
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track model usage
        self.last_vision_response = None
        self.last_planning_response = None
        
        self.logger.info(
            f"DualModelProvider initialized: "
            f"vision={self.vision_model}, planning={self.planning_model}"
        )
    
    def set_vision_model(self, model: str) -> bool:
        """Update the vision model"""
        if model in self.VISION_CAPABLE_MODELS or model in VisionModelType.values():
            self.vision_model = model
            self.logger.info(f"Vision model updated to: {model}")
            return True
        self.logger.warning(f"Invalid vision model: {model}")
        return False
    
    def set_planning_model(self, model: str) -> bool:
        """Update the planning model"""
        if model in self.PLANNING_MODELS or model in PlanningModelType.values():
            self.planning_model = model
            self.logger.info(f"Planning model updated to: {model}")
            return True
        self.logger.warning(f"Invalid planning model: {model}")
        return False
    
    def analyze_screen(self, image_bytes: bytes, context: Dict[str, Any] = None) -> VisionAnalysis:
        """
        Use vision model to analyze the game screen.
        
        Args:
            image_bytes: Raw screenshot bytes
            context: Additional context (goal, game state, etc.)
            
        Returns:
            VisionAnalysis with extracted game state
        """
        import requests
        
        context = context or {}
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Build vision prompt
        prompt = self._build_vision_prompt(context)
        
        try:
            # Call OpenClaw vision endpoint
            payload = {
                "prompt": prompt,
                "image": image_base64,
                "model": self._get_model_id(self.vision_model, is_vision=True),
            }
            
            response = requests.post(
                f"{self.openclaw_endpoint}/api/vision/analyze",
                json=payload,
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                vision_text = data.get('response') or data.get('text') or data.get('result', '')
                
                # Parse vision response into structured analysis
                analysis = self._parse_vision_response(vision_text, context)
                self.last_vision_response = vision_text
                self.logger.debug(f"Vision analysis: {analysis.game_state[:100]}...")
                return analysis
            else:
                self.logger.warning(f"Vision API returned {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Vision analysis failed: {e}")
        
        # Fallback: minimal analysis
        return VisionAnalysis(
            game_state="Unable to analyze screen",
            raw_description="Vision model unavailable",
            danger_level="unknown"
        )
    
    def plan_action(
        self,
        vision_analysis: VisionAnalysis,
        goal: str,
        action_history: List[str],
        context: Dict[str, Any] = None
    ) -> PlanningResult:
        """
        Use planning model to decide next action based on vision analysis.
        
        Args:
            vision_analysis: Result from vision model
            goal: Current objective
            action_history: Recent actions taken
            context: Additional context
            
        Returns:
            PlanningResult with action decision
        """
        import requests
        
        context = context or {}
        
        # Build planning prompt with vision context
        prompt = self._build_planning_prompt(vision_analysis, goal, action_history, context)
        
        try:
            # Call OpenClaw chat endpoint
            payload = {
                "prompt": prompt,
                "model": self._get_model_id(self.planning_model, is_vision=False),
            }
            
            response = requests.post(
                f"{self.openclaw_endpoint}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                planning_text = data.get('response') or data.get('text') or data.get('result', '')
                
                # Parse planning response
                result = self._parse_planning_response(planning_text)
                self.last_planning_response = planning_text
                self.logger.info(f"Planning decision: {result.action} (confidence: {result.confidence})")
                return result
            else:
                self.logger.warning(f"Planning API returned {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
        
        # Fallback action
        return PlanningResult(
            action=self._get_fallback_action(action_history),
            reasoning="Planning model unavailable, using fallback",
            confidence=0.3
        )
    
    def get_next_action(
        self,
        image_bytes: bytes,
        goal: str,
        action_history: List[str],
        context: Dict[str, Any] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Complete dual-model flow: Vision -> Planning -> Action
        
        This is the main entry point for the dual-model architecture.
        
        Args:
            image_bytes: Raw screenshot bytes
            goal: Current objective
            action_history: Recent actions
            context: Additional context
            
        Returns:
            Tuple of (action, model_used)
        """
        self.logger.info(f"Dual-model flow starting for goal: {goal[:50]}...")
        
        # Step 1: Vision analysis
        vision_analysis = self.analyze_screen(image_bytes, context)
        
        # Step 2: Planning based on vision
        planning_result = self.plan_action(vision_analysis, goal, action_history, context)
        
        # Return action with model info
        models_used = f"vision:{self.vision_model}+planning:{self.planning_model}"
        self.logger.info(f"Dual-model decision: {planning_result.action} via {models_used}")
        
        return planning_result.action, models_used
    
    def chat_with_ai(
        self,
        message: str,
        image_bytes: bytes,
        context: Dict[str, Any]
    ) -> str:
        """
        Chat with AI using dual-model architecture.
        
        If image is provided, vision model analyzes it first,
        then planning model responds with context.
        """
        vision_context = ""
        
        if image_bytes:
            # Use vision model for image analysis
            vision_analysis = self.analyze_screen(image_bytes, context)
            vision_context = f"\n\n[Vision Analysis]\n{vision_analysis.raw_description}"
        
        # Build chat prompt with vision context
        full_prompt = f"{message}{vision_context}"
        
        # Use planning model for chat response
        import requests
        try:
            payload = {
                "prompt": full_prompt,
                "model": self._get_model_id(self.planning_model, is_vision=False),
            }
            
            response = requests.post(
                f"{self.openclaw_endpoint}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.ok:
                data = response.json()
                return data.get('response') or data.get('text') or data.get('result', '')
                
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
        
        return "I'm having trouble connecting to the AI services right now."
    
    def _build_vision_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for vision model"""
        goal = context.get('goal', 'explore and progress')
        game_type = context.get('game_type', 'Game Boy')
        
        return f"""Analyze this {game_type} game screen. Focus on:

1. **Game State**: What's happening? (exploring, battle, menu, dialog, etc.)
2. **Player Position**: Where is the player character?
3. **Nearby Entities**: NPCs, enemies, items, obstacles
4. **UI Elements**: Menus, text boxes, health bars, indicators
5. **Danger Level**: Is there immediate danger? (low/medium/high)
6. **Opportunities**: Things the player could interact with or do

Current objective: {goal}

Provide a concise, structured analysis. Be specific about positions and directions."""

    def _build_planning_prompt(
        self,
        vision_analysis: VisionAnalysis,
        goal: str,
        action_history: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for planning model with vision context"""
        history_str = ", ".join(action_history[-10:]) if action_history else "none"
        
        return f"""You are playing a Game Boy game. Decide the next button press.

## Vision Analysis
Game State: {vision_analysis.game_state}
Player Position: {vision_analysis.player_position or 'unknown'}
Nearby: {', '.join(vision_analysis.nearby_entities) or 'nothing notable'}
UI Elements: {', '.join(vision_analysis.ui_elements) or 'none visible'}
Danger Level: {vision_analysis.danger_level}
Opportunities: {', '.join(vision_analysis.opportunities) or 'none identified'}

## Context
Goal: {goal}
Recent Actions: {history_str}
Game Type: {context.get('game_type', 'Game Boy')}

## Task
Decide the SINGLE BEST button press next. Valid buttons:
UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

First briefly reason about the situation, then respond with ONLY the button name.

Your decision:"""

    def _parse_vision_response(self, response: str, context: Dict[str, Any]) -> VisionAnalysis:
        """Parse vision model response into structured analysis"""
        response_lower = response.lower()
        
        # Extract game state
        game_state = "unknown"
        if "battle" in response_lower:
            game_state = "battle"
        elif "menu" in response_lower:
            game_state = "menu"
        elif "dialog" in response_lower or "text" in response_lower:
            game_state = "dialog"
        elif "exploring" in response_lower or "overworld" in response_lower:
            game_state = "exploring"
        elif "title" in response_lower:
            game_state = "title_screen"
        
        # Extract danger level
        danger = "low"
        if "high danger" in response_lower or "critical" in response_lower:
            danger = "high"
        elif "medium danger" in response_lower or "caution" in response_lower:
            danger = "medium"
        
        # Extract entities (simplified)
        entities = []
        entity_keywords = ["npc", "enemy", "pokemon", "trainer", "item", "person"]
        for keyword in entity_keywords:
            if keyword in response_lower:
                entities.append(keyword)
        
        return VisionAnalysis(
            game_state=game_state,
            raw_description=response[:500],  # Truncate for storage
            danger_level=danger,
            nearby_entities=entities[:5],  # Limit to 5
        )
    
    def _parse_planning_response(self, response: str) -> PlanningResult:
        """Parse planning model response into action"""
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        
        # Find action in response
        response_upper = response.upper()
        action = None
        
        # Look for explicit action
        for valid_action in valid_actions:
            if valid_action in response_upper:
                action = valid_action
                break
        
        if not action:
            # Try to extract from common patterns
            action_words = response_upper.split()
            for word in action_words:
                if word in valid_actions:
                    action = word
                    break
        
        if not action:
            action = 'A'  # Default to A button
        
        # Extract reasoning (everything before the action)
        reasoning = response.split('\n')[0] if '\n' in response else response[:200]
        
        return PlanningResult(
            action=action,
            reasoning=reasoning,
            confidence=0.8 if action in response_upper else 0.5
        )
    
    def _get_model_id(self, model: str, is_vision: bool) -> str:
        """Get full model ID for OpenClaw API"""
        if is_vision:
            return self.VISION_CAPABLE_MODELS.get(model, model)
        else:
            return self.PLANNING_MODELS.get(model, model)
    
    def _get_fallback_action(self, action_history: List[str]) -> str:
        """Get fallback action when planning fails"""
        if not action_history:
            return 'A'
        
        # Rotate through actions
        last = action_history[-1] if action_history else 'A'
        rotation = {
            'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'A',
            'A': 'B', 'B': 'UP'
        }
        return rotation.get(last, 'A')
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dual-model provider status"""
        return {
            "vision_model": self.vision_model,
            "planning_model": self.planning_model,
            "openclaw_endpoint": self.openclaw_endpoint,
            "last_vision_response": self.last_vision_response[:200] if self.last_vision_response else None,
            "last_planning_response": self.last_planning_response[:200] if self.last_planning_response else None,
            "available_vision_models": list(self.VISION_CAPABLE_MODELS.keys()),
            "available_planning_models": list(self.PLANNING_MODELS.keys()),
        }