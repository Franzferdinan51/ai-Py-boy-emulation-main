"""
OpenClaw AI Provider - Uses OpenClaw MCP for AI actions
This provider integrates with the local OpenClaw Gateway for unified model access.
"""
import logging
import requests
from typing import List, Optional, Any, Dict
from .ai_api_base import AIAPIConnector

logger = logging.getLogger(__name__)


class OpenClawAIProvider(AIAPIConnector):
    """
    OpenClaw AI Provider - Routes AI requests through OpenClaw Gateway
    Uses whatever models are configured in OpenClaw (Bailian, LM Studio, etc.)
    """

    def __init__(self, api_key: str = "openclaw-mcp-key", base_url: str = "http://localhost:18789"):
        """
        Initialize OpenClaw AI Provider
        
        Args:
            api_key: Not used (OpenClaw uses local MCP), but required for interface compatibility
            base_url: OpenClaw Gateway MCP endpoint
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = "openclaw/auto"  # Let OpenClaw choose the best model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"OpenClaw AI Provider initialized at {self.base_url}")

    def get_models(self) -> List[str]:
        """Get available models from OpenClaw"""
        try:
            # Try to get models from OpenClaw session status
            response = requests.get(f"{self.base_url}/session/status", timeout=5)
            if response.ok:
                data = response.json()
                models = []
                if 'model' in data:
                    models.append(data['model'])
                if 'available_models' in data:
                    models.extend(data['available_models'])
                return models if models else ["openclaw/auto"]
            return ["openclaw/auto"]
        except Exception as e:
            self.logger.debug(f"Could not fetch OpenClaw models: {e}")
            return ["openclaw/auto"]

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str]) -> str:
        """
        Get next action using OpenClaw
        
        Args:
            image_bytes: Current game screen
            goal: Current objective
            action_history: List of recent actions
            
        Returns:
            Next action to take
        """
        import base64
        
        # Encode image as base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Build prompt for game action
        prompt = self._build_game_prompt(goal, action_history)
        
        try:
            # Use OpenClaw's vision capability
            response = self._call_openclaw_vision(image_base64, prompt)
            if response:
                action = self._parse_action(response)
                self.logger.info(f"OpenClaw suggested action: {action}")
                return action
        except Exception as e:
            self.logger.error(f"OpenClaw vision call failed: {e}")
        
        # Fallback to text-only if vision fails
        try:
            response = self._call_openclaw_text(prompt)
            if response:
                action = self._parse_action(response)
                self.logger.info(f"OpenClaw (text) suggested action: {action}")
                return action
        except Exception as e:
            self.logger.error(f"OpenClaw text call failed: {e}")
        
        # Ultimate fallback
        return self._get_fallback_action(action_history)

    def chat_with_ai(self, message: str, image_bytes: bytes, context: dict) -> str:
        """
        Chat with AI via OpenClaw
        
        Args:
            message: User message
            image_bytes: Optional image (game screen)
            context: Conversation context
            
        Returns:
            AI response
        """
        import base64
        
        # Build chat prompt
        prompt = self._build_chat_prompt(message, context)
        
        try:
            # If image provided, use vision
            if image_bytes:
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                response = self._call_openclaw_vision(image_base64, prompt)
                if response:
                    self.logger.info("OpenClaw vision chat response received")
                    return response
        except Exception as e:
            self.logger.error(f"OpenClaw vision chat failed: {e}")
        
        # Fallback to text-only
        try:
            response = self._call_openclaw_text(prompt)
            if response:
                self.logger.info("OpenClaw text chat response received")
                return response
        except Exception as e:
            self.logger.error(f"OpenClaw text chat failed: {e}")
        
        return "I'm sorry, I couldn't connect to OpenClaw at the moment. Please check your OpenClaw Gateway connection."

    def _build_game_prompt(self, goal: str, action_history: List[str]) -> str:
        """Build prompt for game action decision"""
        history_str = ", ".join(action_history[-5:]) if action_history else "none"
        return f"""You are playing a Game Boy game. Your goal: {goal}

Recent actions: {history_str}

What button should I press next? Respond with ONLY one of these buttons:
UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

Choose the action that makes the most progress toward the goal."""

    def _build_chat_prompt(self, message: str, context: dict) -> str:
        """Build prompt for chat conversation"""
        game_state = context.get('game_state', {})
        rom_name = game_state.get('rom_name', 'unknown')
        return f"""You are helping a player with their Game Boy game: {rom_name}

User question: {message}

Provide a helpful, concise response about the game."""

    def _call_openclaw_vision(self, image_base64: str, prompt: str) -> Optional[str]:
        """Call OpenClaw with vision capability"""
        # This would integrate with OpenClaw's actual API
        # For now, use a simple HTTP call to the gateway
        try:
            payload = {
                "task": prompt,
                "image": image_base64,
                "model": self.model
            }
            response = requests.post(
                f"{self.base_url}/api/vision/analyze",
                json=payload,
                timeout=30
            )
            if response.ok:
                data = response.json()
                return data.get('response') or data.get('text') or data.get('result')
        except Exception as e:
            self.logger.debug(f"OpenClaw vision API call failed: {e}")
        return None

    def _call_openclaw_text(self, prompt: str) -> Optional[str]:
        """Call OpenClaw text-only API"""
        try:
            payload = {
                "prompt": prompt,
                "model": self.model
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            if response.ok:
                data = response.json()
                return data.get('response') or data.get('text') or data.get('result')
        except Exception as e:
            self.logger.debug(f"OpenClaw text API call failed: {e}")
        return None

    def _parse_action(self, response: str) -> str:
        """Parse action from AI response"""
        valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        
        # Look for exact action match
        response_upper = response.upper()
        for action in valid_actions:
            if action in response_upper:
                return action
        
        # Fallback to first word
        first_word = response.split()[0].upper() if response else ''
        if first_word in valid_actions:
            return first_word
        
        # Default
        return 'A'

    def _get_fallback_action(self, action_history: List[str]) -> str:
        """Get fallback action when all else fails"""
        if not action_history:
            return 'UP'
        
        # Avoid repeating last action
        last_action = action_history[-1]
        alternatives = {'UP': 'A', 'DOWN': 'B', 'LEFT': 'RIGHT', 'RIGHT': 'A', 'A': 'UP', 'B': 'DOWN'}
        return alternatives.get(last_action, 'UP')

    def cleanup(self):
        """Clean up resources"""
        pass
