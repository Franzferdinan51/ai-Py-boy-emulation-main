"""
LM Studio AI API Connector
Supports local LM Studio endpoints with separate thinking and vision models
"""
import base64
import os
import requests
from typing import List, Optional, Dict, Any
from .ai_api_base import AIAPIConnector
from openai import OpenAI


class LMStudioConnector(AIAPIConnector):
    """LM Studio AI API connector with support for separate thinking and vision models"""
    
    def __init__(self, api_key: str = "not-needed", base_url: str = None, 
                 thinking_model: str = None, vision_model: str = None):
        """
        Initialize LM Studio connector
        
        Args:
            api_key: API key (usually not needed for local LM Studio)
            base_url: LM Studio endpoint URL (default: http://localhost:1234/v1)
            thinking_model: Model for text/thinking tasks
            vision_model: Model for vision/image analysis tasks
        """
        super().__init__(api_key or "not-needed")
        
        # Validate and set base URL
        self.base_url = self._validate_base_url(base_url)
        
        # Model configuration - supports separate thinking and vision models
        self.thinking_model = thinking_model
        self.vision_model = vision_model
        self.model = thinking_model  # Default to thinking model for backward compatibility
        
        # Valid actions for game control
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
        
        # Timeout and retry configuration
        self.timeout = int(os.environ.get('AI_TIMEOUT', '60'))
        self.max_retries = int(os.environ.get('AI_MAX_RETRIES', '3'))
        
        # Initialize OpenAI-compatible client
        self.client = self._initialize_client()
        
        # Test connection during initialization (non-blocking)
        self._test_connection()
    
    def _validate_base_url(self, base_url: Optional[str]) -> str:
        """Validate and normalize base URL"""
        if not base_url:
            # Check environment variables
            base_url = (os.environ.get('LM_STUDIO_URL') or 
                       os.environ.get('OPENAI_ENDPOINT') or 
                       os.environ.get('AI_ENDPOINT'))
        
        if not base_url:
            # Default to local LM Studio
            return "http://localhost:1234/v1"
        
        # Normalize URL
        base_url = base_url.rstrip('/')
        
        # Handle common LM Studio URL patterns
        if base_url in ['localhost', '127.0.0.1', 'lm-studio']:
            base_url = "http://localhost:1234/v1"
        elif 'lm-studio' in base_url and not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}:1234/v1"
        elif not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        # Ensure /v1 suffix for OpenAI compatibility
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        return base_url
    
    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI-compatible client"""
        try:
            # For local LM Studio, API key is typically not needed
            api_key = self.api_key or "not-needed"
            
            client = OpenAI(
                api_key=api_key, 
                base_url=self.base_url, 
                timeout=self.timeout
            )
            
            self.logger.info(f"Initialized LM Studio client at {self.base_url}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LM Studio client: {e}")
            return None
    
    def _test_connection(self):
        """Test connection to LM Studio"""
        if not self.client:
            self.logger.warning("Cannot test connection - no client initialized")
            return
        
        try:
            # Simple test request to list models
            response = self.client.models.list()
            model_count = len(response.data) if hasattr(response, 'data') else 0
            self.logger.info(f"LM Studio connection test successful. Available models: {model_count}")
            
            # Log available models for debugging
            if hasattr(response, 'data') and response.data:
                model_names = [model.id for model in response.data[:5]]
                self.logger.info(f"Sample models: {', '.join(model_names)}")
                
        except Exception as e:
            self.logger.warning(f"LM Studio connection test failed: {e}")
            # Don't raise - allow connector to work even if test fails
    
    def get_models(self) -> List[str]:
        """Get list of available models from LM Studio"""
        if not self.client:
            self.logger.warning("Cannot fetch models - client not initialized")
            return []
        
        try:
            response = self.client.models.list()
            models = [model.id for model in response.data]
            self.logger.info(f"Found {len(models)} models in LM Studio")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to fetch models from LM Studio: {e}")
            return []
    
    def set_thinking_model(self, model: str):
        """Set the thinking model for text-based tasks"""
        self.thinking_model = model
        self.model = model  # Also update default model
        self.logger.info(f"Set thinking model to: {model}")
    
    def set_vision_model(self, model: str):
        """Set the vision model for image analysis tasks"""
        self.vision_model = model
        self.logger.info(f"Set vision model to: {model}")
    
    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str]) -> str:
        """
        Get the next action from LM Studio based on game state
        
        Args:
            image_bytes: Current game screen image
            goal: Current objective/goal
            action_history: List of recent actions taken
            
        Returns:
            Next action string (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT)
        """
        if not self.client:
            self.logger.error("LM Studio client not initialized")
            return self._get_fallback_action(action_history)
        
        try:
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create action prompt
            prompt = self._create_action_prompt(goal, action_history)
            
            # Use vision model if available, otherwise use thinking model
            model_to_use = self.vision_model or self.thinking_model or self.model
            
            if not model_to_use:
                # Try to get first available model
                models = self.get_models()
                if not models:
                    self.logger.error("No models available in LM Studio")
                    return self._get_fallback_action(action_history)
                model_to_use = models[0]
            
            self.logger.debug(f"LM Studio request - Model: {model_to_use}, Goal: {goal}")
            
            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=10,
                    temperature=0.7,
                    timeout=self.timeout
                )
            
            response = self._retry_with_backoff(make_request)
            
            # Parse and validate action
            action = self._parse_action_response(response)
            if action in self.valid_actions:
                self.logger.info(f"LM Studio returned valid action: {action}")
                return action
            else:
                self.logger.warning(f"LM Studio returned invalid action: '{action}'. Using fallback.")
                return self._get_fallback_action(action_history)
                
        except Exception as e:
            self.logger.error(f"Error calling LM Studio: {e}", exc_info=True)
            raise e
    
    def chat_with_ai(self, user_message: str, image_bytes: bytes, context: dict) -> str:
        """
        Chat with AI about game state
        
        Args:
            user_message: User's question or message
            image_bytes: Current game screen image
            context: Additional context (goal, action history, etc.)
            
        Returns:
            AI response text
        """
        if not self.client:
            return "I'm sorry, the AI service is not available right now."
        
        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use thinking model for chat (unless vision is needed)
            model_to_use = self.thinking_model or self.model
            
            if not model_to_use:
                models = self.get_models()
                model_to_use = models[0] if models else "default"
            
            prompt = self._create_chat_prompt(user_message, context)
            
            def make_request():
                return self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful game assistant with expertise in retro video games."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]}
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    timeout=self.timeout
                )
            
            response = self._retry_with_backoff(make_request)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error calling LM Studio for chat: {e}", exc_info=True)
            return "I'm sorry, I encountered an error processing your request."
    
    def _create_action_prompt(self, goal: str, action_history: List[str]) -> str:
        """Create enhanced prompt for action selection"""
        recent_actions = ', '.join(action_history[-5:]) if action_history else "none"
        
        return f"""You are an expert AI playing a Game Boy game. Your current objective is: "{goal}".

GAME BOY CONTROLS:
- D-PAD: UP, DOWN, LEFT, RIGHT (move character, navigate menus)
- ACTION BUTTONS: A (primary action/jump/confirm), B (secondary action/cancel/back)
- SYSTEM BUTTONS: START (pause/start game), SELECT (menu/option selection)

Recent actions: {recent_actions}.

Analyze the game screen and choose the BEST single action to progress toward your objective.
Consider game context: character position, enemies, items, menus, dialogue boxes, etc.

Your response MUST be exactly one of: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT.
No explanation - just the action word."""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI"""
        return "You are an expert AI playing a retro video game. Respond with only the action name in uppercase."
    
    def _parse_action_response(self, response) -> str:
        """Parse and clean action response from AI"""
        try:
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content.strip().upper()
                # Clean up common response issues
                content = content.replace('.', '').replace(',', '').strip()
                return content
            return "SELECT"
        except Exception as e:
            self.logger.error(f"Error parsing action response: {e}")
            return "SELECT"
    
    def _get_fallback_action(self, action_history: List[str]) -> str:
        """Get fallback action when AI fails"""
        if not action_history:
            return "UP"
        
        last_action = action_history[-1]
        if last_action == "UP":
            return "RIGHT"
        elif last_action == "RIGHT":
            return "DOWN"
        elif last_action == "DOWN":
            return "LEFT"
        else:
            return "A"
    
    def _create_chat_prompt(self, user_message: str, context: dict) -> str:
        """Create enhanced chat prompt with context"""
        current_goal = context.get('current_goal', 'None')
        action_history = ', '.join(context.get('action_history', [])[-5:]) if context.get('action_history') else "none"
        game_type = context.get('game_type', 'Unknown')
        
        return f"""User Message: {user_message}

Game Context:
- Current Goal: {current_goal}
- Recent Actions: {action_history}
- Game Type: {game_type}

Please provide a helpful response based on the game screen and the user's message. You can see the current game state and provide specific advice about what's happening in the game."""
