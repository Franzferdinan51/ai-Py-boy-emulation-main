"""
AI Provider Manager - Handles automatic provider detection and fallback

Supports dual-model architecture:
- Vision Model: Screen analysis and game state extraction
- Planning Model: Decision making based on vision analysis
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .ai_api_base import AIAPIConnector
from .gemini_api import GeminiAPIConnector
from .openrouter_api import OpenRouterAPIConnector
from .openai_compatible import OpenAICompatibleConnector
from .nvidia_api import NVIDIAAPIConnector
from .mock_ai_provider import MockAIProvider
from .tetris_genetic_ai import TetrisGeneticAI
from .openclaw_ai_provider import OpenClawAIProvider
from .lmstudio_connector import LMStudioConnector
from .dual_model_provider import DualModelProvider
from .dual_model_provider import DualModelProvider

class ProviderStatus(Enum):
    """Provider status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"

class AIProviderManager:
    """Manages AI providers with automatic detection and fallback
    
    Supports dual-model architecture:
    - Vision Model: Analyzes screenshots, extracts game state
    - Planning Model: Makes decisions based on vision analysis
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.provider_order: List[str] = []
        self.fallback_providers: List[str] = []
        self.last_refresh_time = 0
        self.refresh_interval = 300  # Refresh every 5 minutes
        # Default to openclaw for OpenClaw-native integration, fallback to mock only if explicitly set
        self.default_provider = os.environ.get('DEFAULT_AI_PROVIDER', 'openclaw')
        
        # Dual-model architecture support
        self.dual_model_provider: Optional[DualModelProvider] = None
        self.use_dual_model = os.environ.get('USE_DUAL_MODEL', 'true').lower() == 'true'
        self.vision_model = os.environ.get('VISION_MODEL', 'kimi-k2.5')
        self.planning_model = os.environ.get('PLANNING_MODEL', 'glm-5')
        
        self.initialize_providers()
        # self._initialize_dual_model()  # Temporarily disabled - dual model not yet implemented

    def _initialize_dual_model(self):
        """Initialize dual-model architecture (vision + planning)"""
        # TODO: Implement dual-model support
        self.logger.debug("Dual-model initialization not yet implemented")
        pass

    def initialize_providers(self):
        """Initialize all available AI providers"""
        self.logger.info("Initializing AI providers...")

        # Define provider configurations (OpenClaw-first for native integration)
        provider_configs = [
            {
                'name': 'openclaw',
                'env_key': None,  # No API key required - uses local MCP
                'class': OpenClawAIProvider,
                'priority': 1,  # Highest priority for OpenClaw-native
                'extra_params': {
                    'base_url': os.environ.get('OPENCLAW_MCP_ENDPOINT', 'http://localhost:18789')
                }
            },
            {
                'name': 'lmstudio',
                'env_key': None,  # No API key required for local LM Studio
                'class': LMStudioConnector,
                'priority': 2,  # High priority for local models
                'extra_params': {
                    'base_url': os.environ.get('LM_STUDIO_URL'),
                    'thinking_model': os.environ.get('LM_STUDIO_THINKING_MODEL'),
                    'vision_model': os.environ.get('LM_STUDIO_VISION_MODEL')
                }
            },
            {
                'name': 'gemini',
                'env_key': 'GEMINI_API_KEY',
                'class': GeminiAPIConnector,
                'priority': 3
            },
            {
                'name': 'openrouter',
                'env_key': 'OPENROUTER_API_KEY',
                'class': OpenRouterAPIConnector,
                'priority': 4
            },
            {
                'name': 'openai-compatible',
                'env_key': 'OPENAI_API_KEY',
                'class': OpenAICompatibleConnector,
                'priority': 5,
                'extra_params': {
                    'base_url': os.environ.get('OPENAI_ENDPOINT')
                }
            },
            {
                'name': 'nvidia',
                'env_key': 'NVIDIA_API_KEY',
                'class': NVIDIAAPIConnector,
                'priority': 6
            },
            {
                'name': 'mock',
                'env_key': None,  # No API key required
                'class': MockAIProvider,
                'priority': 99,  # Lowest priority - only if nothing else works
                'extra_params': {}
            },
            {
                'name': 'tetris-genetic',
                'env_key': None,  # No API key required
                'class': TetrisGeneticAI,
                'priority': 10,  # Lower priority for specialized AI
                'extra_params': {}
            }
        ]

        # Initialize providers
        for config in provider_configs:
            self._initialize_provider(config)

        # Sort providers by priority
        self.provider_order = sorted(
            [name for name, info in self.providers.items() if info['status'] == ProviderStatus.AVAILABLE],
            key=lambda x: self.providers[x]['priority']
        )

        # Set up fallback providers
        self.fallback_providers = self.provider_order.copy()
        self.logger.info(f"Provider initialization complete. Available providers: {self.provider_order}")

    def _initialize_provider(self, config: Dict[str, Any]):
        """Initialize a single provider"""
        name = config['name']
        env_key = config['env_key']
        provider_class = config['class']
        priority = config['priority']
        extra_params = config.get('extra_params', {})

        try:
            # Special handling for providers that don't need API keys
            if name == 'openclaw':
                # OpenClaw uses local MCP - no API key needed
                api_key = "openclaw-mcp-key"
                self.logger.info(f"Using OpenClaw provider - local MCP integration")
            elif name == 'lmstudio':
                # LM Studio uses local endpoint - no API key needed
                api_key = "not-needed"
                self.logger.info(f"Using LM Studio provider - local inference")
                
                # Ensure base_url is set from environment or default
                base_url = (extra_params.get('base_url') or
                           os.environ.get('LM_STUDIO_URL') or
                           'http://localhost:1234/v1')
                extra_params['base_url'] = base_url
                
                # Log model configuration
                if extra_params.get('thinking_model'):
                    self.logger.info(f"LM Studio thinking model: {extra_params['thinking_model']}")
                if extra_params.get('vision_model'):
                    self.logger.info(f"LM Studio vision model: {extra_params['vision_model']}")
                    
            elif name == 'mock':
                # Mock provider never needs API key
                api_key = "mock-key"
                self.logger.info(f"Using mock provider - no API key required")
            elif name == 'tetris-genetic':
                # Tetris genetic AI doesn't need API key
                api_key = "genetic-key"
                self.logger.info(f"Using tetris-genetic provider - no API key required")
            elif env_key:
                api_key = os.environ.get(env_key)
            else:
                api_key = None

            # For local providers, check multiple environment variables
            if name == 'openai-compatible':
                # Check various environment variables for local endpoints
                base_url = (extra_params.get('base_url') or
                           os.environ.get('OPENAI_ENDPOINT') or
                           os.environ.get('LM_STUDIO_URL') or
                           os.environ.get('AI_ENDPOINT') or
                           os.environ.get('OLLAMA_URL'))

                if base_url and ('localhost' in base_url or '127.0.0.1' in base_url):
                    # For local providers, API key is optional
                    api_key = api_key or "not-needed"
                    extra_params['base_url'] = base_url
                elif not base_url and os.environ.get('LM_STUDIO_URL'):
                    # LM Studio specific fallback
                    base_url = os.environ.get('LM_STUDIO_URL')
                    api_key = api_key or "not-needed"
                    extra_params['base_url'] = base_url
                elif not base_url and os.environ.get('OLLAMA_URL'):
                    # Ollama specific fallback
                    base_url = os.environ.get('OLLAMA_URL')
                    api_key = api_key or "not-needed"
                    extra_params['base_url'] = base_url
                elif not base_url and os.environ.get('OPENAI_ENDPOINT'):
                    # OpenAI endpoint fallback
                    base_url = os.environ.get('OPENAI_ENDPOINT')
                    extra_params['base_url'] = base_url

            # Special handling for providers that might work without API keys
            if name == 'mock':
                # Mock provider never needs API key
                api_key = "mock-key"
                self.logger.info(f"Using mock provider - no API key required")
            elif name == 'tetris-genetic':
                # Tetris genetic AI doesn't need API key
                api_key = "genetic-key"
                self.logger.info(f"Using tetris-genetic provider - no API key required")
            elif name == 'openai-compatible' and not api_key:
                # Check if we have a local endpoint
                base_url = extra_params.get('base_url')
                if base_url and ('localhost' in base_url or '127.0.0.1' in base_url):
                    # Local provider - no API key needed
                    api_key = "not-needed"
                    self.logger.info(f"Using local {name} provider at {base_url} without API key")
                else:
                    self.logger.info(f"API key not found for {name} (environment variable: {env_key})")
                    self.providers[name] = {
                        'status': ProviderStatus.UNAVAILABLE,
                        'connector': None,
                        'priority': priority,
                        'error': f"API key not found in environment variable: {env_key}"
                    }
                    return

            if not api_key:
                self.logger.info(f"API key not found for {name} (environment variable: {env_key})")
                self.providers[name] = {
                    'status': ProviderStatus.UNAVAILABLE,
                    'connector': None,
                    'priority': priority,
                    'error': f"API key not found in environment variable: {env_key}"
                }
                return

            # Initialize the connector with custom model support
            try:
                # Get custom model from environment if available
                model_env_vars = {
                    'gemini': 'GEMINI_MODEL',
                    'openrouter': 'OPENROUTER_MODEL',
                    'nvidia': 'NVIDIA_MODEL',
                    'openai-compatible': 'OPENAI_MODEL'
                }

                custom_model = None
                if name in model_env_vars:
                    custom_model = os.environ.get(model_env_vars[name])

                # Initialize connector with model parameter if supported
                if custom_model and hasattr(provider_class, '__init__'):
                    # Check if the constructor accepts a model parameter
                    import inspect
                    init_signature = inspect.signature(provider_class.__init__)
                    if 'model' in init_signature.parameters:
                        if extra_params:
                            connector = provider_class(api_key, model=custom_model, **extra_params)
                        else:
                            connector = provider_class(api_key, model=custom_model)
                    else:
                        if extra_params:
                            connector = provider_class(api_key, **extra_params)
                        else:
                            connector = provider_class(api_key)
                else:
                    if extra_params:
                        connector = provider_class(api_key, **extra_params)
                    else:
                        connector = provider_class(api_key)

                # Test the connection with a simple request
                test_result = self._test_provider_connection(connector, name)
                if test_result:
                    self.providers[name] = {
                        'status': ProviderStatus.AVAILABLE,
                        'connector': connector,
                        'priority': priority,
                        'error': None
                    }
                    self.logger.info(f"Successfully initialized {name} provider")
                else:
                    self.providers[name] = {
                        'status': ProviderStatus.ERROR,
                        'connector': connector,
                        'priority': priority,
                        'error': "Connection test failed"
                    }
                    self.logger.warning(f"Connection test failed for {name} provider")

            except Exception as e:
                self.logger.error(f"Failed to initialize {name} provider: {e}", exc_info=True)
                self.providers[name] = {
                    'status': ProviderStatus.ERROR,
                    'connector': None,
                    'priority': priority,
                    'error': str(e)
                }

        except Exception as e:
            self.logger.error(f"Failed to initialize {name} provider: {e}", exc_info=True)
            self.providers[name] = {
                'status': ProviderStatus.ERROR,
                'connector': None,
                'priority': priority,
                'error': str(e)
            }

    def _test_provider_connection(self, connector: AIAPIConnector, provider_name: str) -> bool:
        """Test if a provider connection is working"""
        try:
            # OpenClaw and mock providers are always available (local)
            if provider_name in ('openclaw', 'mock'):
                return True

            # Try to get models list as a simple test for other providers
            models = connector.get_models()
            return models is not None and len(models) > 0
        except Exception as e:
            self.logger.debug(f"Connection test failed for {provider_name}: {e}")
            return False

    def get_provider(self, provider_name: Optional[str] = None) -> Optional[AIAPIConnector]:
        """Get a provider connector by name, or use the first available one"""
        if provider_name:
            # Try specific provider first
            if provider_name in self.providers:
                provider_info = self.providers[provider_name]
                if provider_info['status'] == ProviderStatus.AVAILABLE:
                    self.logger.info(f"Using requested provider: {provider_name}")
                    return provider_info['connector']
                else:
                    self.logger.warning(f"Provider {provider_name} is not available: {provider_info.get('error', 'Unknown error')}")
                    self.logger.info(f"Falling back to available providers: {self.provider_order}")
            else:
                self.logger.warning(f"Unknown provider: {provider_name}")
                self.logger.info(f"Falling back to available providers: {self.provider_order}")

        # Fall back to automatic provider selection
        for provider_name in self.provider_order:
            provider_info = self.providers[provider_name]
            if provider_info['status'] == ProviderStatus.AVAILABLE:
                self.logger.info(f"Using provider: {provider_name}")
                return provider_info['connector']

        self.logger.error("No available AI providers found")
        return None

    def get_next_action(self, image_bytes: bytes, goal: str, action_history: List[str],
                      provider_name: Optional[str] = None, model: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Get next action with automatic fallback"""
        self.logger.info(f"get_next_action called with provider: {provider_name}, available_providers: {self.get_available_providers()}")

        # Try specified provider first
        if provider_name:
            self.logger.info(f"Trying specified provider: {provider_name}")
            connector = self.get_provider(provider_name)
            if connector:
                try:
                    if model:
                        connector.model = model
                    action = connector.get_next_action(image_bytes, goal, action_history)
                    self.logger.info(f"Successfully got action from {provider_name}: {action}")
                    return action, provider_name
                except Exception as e:
                    self.logger.error(f"Provider {provider_name} failed: {e}")
                    # Continue to fallback
            else:
                self.logger.warning(f"Provider {provider_name} not available, falling back to: {self.provider_order}")

        # Try providers in order
        self.logger.info(f"Trying fallback providers in order: {self.fallback_providers}")
        for fallback_provider in self.fallback_providers:
            self.logger.info(f"Attempting fallback provider: {fallback_provider}")
            connector = self.get_provider(fallback_provider)
            if connector:
                try:
                    # Set model if provided
                    if model:
                        connector.model = model
                    action = connector.get_next_action(image_bytes, goal, action_history)
                    self.logger.info(f"Successfully used fallback provider: {fallback_provider}, action: {action}")
                    return action, fallback_provider
                except Exception as e:
                    self.logger.error(f"Fallback provider {fallback_provider} failed: {e}")
                    continue
            else:
                self.logger.warning(f"Fallback provider {fallback_provider} not available")

        # Ultimate fallback - use a default action
        self.logger.error("All providers failed, using default action")
        return self._get_default_action(action_history), None

    def chat_with_ai(self, message: str, image_bytes: bytes, context: dict,
                    provider_name: Optional[str] = None, model: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Chat with AI with automatic fallback"""
        # Try specified provider first
        if provider_name:
            connector = self.get_provider(provider_name)
            if connector:
                try:
                    if model:
                        connector.model = model
                    response = connector.chat_with_ai(message, image_bytes, context)
                    return response, provider_name
                except Exception as e:
                    self.logger.error(f"Provider {provider_name} failed: {e}")
                    # Continue to fallback
            else:
                self.logger.warning(f"Provider {provider_name} not available, falling back")

        # Try providers in order
        for fallback_provider in self.fallback_providers:
            connector = self.get_provider(fallback_provider)
            if connector:
                try:
                    # Set model if provided
                    if model:
                        connector.model = model
                    response = connector.chat_with_ai(message, image_bytes, context)
                    self.logger.info(f"Successfully used fallback provider for chat: {fallback_provider}")
                    return response, fallback_provider
                except Exception as e:
                    self.logger.error(f"Fallback provider {fallback_provider} failed: {e}")
                    continue

        # Ultimate fallback
        return "I'm sorry, all AI services are currently unavailable. Please try again later.", None

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        for name, info in self.providers.items():
            status[name] = {
                'status': info['status'].value,
                'priority': info['priority'],
                'error': info.get('error'),
                'available': info['status'] == ProviderStatus.AVAILABLE
            }
        return status

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, info in self.providers.items() if info['status'] == ProviderStatus.AVAILABLE]

    def get_models(self, provider_name: str) -> List[str]:
        """Get a list of available models for a given provider"""
        provider = self.get_provider(provider_name)
        if provider:
            try:
                return provider.get_models()
            except Exception as e:
                self.logger.error(f"Failed to get models for {provider_name}: {e}")
                return []
        return []

    def _get_default_action(self, action_history: List[str]) -> str:
        """Get a default action when all providers fail"""
        # More intelligent default strategy
        if not action_history:
            return "UP"

        # Check if we're stuck in a pattern
        if len(action_history) >= 3:
            last_three = action_history[-3:]
            if len(set(last_three)) == 1:  # Same action repeated 3 times
                # Break the pattern
                if last_three[0] == "UP":
                    return "A"
                elif last_three[0] == "A":
                    return "RIGHT"
                else:
                    return "UP"

        # Check last action and vary it
        last_action = action_history[-1]
        action_cycle = {
            'UP': 'RIGHT',
            'RIGHT': 'DOWN',
            'DOWN': 'LEFT',
            'LEFT': 'A',
            'A': 'B',
            'B': 'START',
            'START': 'SELECT',
            'SELECT': 'UP'
        }

        return action_cycle.get(last_action, "UP")

    def refresh_provider_status(self):
        """Refresh the status of all providers"""
        self.logger.info("Refreshing provider status...")
        for name, info in self.providers.items():
            if info['connector']:
                try:
                    # Simple test - try to access a basic property
                    if hasattr(info['connector'], 'client'):
                        if info['connector'].client:
                            info['status'] = ProviderStatus.AVAILABLE
                        else:
                            info['status'] = ProviderStatus.UNAVAILABLE
                    else:
                        info['status'] = ProviderStatus.AVAILABLE
                except Exception as e:
                    info['status'] = ProviderStatus.ERROR
                    info['error'] = str(e)
        self.logger.info("Provider status refresh complete")

    def cleanup(self):
        """Clean up all provider resources"""
        self.logger.info("Cleaning up AI provider manager...")
        for name, info in self.providers.items():
            if info['connector']:
                try:
                    # Call cleanup method if it exists
                    if hasattr(info['connector'], 'cleanup'):
                        info['connector'].cleanup()
                    # Clear references
                    info['connector'] = None
                    self.logger.debug(f"Cleaned up {name} provider")
                except Exception as e:
                    self.logger.error(f"Error cleaning up {name} provider: {e}")

        # Clean up dual-model provider
        if self.dual_model_provider:
            self.dual_model_provider = None

        # Clear all provider references
        self.providers.clear()
        self.provider_order.clear()
        self.fallback_providers.clear()
        self.logger.info("AI provider manager cleanup complete")
    
    # ========================================================================
    # DUAL-MODEL ARCHITECTURE METHODS
    # ========================================================================
    
    def _initialize_dual_model(self):
        """Initialize the dual-model provider"""
        try:
            openclaw_endpoint = os.environ.get('OPENCLAW_MCP_ENDPOINT', 'http://localhost:18789')
            self.dual_model_provider = DualModelProvider(
                openclaw_endpoint=openclaw_endpoint,
                vision_model=self.vision_model,
                planning_model=self.planning_model,
            )
            self.logger.info(
                f"Dual-model provider initialized: "
                f"vision={self.vision_model}, planning={self.planning_model}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize dual-model provider: {e}")
            self.dual_model_provider = None
    
    def set_vision_model(self, model: str) -> bool:
        """Set the vision model for dual-model architecture"""
        self.vision_model = model
        if self.dual_model_provider:
            return self.dual_model_provider.set_vision_model(model)
        return False
    
    def set_planning_model(self, model: str) -> bool:
        """Set the planning model for dual-model architecture"""
        self.planning_model = model
        if self.dual_model_provider:
            return self.dual_model_provider.set_planning_model(model)
        return False
    
    def get_dual_model_status(self) -> Dict[str, Any]:
        """Get dual-model provider status"""
        if self.dual_model_provider:
            return self.dual_model_provider.get_status()
        return {
            "available": False,
            "vision_model": self.vision_model,
            "planning_model": self.planning_model,
            "error": "Dual-model provider not initialized"
        }
    
    def get_next_action_dual_model(
        self,
        image_bytes: bytes,
        goal: str,
        action_history: List[str],
        context: Dict[str, Any] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Get next action using dual-model architecture.
        
        Flow: Screenshot -> Vision Model -> Planning Model -> Action
        
        Args:
            image_bytes: Raw screenshot bytes
            goal: Current objective
            action_history: Recent actions
            context: Additional context
            
        Returns:
            Tuple of (action, models_used)
        """
        if not self.dual_model_provider:
            self.logger.warning("Dual-model provider not available, falling back to single model")
            return self.get_next_action(image_bytes, goal, action_history)
        
        try:
            return self.dual_model_provider.get_next_action(
                image_bytes, goal, action_history, context
            )
        except Exception as e:
            self.logger.error(f"Dual-model action failed: {e}")
            # Fallback to single model
            return self.get_next_action(image_bytes, goal, action_history)
    
    def configure_dual_model(
        self,
        vision_model: str = None,
        planning_model: str = None,
        use_dual_model: bool = None
    ) -> Dict[str, Any]:
        """
        Configure dual-model architecture settings.
        
        Args:
            vision_model: Vision model identifier
            planning_model: Planning model identifier
            use_dual_model: Enable/disable dual-model
            
        Returns:
            Configuration status
        """
        result = {
            "success": True,
            "changes": []
        }
        
        if vision_model is not None:
            if self.set_vision_model(vision_model):
                result["changes"].append(f"vision_model: {vision_model}")
            else:
                result["success"] = False
                result["error"] = f"Failed to set vision model: {vision_model}"
        
        if planning_model is not None:
            if self.set_planning_model(planning_model):
                result["changes"].append(f"planning_model: {planning_model}")
            else:
                result["success"] = False
                result["error"] = f"Failed to set planning model: {planning_model}"
        
        if use_dual_model is not None:
            self.use_dual_model = use_dual_model
            result["changes"].append(f"use_dual_model: {use_dual_model}")
        
        result["current_config"] = {
            "vision_model": self.vision_model,
            "planning_model": self.planning_model,
            "use_dual_model": self.use_dual_model,
        }
        
        return result

# Global instance
ai_provider_manager = AIProviderManager()