"""
AI Provider Manager - Handles automatic provider detection and fallback
"""
import os
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from .ai_api_base import AIAPIConnector
from .gemini_api import GeminiAPIConnector
from .openrouter_api import OpenRouterAPIConnector
from .openai_compatible import OpenAICompatibleConnector
from .nvidia_api import NVIDIAAPIConnector
from .mock_ai_provider import MockAIProvider
from .tetris_genetic_ai import TetrisGeneticAI

class ProviderStatus(Enum):
    """Provider status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"

class AIProviderManager:
    """Manages AI providers with automatic detection and fallback"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.provider_order: List[str] = []
        self.fallback_providers: List[str] = []
        self.last_refresh_time = 0
        self.refresh_interval = 300  # Refresh every 5 minutes
        self.default_provider = os.environ.get('DEFAULT_AI_PROVIDER', 'mock')
        self.initialize_providers()

    def initialize_providers(self):
        """Initialize all available AI providers"""
        self.logger.info("Initializing AI providers...")

        # Define provider configurations
        provider_configs = [
            {
                'name': 'gemini',
                'env_key': 'GEMINI_API_KEY',
                'class': GeminiAPIConnector,
                'priority': 1
            },
            {
                'name': 'openrouter',
                'env_key': 'OPENROUTER_API_KEY',
                'class': OpenRouterAPIConnector,
                'priority': 2
            },
            {
                'name': 'openai-compatible',
                'env_key': 'OPENAI_API_KEY',
                'class': OpenAICompatibleConnector,
                'priority': 3,
                'extra_params': {
                    'base_url': os.environ.get('OPENAI_ENDPOINT')
                }
            },
            {
                'name': 'nvidia',
                'env_key': 'NVIDIA_API_KEY',
                'class': NVIDIAAPIConnector,
                'priority': 4
            },
            {
                'name': 'mock',
                'env_key': None,  # No API key required
                'class': MockAIProvider,
                'priority': 1 if self.default_provider == 'mock' else 99,  # High priority if set as default
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
            # Special handling for mock provider - no API key needed
            if name == 'mock':
                api_key = "mock-key"
            elif name == 'tetris-genetic':
                api_key = None  # No API key required
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
            # Mock provider is always available
            if provider_name == 'mock':
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

        # Clear all provider references
        self.providers.clear()
        self.provider_order.clear()
        self.fallback_providers.clear()
        self.logger.info("AI provider manager cleanup complete")

# Global instance
ai_provider_manager = AIProviderManager()