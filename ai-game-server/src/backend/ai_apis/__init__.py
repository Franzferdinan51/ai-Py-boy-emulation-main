"""
AI Game Server - AI APIs package

Provides AI provider abstractions for game decision making.

Architecture:
- ai_api_base.py: Abstract base class for all providers
- ai_provider_manager.py: Manages provider selection and fallback
- dual_model_provider.py: Dual-model architecture (vision + planning)
- openclaw_ai_provider.py: OpenClaw Gateway integration
- openai_compatible.py: OpenAI-compatible endpoints (LM Studio, Ollama, etc.)
- openrouter_api.py: OpenRouter API connector
- mock_ai_provider.py: Mock provider for testing
- tetris_genetic_ai.py: Specialized Tetris AI

Dual-Model Architecture (NEW):
- Vision Model: Analyzes screenshots, extracts game state
- Planning Model: Makes decisions based on vision analysis
- Flow: Screenshot -> Vision -> Planning -> Action
"""

from .ai_api_base import AIAPIConnector
from .ai_provider_manager import AIProviderManager, ai_provider_manager
from .dual_model_provider import (
    DualModelProvider,
    VisionModelType,
    PlanningModelType,
    VisionAnalysis,
    PlanningResult,
)

__all__ = [
    'AIAPIConnector',
    'AIProviderManager',
    'ai_provider_manager',
    'DualModelProvider',
    'VisionModelType',
    'PlanningModelType',
    'VisionAnalysis',
    'PlanningResult',
]