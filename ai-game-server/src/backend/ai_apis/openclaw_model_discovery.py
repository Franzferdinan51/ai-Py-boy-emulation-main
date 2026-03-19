"""
OpenClaw Model Discovery Service

Discovers and surfaces available models from OpenClaw Gateway.
Provides real-time model lists instead of hardcoded defaults.
"""
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model"""
    id: str
    name: str
    provider: str
    capabilities: List[str] = field(default_factory=list)
    context_window: int = 0
    is_vision_capable: bool = False
    is_free: bool = False
    description: str = ""
    priority: int = 0  # For sorting recommendations


@dataclass
class ProviderInfo:
    """Information about a model provider"""
    id: str
    name: str
    status: str  # 'available', 'unavailable', 'degraded'
    models_count: int = 0
    error: Optional[str] = None


class OpenClawModelDiscovery:
    """
    Discovers models from OpenClaw Gateway.
    
    Features:
    - Real-time model discovery from OpenClaw
    - Caching to reduce API calls
    - Model categorization (vision vs planning)
    - Provider health status
    """
    
    def __init__(self, openclaw_endpoint: str = "http://localhost:18789"):
        self.openclaw_endpoint = openclaw_endpoint.rstrip('/')
        self.cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if not self.cache_timestamp:
            return False
        return datetime.now() - self.cache_timestamp < self.cache_ttl
    
    def get_available_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        Get list of available models from OpenClaw.
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            List of ModelInfo objects
        """
        if not force_refresh and self._is_cache_valid():
            self.logger.debug("Using cached model list")
            return self.cache.get('models', [])
        
        try:
            # Query OpenClaw for models
            models = self._fetch_models_from_openclaw()
            
            # Cache the results
            self.cache['models'] = models
            self.cache_timestamp = datetime.now()
            
            self.logger.info(f"Discovered {len(models)} models from OpenClaw")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to fetch models: {e}")
            # Return fallback models if OpenClaw is unavailable
            return self._get_fallback_models()
    
    def _fetch_models_from_openclaw(self) -> List[ModelInfo]:
        """Fetch models directly from OpenClaw Gateway"""
        models = []
        
        try:
            # Try to get models via OpenClaw session status or models endpoint
            response = requests.get(
                f"{self.openclaw_endpoint}/api/models",
                timeout=5
            )
            
            if response.ok:
                data = response.json()
                models = self._parse_openclaw_models(data.get('models', []))
                
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"OpenClaw models endpoint not available: {e}")
            # Try alternative endpoint
            models = self._fetch_models_from_session_status()
        
        return models
    
    def _fetch_models_from_session_status(self) -> List[ModelInfo]:
        """Fallback: Get model info from session status"""
        try:
            response = requests.get(
                f"{self.openclaw_endpoint}/api/session/status",
                timeout=5
            )
            
            if response.ok:
                data = response.json()
                # Extract model information from session config
                return self._extract_models_from_session(data)
                
        except Exception as e:
            self.logger.debug(f"Session status endpoint not available: {e}")
        
        return []
    
    def _parse_openclaw_models(self, raw_models: List[Dict]) -> List[ModelInfo]:
        """Parse raw OpenClaw model data into ModelInfo objects"""
        models = []
        
        for model_data in raw_models:
            try:
                model_id = model_data.get('id', '')
                if not model_id:
                    continue
                
                # Determine capabilities
                capabilities = model_data.get('capabilities', [])
                is_vision = 'vision' in capabilities or 'multimodal' in capabilities
                
                # Categorize by provider
                provider = self._extract_provider(model_id)
                
                model = ModelInfo(
                    id=model_id,
                    name=model_data.get('name', model_id.split('/')[-1]),
                    provider=provider,
                    capabilities=capabilities,
                    context_window=model_data.get('context_window', 0),
                    is_vision_capable=is_vision,
                    is_free=model_data.get('is_free', False),
                    description=model_data.get('description', ''),
                    priority=model_data.get('priority', 0)
                )
                models.append(model)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse model data: {e}")
        
        return models
    
    def _extract_models_from_session(self, session_data: Dict) -> List[ModelInfo]:
        """Extract model info from session status data"""
        models = []
        
        # Extract current model configuration
        config = session_data.get('config', {})
        default_model = config.get('default_model')
        current_model = config.get('model')
        
        if default_model:
            models.append(ModelInfo(
                id=default_model,
                name=default_model.split('/')[-1],
                provider=self._extract_provider(default_model),
                is_vision_capable=self._is_vision_model(default_model),
                priority=100  # Current model gets high priority
            ))
        
        if current_model and current_model != default_model:
            models.append(ModelInfo(
                id=current_model,
                name=current_model.split('/')[-1],
                provider=self._extract_provider(current_model),
                is_vision_capable=self._is_vision_model(current_model),
                priority=90
            ))
        
        return models
    
    def _extract_provider(self, model_id: str) -> str:
        """Extract provider name from model ID"""
        if '/' in model_id:
            return model_id.split('/')[0]
        return 'unknown'
    
    def _is_vision_model(self, model_id: str) -> bool:
        """Check if model is vision-capable based on known patterns"""
        vision_patterns = [
            'vl', 'vision', 'kimi', 'qwen-vl', 'gemini',
            'gpt-4v', 'gpt-4o', 'llava'
        ]
        model_lower = model_id.lower()
        return any(pattern in model_lower for pattern in vision_patterns)
    
    def _get_fallback_models(self) -> List[ModelInfo]:
        """Return fallback models when OpenClaw is unavailable"""
        return [
            ModelInfo(
                id="bailian/kimi-k2.5",
                name="Kimi K2.5",
                provider="bailian",
                capabilities=["vision", "multimodal"],
                is_vision_capable=True,
                is_free=True,
                description="Best for game screen analysis (FREE)",
                priority=100
            ),
            ModelInfo(
                id="bailian/glm-5",
                name="GLM-5",
                provider="bailian",
                capabilities=["text", "reasoning"],
                is_vision_capable=False,
                description="Fast decisions, great for games",
                priority=90
            ),
            ModelInfo(
                id="bailian/qwen3.5-plus",
                name="Qwen 3.5 Plus",
                provider="bailian",
                capabilities=["text", "reasoning", "vision"],
                is_vision_capable=True,
                description="Best reasoning (quota)",
                priority=80
            ),
            ModelInfo(
                id="bailian/MiniMax-M2.5",
                name="MiniMax M2.5",
                provider="bailian",
                capabilities=["text", "reasoning"],
                is_vision_capable=False,
                is_free=True,
                description="Unlimited, reliable (FREE)",
                priority=70
            ),
        ]
    
    def get_vision_models(self) -> List[ModelInfo]:
        """Get only vision-capable models"""
        all_models = self.get_available_models()
        return [m for m in all_models if m.is_vision_capable]
    
    def get_planning_models(self) -> List[ModelInfo]:
        """Get models suitable for planning/decision making"""
        all_models = self.get_available_models()
        # All models can do planning, but exclude pure vision models
        return [m for m in all_models if 'text' in m.capabilities or 'reasoning' in m.capabilities]
    
    def get_provider_status(self) -> List[ProviderInfo]:
        """Get status of all model providers"""
        providers = {}
        
        try:
            response = requests.get(
                f"{self.openclaw_endpoint}/api/providers/status",
                timeout=5
            )
            
            if response.ok:
                data = response.json()
                for provider_id, status_data in data.items():
                    providers[provider_id] = ProviderInfo(
                        id=provider_id,
                        name=status_data.get('name', provider_id),
                        status=status_data.get('status', 'unknown'),
                        models_count=status_data.get('models_count', 0),
                        error=status_data.get('error')
                    )
                    
        except Exception as e:
            self.logger.warning(f"Failed to get provider status: {e}")
        
        return list(providers.values())
    
    def recommend_model(self, use_case: str) -> Optional[ModelInfo]:
        """
        Recommend a model for a specific use case.
        
        Args:
            use_case: 'vision', 'planning', 'fast', 'quality', 'free'
            
        Returns:
            Recommended ModelInfo or None
        """
        models = self.get_available_models()
        
        if not models:
            return None
        
        if use_case == 'vision':
            # Best vision model
            vision_models = [m for m in models if m.is_vision_capable]
            return max(vision_models, key=lambda m: m.priority) if vision_models else None
            
        elif use_case == 'planning':
            # Best reasoning model
            planning_models = self.get_planning_models()
            return max(planning_models, key=lambda m: m.priority) if planning_models else None
            
        elif use_case == 'fast':
            # Fastest model (usually smaller context, lower priority)
            return min(models, key=lambda m: m.context_window or 999999)
            
        elif use_case == 'quality':
            # Highest quality model
            return max(models, key=lambda m: m.priority)
            
        elif use_case == 'free':
            # Best free model
            free_models = [m for m in models if m.is_free]
            return max(free_models, key=lambda m: m.priority) if free_models else None
        
        return None
    
    def clear_cache(self):
        """Clear the model cache"""
        self.cache = {}
        self.cache_timestamp = None
        self.logger.info("Model cache cleared")


# Singleton instance for reuse
model_discovery: Optional[OpenClawModelDiscovery] = None


def get_model_discovery(openclaw_endpoint: str = "http://localhost:18789") -> OpenClawModelDiscovery:
    """Get or create model discovery singleton"""
    global model_discovery
    
    if model_discovery is None:
        model_discovery = OpenClawModelDiscovery(openclaw_endpoint)
    
    return model_discovery
