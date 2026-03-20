"""
OpenClaw Model Discovery Service

Discovers and surfaces available models from OpenClaw Gateway.
Provides real-time model lists instead of hardcoded defaults.

OpenClaw-Native Model Metadata Shape:
- id: Unique identifier (used in API calls)
- name: Short display name
- label: Full display name for dropdowns (includes category suffix)
- provider: Provider family
- category: 'vision', 'reasoning', or 'general'
- capabilities: Array of capabilities ['vision', 'reasoning', 'text']
- is_vision_capable: Boolean for quick filtering
- is_free: Boolean indicating free/unlimited usage
- manual_allowed: Boolean - can user enter custom model ID?
- is_default: Boolean - is this the default for this role?
- role: 'primary', 'vision', 'planning', 'fallback', or 'general'
- context_window: Estimated context window size
- description: Human-readable description
"""
import logging
import requests
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Type aliases for OpenClaw-native semantics
ModelCategory = Literal['vision', 'reasoning', 'general']
ModelRole = Literal['primary', 'vision', 'planning', 'fallback', 'general']


@dataclass
class ModelInfo:
    """
    Information about a model with OpenClaw-native metadata.
    
    This shape matches OpenClaw's model discovery contract and provides
    consistent metadata across all backend routes and frontend consumers.
    """
    id: str
    name: str
    provider: str
    label: str = ""
    category: ModelCategory = 'general'
    capabilities: List[str] = field(default_factory=list)
    is_vision_capable: bool = False
    is_free: bool = False
    manual_allowed: bool = True
    is_default: bool = False
    role: ModelRole = 'general'
    context_window: int = 0
    priority: int = 0
    description: str = ""
    
    def __post_init__(self):
        """Auto-generate label if not provided"""
        if not self.label:
            category_suffix = f" ({self.category.title()})" if self.category != 'general' else ""
            self.label = f"{self.name}{category_suffix}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'label': self.label,
            'provider': self.provider,
            'category': self.category,
            'capabilities': list(self.capabilities) if self.capabilities else ['text'],
            'is_vision_capable': self.is_vision_capable,
            'is_free': self.is_free,
            'manual_allowed': self.manual_allowed,
            'is_default': self.is_default,
            'role': self.role,
            'context_window': self.context_window,
            'priority': self.priority,
            'description': self.description or f"AI model: {self.name}"
        }


@dataclass
class ProviderInfo:
    """Information about a model provider with OpenClaw-native metadata"""
    id: str
    name: str
    status: str  # 'available', 'unavailable', 'degraded'
    available: bool = True
    manual_allowed: bool = True
    priority: int = 99
    models_count: int = 0
    error: Optional[str] = None
    default_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'available': self.available,
            'manual_allowed': self.manual_allowed,
            'priority': self.priority,
            'models_count': self.models_count,
            'error': self.error,
            'default_model': self.default_model
        }


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
        """
        Parse raw OpenClaw model data into ModelInfo objects with full metadata.
        
        Handles both OpenClaw Gateway format and standard model list format.
        """
        models = []
        
        for i, model_data in enumerate(raw_models):
            try:
                model_id = model_data.get('id', '')
                if not model_id:
                    continue
                
                # Determine capabilities from model data or infer from name
                capabilities = model_data.get('capabilities', [])
                is_vision = model_data.get('is_vision_capable', False)
                
                # Infer capabilities if not provided
                if not capabilities:
                    capabilities = ['text']
                    if is_vision or self._is_vision_model(model_id):
                        capabilities.extend(['vision'])
                        is_vision = True
                    if 'reason' in model_id.lower() or 'think' in model_id.lower():
                        capabilities.append('reasoning')
                
                # Determine category based on capabilities
                if is_vision:
                    category: ModelCategory = 'vision'
                elif 'reasoning' in capabilities:
                    category = 'reasoning'
                else:
                    category = 'general'
                
                # Determine role based on priority and capabilities
                priority = model_data.get('priority', 50)
                if priority >= 90:
                    role: ModelRole = 'vision' if is_vision else 'primary'
                elif priority >= 70:
                    role = 'planning' if not is_vision else 'fallback'
                else:
                    role = 'fallback'
                
                # Extract provider
                provider = self._extract_provider(model_id)
                
                # Build display name
                name = model_data.get('name', model_id.split('/')[-1])
                category_suffix = f" ({category.title()})" if category != 'general' else ""
                label = f"{name}{category_suffix}"
                
                model = ModelInfo(
                    id=model_id,
                    name=name,
                    label=label,
                    provider=provider,
                    category=category,
                    capabilities=list(capabilities),
                    is_vision_capable=is_vision,
                    is_free=model_data.get('is_free', False),
                    manual_allowed=model_data.get('manual_allowed', True),
                    is_default=model_data.get('is_default', i == 0),  # First is default
                    role=role,
                    context_window=model_data.get('context_window', 4096),
                    priority=priority,
                    description=model_data.get('description', f"AI model: {name}")
                )
                models.append(model)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse model data: {e}")
        
        # Ensure at least one model is marked as default per category
        self._assign_defaults(models)
        
        return models
    
    def _assign_defaults(self, models: List[ModelInfo]) -> None:
        """Assign default flags to best models in each category"""
        categories = {'vision': [], 'reasoning': [], 'general': []}
        
        for model in models:
            categories[model.category].append(model)
        
        for category_models in categories.values():
            if category_models:
                # Sort by priority and mark first as default
                category_models.sort(key=lambda m: m.priority, reverse=True)
                category_models[0].is_default = True
                category_models[0].role = 'vision' if category_models[0].is_vision_capable else 'primary'
    
    def _extract_models_from_session(self, session_data: Dict) -> List[ModelInfo]:
        """
        Extract model info from session status data with full OpenClaw metadata.
        """
        models = []
        
        # Extract current model configuration
        config = session_data.get('config', {})
        default_model = config.get('default_model')
        current_model = config.get('model')
        
        if default_model:
            is_vision = self._is_vision_model(default_model)
            category = 'vision' if is_vision else 'reasoning'
            name = default_model.split('/')[-1]
            
            models.append(ModelInfo(
                id=default_model,
                name=name,
                label=f"{name} ({category.title()})",
                provider=self._extract_provider(default_model),
                category=category,
                capabilities=['vision', 'reasoning', 'text'] if is_vision else ['reasoning', 'text'],
                is_vision_capable=is_vision,
                is_free=False,  # Unknown from session
                manual_allowed=True,
                is_default=True,
                role='vision' if is_vision else 'primary',
                context_window=4096,
                priority=100  # Current model gets high priority
            ))
        
        if current_model and current_model != default_model:
            is_vision = self._is_vision_model(current_model)
            category = 'vision' if is_vision else 'reasoning'
            name = current_model.split('/')[-1]
            
            models.append(ModelInfo(
                id=current_model,
                name=name,
                label=f"{name} ({category.title()})",
                provider=self._extract_provider(current_model),
                category=category,
                capabilities=['vision', 'reasoning', 'text'] if is_vision else ['reasoning', 'text'],
                is_vision_capable=is_vision,
                is_free=False,
                manual_allowed=True,
                is_default=False,
                role='fallback',
                context_window=4096,
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
        """
        Return fallback models when OpenClaw is unavailable.
        
        These models use the full OpenClaw-native metadata shape with
        role assignments, capability flags, and default semantics.
        """
        return [
            # Primary Vision Model (FREE)
            ModelInfo(
                id="bailian/kimi-k2.5",
                name="Kimi K2.5",
                label="Kimi K2.5 (Vision)",
                provider="bailian",
                category="vision",
                capabilities=["vision", "reasoning", "text"],
                is_vision_capable=True,
                is_free=True,
                manual_allowed=True,
                is_default=True,
                role="vision",
                context_window=196608,
                priority=100,
                description="Best for game screen analysis (FREE)"
            ),
            # Primary Planning Model
            ModelInfo(
                id="bailian/glm-5",
                name="GLM-5",
                label="GLM-5 (Reasoning)",
                provider="bailian",
                category="reasoning",
                capabilities=["reasoning", "text"],
                is_vision_capable=False,
                is_free=False,
                manual_allowed=True,
                is_default=True,
                role="planning",
                context_window=128000,
                priority=95,
                description="Fast decisions, great for games"
            ),
            # High-quality vision alternative
            ModelInfo(
                id="bailian/qwen3.5-plus",
                name="Qwen 3.5 Plus",
                label="Qwen 3.5 Plus (Vision)",
                provider="bailian",
                category="vision",
                capabilities=["vision", "reasoning", "text"],
                is_vision_capable=True,
                is_free=False,
                manual_allowed=True,
                is_default=False,
                role="fallback",
                context_window=1000000,
                priority=85,
                description="Best reasoning (quota)"
            ),
            # Free unlimited planning model
            ModelInfo(
                id="bailian/MiniMax-M2.5",
                name="MiniMax M2.5",
                label="MiniMax M2.5 (FREE)",
                provider="bailian",
                category="reasoning",
                capabilities=["reasoning", "text"],
                is_vision_capable=False,
                is_free=True,
                manual_allowed=True,
                is_default=False,
                role="fallback",
                context_window=196608,
                priority=80,
                description="Unlimited, reliable (FREE)"
            ),
        ]
    
    def get_vision_models(self) -> List[ModelInfo]:
        """Get only vision-capable models sorted by priority"""
        all_models = self.get_available_models()
        vision_models = [m for m in all_models if m.is_vision_capable]
        return sorted(vision_models, key=lambda m: m.priority, reverse=True)
    
    def get_planning_models(self) -> List[ModelInfo]:
        """Get models suitable for planning/decision making sorted by priority"""
        all_models = self.get_available_models()
        # All models can do planning, prioritize by role
        planning_models = [m for m in all_models if 'text' in m.capabilities or 'reasoning' in m.capabilities]
        return sorted(planning_models, key=lambda m: m.priority, reverse=True)
    
    def get_models_by_role(self, role: ModelRole) -> List[ModelInfo]:
        """
        Get models by their role assignment.
        
        Args:
            role: 'primary', 'vision', 'planning', 'fallback', or 'general'
            
        Returns:
            List of models with the specified role, sorted by priority
        """
        all_models = self.get_available_models()
        role_models = [m for m in all_models if m.role == role]
        return sorted(role_models, key=lambda m: m.priority, reverse=True)
    
    def get_default_model(self, category: ModelCategory = 'vision') -> Optional[ModelInfo]:
        """
        Get the default model for a specific category.
        
        Args:
            category: 'vision', 'reasoning', or 'general'
            
        Returns:
            Default ModelInfo for the category or None
        """
        all_models = self.get_available_models()
        defaults = [m for m in all_models if m.is_default and m.category == category]
        return defaults[0] if defaults else None
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Get OpenClaw-native runtime configuration.
        
        Returns a unified config shape for the settings UI with:
        - Default vision model
        - Default planning model
        - All available models with full metadata
        - Provider status
        """
        all_models = self.get_available_models()
        vision_models = self.get_vision_models()
        planning_models = self.get_planning_models()
        
        # Get defaults
        default_vision = next((m for m in vision_models if m.is_default), vision_models[0] if vision_models else None)
        default_planning = next((m for m in planning_models if m.is_default), planning_models[0] if planning_models else None)
        
        return {
            'defaults': {
                'vision_model': default_vision.to_dict() if default_vision else None,
                'planning_model': default_planning.to_dict() if default_planning else None,
            },
            'models': {
                'all': [m.to_dict() for m in all_models],
                'vision': [m.to_dict() for m in vision_models],
                'planning': [m.to_dict() for m in planning_models],
            },
            'counts': {
                'total': len(all_models),
                'vision': len(vision_models),
                'planning': len(planning_models),
                'free': len([m for m in all_models if m.is_free]),
            },
            'timestamp': datetime.now().isoformat(),
            'cached': self._is_cache_valid()
        }
    
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
