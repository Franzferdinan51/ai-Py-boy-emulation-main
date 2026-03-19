"""
OpenAI Integration for PyBoy Game Agent

Provides multi-provider AI integration for game decision-making:
- OpenAIProvider: GPT-4o for vision, GPT-4 for reasoning, function calling
- AnthropicProvider: Claude 3 for vision and structured outputs
- LocalProvider: LM Studio and Ollama for local inference
- AgentRunner: Routes requests, handles retries, tracks costs
"""

import os
import json
import time
import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class AIResponse:
    """Response from an AI provider"""
    content: str
    provider: ProviderType
    model: str
    cost: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[dict] = None


@dataclass
class CostTracker:
    """Track API costs across providers"""
    total_cost: float = 0.0
    openai_cost: float = 0.0
    anthropic_cost: float = 0.0
    local_cost: float = 0.0
    requests_count: int = 0
    
    def add_cost(self, provider: ProviderType, cost: float):
        self.total_cost += cost
        self.requests_count += 1
        if provider == ProviderType.OPENAI:
            self.openai_cost += cost
        elif provider == ProviderType.ANTHROPIC:
            self.anthropic_cost += cost
        elif provider == ProviderType.LOCAL:
            self.local_cost += cost
    
    def get_summary(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 6),
            "openai_cost": round(self.openai_cost, 6),
            "anthropic_cost": round(self.anthropic_cost, 6),
            "local_cost": round(self.local_cost, 6),
            "requests": self.requests_count
        }


class BaseProvider(ABC):
    """Base class for all AI providers"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self._client = None
    
    @abstractmethod
    def chat(self, messages: list, model: str, **kwargs) -> AIResponse:
        """Send a chat request"""
        pass
    
    @abstractmethod
    def vision(self, image_data: bytes | str, prompt: str, model: str, **kwargs) -> AIResponse:
        """Analyze an image"""
        pass
    
    @abstractmethod
    def function_call(self, messages: list, functions: list, model: str, **kwargs) -> AIResponse:
        """Make a function call request"""
        pass
    
    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        pass


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider for GPT-4o, GPT-4, and function calling.
    
    Supports:
    - Vision analysis with GPT-4o
    - Text reasoning with GPT-4
    - Function calling for game decisions
    """
    
    # Pricing per 1M tokens (2024 rates)
    PRICING = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key)
        self.base_url = base_url
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self.base_url
            )
        except ImportError:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage"""
        pricing = self.PRICING.get(model, {"input": 10.0, "output": 30.0})
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def chat(self, messages: list, model: str = "gpt-4o", **kwargs) -> AIResponse:
        """Send a text chat request"""
        start_time = time.time()
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage
            
            cost = self._calculate_cost(
                model,
                tokens.prompt_tokens,
                tokens.completion_tokens
            )
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=cost,
                tokens_used=tokens.total_tokens,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    def vision(self, image_data: bytes | str, prompt: str, model: str = "gpt-4o", **kwargs) -> AIResponse:
        """
        Analyze an image with GPT-4o.
        
        Args:
            image_data: Base64 encoded image or URL
            prompt: Question about the image
            model: Model to use (default gpt-4o)
        """
        start_time = time.time()
        
        # Convert to base64 if it's bytes
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:image/png;base64,{image_b64}"
        else:
            image_url = image_data
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage
            
            cost = self._calculate_cost(
                model,
                tokens.prompt_tokens,
                tokens.completion_tokens
            )
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=cost,
                tokens_used=tokens.total_tokens,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"OpenAI vision error: {e}")
            raise
    
    def function_call(self, messages: list, functions: list, model: str = "gpt-4o", **kwargs) -> AIResponse:
        """
        Make a function call request.
        
        Args:
            messages: Chat messages
            functions: List of function definitions
            model: Model to use
        """
        start_time = time.time()
        
        # Convert function definitions to OpenAI format
        openai_functions = []
        for func in functions:
            openai_functions.append({
                "type": "function",
                "function": {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {})
                }
            })
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_functions,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract function call if present
            message = response.choices[0].message
            content = message.content or ""
            
            # Check if there's a tool call
            function_call_result = None
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_call_result = {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
                content = json.dumps(function_call_result)
            
            tokens = response.usage
            cost = self._calculate_cost(model, tokens.prompt_tokens, tokens.completion_tokens)
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=cost,
                tokens_used=tokens.total_tokens,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"OpenAI function call error: {e}")
            raise


class AnthropicProvider(BaseProvider):
    """
    Anthropic API provider for Claude 3.
    
    Supports:
    - Vision analysis with Claude 3
    - Structured outputs
    """
    
    # Pricing per 1M tokens (2024 rates)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.anthropic.com"):
        super().__init__(api_key)
        self.base_url = base_url
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                base_url=self.base_url
            )
        except ImportError:
            logger.warning("Anthropic package not installed. Install with: pip install anthropic")
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def chat(self, messages: list, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> AIResponse:
        """Send a text chat request"""
        start_time = time.time()
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # System message handled separately
                continue
            anthropic_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        system = kwargs.pop("system", None)
        
        try:
            response = self._client.messages.create(
                model=model,
                messages=anthropic_messages,
                system=system,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text
            tokens = response.usage
            
            cost = self._calculate_cost(
                model,
                tokens.input_tokens,
                tokens.output_tokens
            )
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=cost,
                tokens_used=tokens.input_tokens + tokens.output_tokens,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    def vision(self, image_data: bytes | str, prompt: str, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> AIResponse:
        """
        Analyze an image with Claude 3.
        
        Args:
            image_data: Base64 encoded image or URL
            prompt: Question about the image
            model: Model to use (default claude-3-5-sonnet)
        """
        start_time = time.time()
        
        # Convert to base64 if it's bytes
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            media_type = "image/png"
        else:
            # URL - fetch it
            image_b64 = None
            media_type = "image/jpeg"
        
        if image_b64:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        else:
            # Use URL
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        
        try:
            response = self._client.messages.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text
            tokens = response.usage
            
            cost = self._calculate_cost(
                model,
                tokens.input_tokens,
                tokens.output_tokens
            )
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=cost,
                tokens_used=tokens.input_tokens + tokens.output_tokens,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
            )
        except Exception as e:
            logger.error(f"Anthropic vision error: {e}")
            raise
    
    def function_call(self, messages: list, functions: list, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> AIResponse:
        """
        Make a structured output request with Claude.
        Uses JSON mode for structured responses.
        """
        # Claude doesn't have native function calling like OpenAI
        # Use JSON mode instead
        kwargs["response_format"] = {"type": "json_object"}
        
        # Add function definitions to system prompt
        functions_desc = json.dumps(functions, indent=2)
        system = kwargs.pop("system", "")
        system += f"\n\nYou must respond with valid JSON. Available functions:\n{functions_desc}"
        
        return self.chat(messages, model, system=system, **kwargs)


class LocalProvider(BaseProvider):
    """
    Local AI provider for LM Studio and Ollama.
    
    Supports:
    - LM Studio (OpenAI-compatible API)
    - Ollama (native API)
    - Local vision models
    """
    
    def __init__(
        self,
        provider: str = "lmstudio",  # "lmstudio" or "ollama"
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed"
    ):
        super().__init__(api_key, base_url)
        self.local_provider = provider
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize local client"""
        try:
            from openai import OpenAI
            
            # Adjust URL for Ollama
            if self.local_provider == "ollama":
                self.base_url = self.base_url.replace("/v1", "")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            logger.warning("OpenAI package required for local providers")
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Local models are free"""
        return 0.0
    
    def list_models(self) -> list[str]:
        """List available local models"""
        try:
            if self.local_provider == "lmstudio":
                response = self._client.models.list()
                return [m.id for m in response.data]
            elif self.local_provider == "ollama":
                import requests
                resp = requests.get(f"{self.base_url}/api/tags")
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def chat(self, messages: list, model: str = "qwen2.5-coder", **kwargs) -> AIResponse:
        """Send a text chat request to local model"""
        start_time = time.time()
        
        try:
            # LM Studio uses OpenAI format
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            
            tokens = response.usage if response.usage else None
            tokens_used = tokens.total_tokens if tokens else 0
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=0.0,  # Local is free
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            logger.error(f"Local chat error: {e}")
            raise
    
    def vision(self, image_data: bytes | str, prompt: str, model: str = "qwen2-vl", **kwargs) -> AIResponse:
        """
        Analyze an image with a local vision model.
        
        Supports:
        - LM Studio: qwen2-vl, llava, etc.
        - Ollama: llava, bakllava
        """
        start_time = time.time()
        
        # Convert to base64 if bytes
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:image/png;base64,{image_b64}"
        else:
            image_url = image_data
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage if response.usage else None
            tokens_used = tokens.total_tokens if tokens else 0
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=0.0,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            logger.error(f"Local vision error: {e}")
            raise
    
    def function_call(self, messages: list, functions: list, model: str = "qwen2.5-coder", **kwargs) -> AIResponse:
        """Make a function call to local model"""
        # Convert function definitions
        openai_functions = []
        for func in functions:
            openai_functions.append({
                "type": "function",
                "function": {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {})
                }
            })
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_functions,
                **kwargs
            )
            
            message = response.choices[0].message
            content = message.content or ""
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                content = json.dumps({
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })
            
            return AIResponse(
                content=content,
                provider=self.provider_type,
                model=model,
                cost=0.0,
                latency_ms=0.0,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else {}
            )
        except Exception as e:
            logger.error(f"Local function call error: {e}")
            raise


class AgentRunner:
    """
    Main agent runner that routes to appropriate providers.
    
    Features:
    - Automatic provider selection based on task type
    - Retry logic with exponential backoff
    - Cost tracking across all providers
    - Fallback to local when API fails
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        local_provider: str = "lmstudio",
        local_url: str = "http://localhost:1234/v1",
        default_model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        # Initialize providers
        self.openai = OpenAIProvider(api_key=openai_key)
        self.anthropic = AnthropicProvider(api_key=anthropic_key)
        self.local = LocalProvider(provider=local_provider, base_url=local_url)
        
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cost_tracker = CostTracker()
        
        # Current active provider
        self._provider: BaseProvider = self.openai
    
    def set_provider(self, provider: str):
        """Set the active provider"""
        if provider == "openai":
            self._provider = self.openai
        elif provider == "anthropic":
            self._provider = self.anthropic
        elif provider == "local":
            self._provider = self.local
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> AIResponse:
        """Execute with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
        
        raise last_error
    
    def think(self, prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> AIResponse:
        """
        Send a text reasoning request.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            model: Optional model override
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        model = model or self.default_model
        
        try:
            response = self._retry_with_backoff(self._provider.chat, messages, model)
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
        except Exception as e:
            # Fallback to local
            logger.warning(f"Provider failed, falling back to local: {e}")
            self.set_provider("local")
            response = self._provider.chat(messages, "qwen2.5-coder")
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
    
    def see(self, image_data: bytes | str, prompt: str, model: Optional[str] = None) -> AIResponse:
        """
        Analyze a game screenshot.
        
        Args:
            image_data: Screenshot bytes or path
            prompt: Question about the image
            model: Optional model override (default gpt-4o for OpenAI)
        """
        # Use vision-capable model
        model = model or self.default_model
        
        try:
            response = self._retry_with_backoff(self._provider.vision, image_data, prompt, model)
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
        except Exception as e:
            # Fallback to local vision
            logger.warning(f"Provider vision failed, falling back to local: {e}")
            self.set_provider("local")
            response = self._provider.vision(image_data, prompt, "qwen2-vl")
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
    
    def act(self, context: dict, actions: list, prompt: Optional[str] = None) -> AIResponse:
        """
        Make a decision using function calling.
        
        Args:
            context: Game state context
            actions: Available actions as function definitions
            prompt: Optional reasoning prompt
        """
        messages = [
            {"role": "system", "content": "You are a game AI. Choose the best action based on the game state."},
            {"role": "user", "content": prompt or f"Game state: {json.dumps(context)}. Choose an action."}
        ]
        
        try:
            response = self._retry_with_backoff(
                self._provider.function_call,
                messages,
                actions,
                self.default_model
            )
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
        except Exception as e:
            # Fallback to local
            logger.warning(f"Provider function call failed, falling back to local: {e}")
            self.set_provider("local")
            response = self._provider.function_call(messages, actions, "qwen2.5-coder")
            self.cost_tracker.add_cost(response.provider, response.cost)
            return response
    
    def get_cost_summary(self) -> dict:
        """Get cost tracking summary"""
        return self.cost_tracker.get_summary()


# Example game action functions for function calling
GAME_ACTIONS = [
    {
        "name": "move_north",
        "description": "Move the character north (up)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_south",
        "description": "Move the character south (down)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_east",
        "description": "Move the character east (right)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "move_west",
        "description": "Move the character west (left)",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "attack",
        "description": "Attack the current target",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target name or type"}
            }
        }
    },
    {
        "name": "use_item",
        "description": "Use an item from inventory",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item name"}
            }
        }
    },
    {
        "name": "wait",
        "description": "Wait/Rest for a turn",
        "parameters": {"type": "object", "properties": {}}
    }
]


# Convenience function for quick initialization
def create_agent(
    provider: str = "openai",
    api_key: Optional[str] = None
) -> AgentRunner:
    """
    Create an AgentRunner with common defaults.
    
    Args:
        provider: "openai", "anthropic", or "local"
        api_key: Optional API key (reads from env if not provided)
    
    Returns:
        Configured AgentRunner instance
    """
    if provider == "openai":
        return AgentRunner(openai_key=api_key, default_model="gpt-4o")
    elif provider == "anthropic":
        return AgentRunner(anthropic_key=api_key, default_model="claude-3-5-sonnet-20241022")
    elif provider == "local":
        return AgentRunner(local_provider="lmstudio")
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    # Demo usage
    print("OpenAI Integration Module")
    print("=" * 40)
    
    # Create agent (will use environment variables)
    # agent = create_agent("local")
    
    # Example: Analyze a game screenshot
    # response = agent.see(screenshot_bytes, "What Pokemon is on screen?")
    # print(response.content)
    
    # Example: Make a game decision
    # context = {"health": 50, "enemies": ["Rattata", "Pidgey"]}
    # response = agent.act(context, GAME_ACTIONS)
    # print(response.content)
    
    print("Import and use:")
    print("  from openai_integration import create_agent, AgentRunner")
    print("  agent = create_agent('local')")
    print("  response = agent.think('What should I do in Pokemon?')")