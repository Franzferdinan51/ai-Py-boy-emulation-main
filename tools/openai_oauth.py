"""
OpenAI OAuth Integration
Uses ChatGPT Plus subscription for API access
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class OpenAIConfig:
    """OpenAI OAuth configuration"""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:5000/callback"
    auth_url: str = "https://auth.openai.com/oauth/authorize"
    token_url: str = "https://auth.openai.com/oauth/token"
    api_base: str = "https://api.openai.com/v1"
    
    # ChatGPT Plus uses these scopes
    scopes: list = None
    
    def __post_init__(self):
        self.scopes = [
            "openai.api",
            "model.access",
            "assistant.api"
        ]

class OpenAIOAuth:
    """
    OpenAI OAuth handler for ChatGPT Plus
    Handles token refresh and API calls
    """
    
    def __init__(self, config: OpenAIConfig = None):
        self.config = config or OpenAIConfig()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_file = ".openai_token.json"
        
    def get_auth_url(self) -> str:
        """Generate OAuth authorization URL"""
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "response_type": "code",
        }
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.config.auth_url}?{query}"
    
    def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.config.redirect_uri,
        }
        
        response = requests.post(self.config.token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self._save_token(token_data)
        return token_data
    
    def refresh_access_token(self) -> Dict:
        """Refresh the access token"""
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        data = {
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }
        
        response = requests.post(self.config.token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self._save_token(token_data)
        return token_data
    
    def _save_token(self, token_data: Dict):
        """Save token to file"""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        
        with open(self.token_file, "w") as f:
            json.dump(token_data, f)
    
    def load_token(self) -> bool:
        """Load token from file"""
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                token_data = json.load(f)
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            return True
        return False
    
    def make_request(self, endpoint: str, method: str = "GET", 
                    data: Dict = None, files: Dict = None) -> Dict:
        """Make authenticated API request"""
        if not self.access_token:
            if not self.load_token():
                raise ValueError("Not authenticated - call exchange_code_for_token first")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.config.api_base}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                del headers["Content-Type"]
                response = requests.post(url, headers=headers, data=data, files=files)
            else:
                response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 401:
            # Token expired, try refresh
            self.refresh_access_token()
            headers["Authorization"] = f"Bearer {self.access_token}"
            response = requests.post(url, headers=headers, json=data)
        
        response.raise_for_status()
        return response.json()

# =============================================================================
# CHATGPT PLUS API CLIENT
# =============================================================================

class ChatGPTPlusClient:
    """
    Client for ChatGPT Plus subscription
    Provides vision, text, and function calling
    """
    
    def __init__(self, oauth: OpenAIOAuth = None):
        self.oauth = oauth or OpenAIOAuth()
        
    def analyze_image(self, image_url: str, prompt: str) -> str:
        """Analyze image using GPT-4V"""
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        result = self.oauth.make_request("chat/completions", "POST", data)
        return result["choices"][0]["message"]["content"]
    
    def text_completion(self, prompt: str, model: str = "gpt-4o") -> str:
        """Text completion"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        result = self.oauth.make_request("chat/completions", "POST", data)
        return result["choices"][0]["message"]["content"]
    
    def function_call(self, prompt: str, functions: list, 
                     function_call: str = "auto") -> Dict:
        """Function calling"""
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "function", "function": f} for f in functions],
            "tool_choice": function_call,
            "max_tokens": 1000
        }
        
        result = self.oauth.make_request("chat/completions", "POST", data)
        return result

if __name__ == "__main__":
    # Example usage
    client = ChatGPTPlusClient()
    print("ChatGPT Plus Client initialized")
    print(f"Auth URL: {client.oauth.get_auth_url()}")
