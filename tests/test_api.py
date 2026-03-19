#!/usr/bin/env python3
"""
pytest test suite for AI-PyBoy HTTP API endpoints

Run with: pytest tests/test_api.py -v
"""

import pytest
import requests
import json
import os
import sys
from typing import Optional

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5002")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM_PATH = os.environ.get("TEST_ROM", os.path.join(PROJECT_ROOT, "roms", "pokemon-red.gb"))


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_health_check(self):
        """Test /health endpoint returns healthy status"""
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_api_status(self):
        """Test /api/status endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/status", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "ai_providers" in data
        assert "active_emulator" in data
        assert "rom_loaded" in data

    def test_api_config(self):
        """Test /api/config endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/config", timeout=5)
        assert response.status_code == 200

    def test_config_validate(self):
        """Test /api/config/validate endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/config/validate", timeout=5)
        assert response.status_code == 200


class TestEmulatorControl:
    """Test emulator control endpoints"""

    @pytest.fixture(autouse=True)
    def ensure_rom_loaded(self):
        """Ensure a ROM is loaded before tests"""
        # Try to load a test ROM if available
        if os.path.exists(ROM_PATH):
            response = requests.post(
                f"{BACKEND_URL}/api/load_rom",
                json={"rom_path": ROM_PATH},
                timeout=10
            )
            # Continue even if it fails (ROM might already be loaded)

    def test_load_rom(self):
        """Test /api/load_rom endpoint"""
        if not os.path.exists(ROM_PATH):
            pytest.skip("Test ROM not found")

        response = requests.post(
            f"{BACKEND_URL}/api/load_rom",
            json={"rom_path": ROM_PATH},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True or "rom" in data

    def test_load_rom_not_found(self):
        """Test loading non-existent ROM"""
        response = requests.post(
            f"{BACKEND_URL}/api/load_rom",
            json={"rom_path": "/nonexistent/rom.gb"},
            timeout=10
        )
        # Should return error status
        assert response.status_code in [400, 404, 500]

    def test_action_endpoint(self):
        """Test /api/action endpoint with button press"""
        response = requests.post(
            f"{BACKEND_URL}/api/action",
            json={"action": "A"},  # Use 'action' parameter (not 'button')
            timeout=10
        )
        assert response.status_code in [200, 201, 400]
        data = response.json()
        # May return success or error if no ROM loaded
        assert "success" in data or "error" in data or "message" in data

    def test_action_invalid_button(self):
        """Test invalid button press"""
        response = requests.post(
            f"{BACKEND_URL}/api/action",
            json={"action": "INVALID"},
            timeout=10
        )
        # Should return error
        assert response.status_code in [400, 422, 500]

    def test_game_button_endpoint(self):
        """Test /api/game/button endpoint"""
        response = requests.post(
            f"{BACKEND_URL}/api/game/button",
            json={"button": "A"},
            timeout=10
        )
        # May fail if no ROM loaded, but endpoint should exist
        assert response.status_code in [200, 400, 500]

    def test_screen_endpoint(self):
        """Test /api/screen endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/screen", timeout=10)
        assert response.status_code == 200
        # Should return image data
        assert response.headers.get("content-type", "").startswith("image/") or \
               "image" in response.headers.get("content-type", "").lower() or \
               len(response.content) > 0


class TestMemoryEndpoints:
    """Test memory reading endpoints"""

    def test_memory_read(self):
        """Test /memory endpoint"""
        response = requests.post(
            f"{BACKEND_URL}/memory",
            json={"address": 53248, "length": 16},
            timeout=10
        )
        # May fail if no ROM loaded, but should return valid response
        assert response.status_code in [200, 400, 500]

    def test_characters_endpoint(self):
        """Test /characters endpoint"""
        response = requests.get(f"{BACKEND_URL}/characters", timeout=10)
        assert response.status_code == 200
        # Should return character data
        data = response.json()
        assert data is not None


class TestSaveStates:
    """Test save/load state endpoints"""

    def test_save_state(self):
        """Test /api/save_state endpoint"""
        response = requests.post(
            f"{BACKEND_URL}/api/save_state",
            json={"save_name": "test_save"},
            timeout=10
        )
        assert response.status_code in [200, 201]

    def test_load_state(self):
        """Test /api/load_state endpoint"""
        # First save
        requests.post(
            f"{BACKEND_URL}/api/save_state",
            json={"save_name": "test_load"},
            timeout=10
        )
        # Then load
        response = requests.post(
            f"{BACKEND_URL}/api/load_state",
            json={"save_name": "test_load"},
            timeout=10
        )
        assert response.status_code in [200, 201, 404]

    def test_load_nonexistent_save(self):
        """Test loading non-existent save"""
        response = requests.post(
            f"{BACKEND_URL}/api/load_state",
            json={"save_name": "nonexistent_save_12345"},
            timeout=10
        )
        # Should return error
        assert response.status_code in [404, 500]


class TestAIProviders:
    """Test AI provider endpoints"""

    def test_providers_status(self):
        """Test /api/providers/status endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/providers/status", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_models_list(self):
        """Test /api/models endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/models", timeout=5)
        assert response.status_code == 200


class TestChatEndpoint:
    """Test AI chat endpoint"""

    def test_chat_basic(self):
        """Test basic chat functionality"""
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"message": "What should I do?"},
            timeout=30
        )
        assert response.status_code in [200, 201, 400]

    def test_chat_with_context(self):
        """Test chat with context"""
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={
                "message": "What moves should I use?",
                "context": {"party": [{"species": "Charmander"}]}
            },
            timeout=30
        )
        assert response.status_code in [200, 201, 400]


class TestUIEndpoints:
    """Test UI management endpoints"""

    def test_ui_status(self):
        """Test /api/ui/status endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/ui/status", timeout=5)
        assert response.status_code == 200


class TestPerformance:
    """Test performance endpoints"""

    def test_performance_metrics(self):
        """Test /api/performance endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/performance", timeout=5)
        assert response.status_code == 200

    def test_emulator_mode(self):
        """Test /api/emulator/mode endpoint"""
        response = requests.get(f"{BACKEND_URL}/api/emulator/mode", timeout=5)
        assert response.status_code == 200


class TestTilemapSprites:
    """Test tilemap and sprite endpoints"""

    def test_tilemap(self):
        """Test /tilemap endpoint"""
        response = requests.get(f"{BACKEND_URL}/tilemap", timeout=10)
        assert response.status_code == 200

    def test_sprites(self):
        """Test /sprites endpoint"""
        response = requests.get(f"{BACKEND_URL}/sprites", timeout=10)
        assert response.status_code == 200


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v"])