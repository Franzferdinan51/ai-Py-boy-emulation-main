"""
pytest configuration for AI-PyBoy tests
"""

import pytest
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai-game-server"))


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require running backend)"
    )
    config.addinivalue_line(
        "markers", "mcp: marks tests as MCP server tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as HTTP API tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def backend_url():
    """Return backend URL from environment or default"""
    return os.environ.get("BACKEND_URL", "http://localhost:5002")


@pytest.fixture(scope="session")
def test_rom_path():
    """Return path to test ROM"""
    rom_path = os.environ.get("TEST_ROM", os.path.join(PROJECT_ROOT, "roms", "pokemon-red.gb"))
    if os.path.exists(rom_path):
        return rom_path
    return None


def pytest_collection_modifyitems(config, items):
    """Add markers based on test names"""
    for item in items:
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_mcp" in item.nodeid:
            item.add_marker(pytest.mark.mcp)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)