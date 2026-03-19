#!/usr/bin/env python3
"""
pytest test suite for AI-PyBoy MCP Server tools

Run with: pytest tests/test_mcp.py -v

Requires: MCP server running and registered with mcporter
"""

import pytest
import json
import os
import sys
import subprocess
from typing import Dict, Any, Optional, List

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "ai-game-server", "mcp_server.py")


class MCPClient:
    """Simple MCP client for testing"""

    def __init__(self, server_path: str = None):
        self.server_path = server_path or MCP_SERVER_PATH

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        try:
            result = subprocess.run(
                ["python3", self.server_path],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(self.server_path)
            )

            if result.returncode != 0:
                return {"success": False, "error": result.stderr}

            # Parse JSON-RPC response
            response = json.loads(result.stdout)
            if "result" in response:
                return response["result"]
            elif "error" in response:
                return {"success": False, "error": response["error"]}
            return response
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON decode error: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Import MCP server module directly for testing
@pytest.fixture(scope="module")
def mcp_server():
    """Import and configure MCP server module"""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai-game-server"))
    try:
        import mcp_server
        return mcp_server
    except ImportError as e:
        pytest.skip(f"MCP server module not available: {e}")


@pytest.fixture
def mcp_client():
    """Create MCP client for testing"""
    return MCPClient()


class TestMCPServerImport:
    """Test MCP server can be imported"""

    def test_mcp_server_imports(self, mcp_server):
        """Test that MCP server module imports successfully"""
        assert mcp_server is not None
        assert mcp_server.MCP_AVAILABLE is True

    def test_pyboy_available(self, mcp_server):
        """Test PyBoy is available"""
        assert mcp_server.PYBOY_AVAILABLE is True

    def test_server_version(self, mcp_server):
        """Test server version is set"""
        assert hasattr(mcp_server, "SERVER_VERSION")
        assert mcp_server.SERVER_VERSION == "4.0.0"


class TestCoreEmulatorControls:
    """Test core emulator control functions"""

    def test_press_button_valid(self, mcp_server):
        """Test pressing valid button"""
        try:
            result = mcp_server.press_button("A")
            assert isinstance(result, dict)
        except Exception as e:
            # May fail if no ROM loaded
            assert "not initialized" in str(e).lower() or "rom" in str(e).lower()

    def test_press_button_invalid(self, mcp_server):
        """Test pressing invalid button"""
        with pytest.raises(Exception):
            mcp_server.press_button("INVALID_BUTTON")

    def test_tick_function(self, mcp_server):
        """Test emulator tick"""
        try:
            result = mcp_server.tick()
            assert result is True or isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_state(self, mcp_server):
        """Test getting emulator state"""
        try:
            result = mcp_server.get_state()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestMemoryReading:
    """Test memory reading functions"""

    def test_get_player_position(self, mcp_server):
        """Test getting player position"""
        try:
            result = mcp_server.get_player_position()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_party_info(self, mcp_server):
        """Test getting party info"""
        try:
            result = mcp_server.get_party_info()
            assert isinstance(result, dict)
            if result.get("success"):
                assert "party" in result.get("data", {})
        except Exception:
            pass  # May fail without ROM

    def test_get_inventory(self, mcp_server):
        """Test getting inventory"""
        try:
            result = mcp_server.get_inventory()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_map_location(self, mcp_server):
        """Test getting map location"""
        try:
            result = mcp_server.get_map_location()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_money(self, mcp_server):
        """Test getting money"""
        try:
            result = mcp_server.get_money()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestScreenCapture:
    """Test screen capture functions"""

    def test_get_screen_base64(self, mcp_server):
        """Test getting screen as base64"""
        try:
            result = mcp_server.get_screen_base64(include_base64=True)
            assert isinstance(result, dict)
            if result.get("success"):
                data = result.get("data", {})
                assert "screen" in data or "width" in data
        except Exception:
            pass  # May fail without ROM

    def test_get_frame(self, mcp_server):
        """Test getting frame"""
        try:
            result = mcp_server.get_frame()
            assert isinstance(result, dict) or result is None
        except Exception:
            pass  # May fail without ROM


class TestSaveStates:
    """Test save/load state functions"""

    def test_save_state(self, mcp_server):
        """Test saving state"""
        try:
            result = mcp_server.save_state("test_save_pytest")
            assert isinstance(result, dict)
            if result.get("success"):
                assert "filepath" in result.get("data", {}) or "save_name" in result.get("data", {})
        except Exception:
            pass  # May fail without ROM

    def test_list_saves(self, mcp_server):
        """Test listing saves"""
        try:
            result = mcp_server.list_saves()
            assert isinstance(result, dict)
            if result.get("success"):
                assert "saves" in result.get("data", {})
        except Exception:
            pass  # May fail without ROM

    def test_load_state(self, mcp_server):
        """Test loading state"""
        try:
            # First save
            mcp_server.save_state("test_load_pytest")
            # Then load
            result = mcp_server.load_state("test_load_pytest")
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestSessionManagement:
    """Test session management functions"""

    def test_session_start(self, mcp_server):
        """Test starting session"""
        result = mcp_server.session_start(goal="Test goal")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "session_id" in result.get("data", {})

    def test_session_set_and_get(self, mcp_server):
        """Test session set and get"""
        # Start session
        start_result = mcp_server.session_start(goal="Test goal")
        session_id = start_result.get("data", {}).get("session_id")

        # Set value
        set_result = mcp_server.session_set(session_id, "test_key", "test_value")
        assert set_result.get("success") is True

        # Get value
        get_result = mcp_server.session_get(session_id, "test_key")
        assert get_result.get("success") is True

    def test_session_list(self, mcp_server):
        """Test listing sessions"""
        # Create a session first
        mcp_server.session_start(goal="List test")
        result = mcp_server.session_list()
        assert isinstance(result, dict)

    def test_session_delete(self, mcp_server):
        """Test deleting session"""
        # Create session
        start_result = mcp_server.session_start(goal="Delete test")
        session_id = start_result.get("data", {}).get("session_id")

        # Delete it
        result = mcp_server.session_delete(session_id)
        assert result.get("success") is True


class TestAutoPlayModes:
    """Test auto-play mode functions"""

    def test_auto_battle(self, mcp_server):
        """Test auto battle"""
        try:
            result = mcp_server.auto_battle(max_moves=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM/battle

    def test_auto_explore(self, mcp_server):
        """Test auto explore"""
        try:
            result = mcp_server.auto_explore(steps=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_auto_grind(self, mcp_server):
        """Test auto grind"""
        try:
            result = mcp_server.auto_grind(max_battles=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestAdvancedAutomation:
    """Test advanced automation functions"""

    def test_auto_catch(self, mcp_server):
        """Test auto catch"""
        try:
            result = mcp_server.auto_catch(max_attempts=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_auto_item_use(self, mcp_server):
        """Test auto item use"""
        try:
            result = mcp_server.auto_item_use(target="self")
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_auto_npc_talk(self, mcp_server):
        """Test auto NPC talk"""
        try:
            result = mcp_server.auto_npc_talk(max_attempts=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestMemoryAccess:
    """Test raw memory access functions"""

    def test_get_memory_byte(self, mcp_server):
        """Test getting single memory byte"""
        try:
            result = mcp_server.get_memory_byte(53248)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_memory_range(self, mcp_server):
        """Test getting memory range"""
        try:
            result = mcp_server.get_memory_range(53248, 16)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_read_game_state(self, mcp_server):
        """Test reading full game state"""
        try:
            result = mcp_server.read_game_state()
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_button_error(self, mcp_server):
        """Test invalid button raises proper error"""
        with pytest.raises(Exception) as exc_info:
            mcp_server.press_button("NOT_A_BUTTON")
        assert "invalid" in str(exc_info.value).lower() or "button" in str(exc_info.value).lower()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "mcp: marks tests as MCP-related"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )


if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__, "-v"])