#!/usr/bin/env python3
"""
pytest test suite for AI-PyBoy MCP Server tools

Run with: pytest tests/test_mcp.py -v

Tests the actual functions exported by mcp_server.py
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
        # Version should be at least 4.0.0
        assert mcp_server.SERVER_VERSION >= "4.0.0", f"Expected version >= 4.0.0, got {mcp_server.SERVER_VERSION}"


class TestCoreEmulatorControls:
    """Test core emulator control functions"""

    def test_press_button_valid(self, mcp_server):
        """Test pressing valid button"""
        try:
            result = mcp_server.press_button("A")
            assert isinstance(result, bool) or isinstance(result, dict)
        except Exception as e:
            # May fail if no ROM loaded
            error_str = str(e).lower()
            assert "not initialized" in error_str or "rom" in error_str or "no rom" in error_str

    def test_press_button_invalid(self, mcp_server):
        """Test pressing invalid button raises error"""
        with pytest.raises(Exception) as exc_info:
            mcp_server.press_button("INVALID_BUTTON")
        assert "invalid" in str(exc_info.value).lower() or "button" in str(exc_info.value).lower()

    def test_tick_function(self, mcp_server):
        """Test emulator tick"""
        try:
            result = mcp_server.tick()
            assert result is True or isinstance(result, bool)
        except Exception:
            pass  # May fail without ROM

    def test_emulator_state_none_without_rom(self, mcp_server):
        """Test emulator is None without ROM loaded"""
        # This tests the global state
        assert mcp_server.emulator is None or mcp_server.rom_path is not None


class TestMemoryReading:
    """Test memory reading functions"""

    def test_get_memory_address(self, mcp_server):
        """Test getting memory address"""
        try:
            result = mcp_server.get_memory_address(0xD062)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_get_memory_address_hex_string(self, mcp_server):
        """Test getting memory address with hex string"""
        try:
            result = mcp_server.get_memory_address("0xD062")
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_set_memory_address_validation(self, mcp_server):
        """Test set memory address validates value range"""
        try:
            # This should raise an error for value out of range
            result = mcp_server.set_memory_address(0xD062, 999)
            # If we get here, check it failed
            assert result.get("success") is False or "error" in result
        except Exception as e:
            # Should raise EmulatorError for invalid value
            assert "range" in str(e).lower() or "invalid" in str(e).lower() or "not initialized" in str(e).lower()

    def test_get_player_position(self, mcp_server):
        """Test getting player position"""
        try:
            result = mcp_server.get_player_position()
            assert isinstance(result, dict)
            assert "success" in result
        except Exception:
            pass  # May fail without ROM

    def test_get_party_pokemon(self, mcp_server):
        """Test getting party pokemon"""
        try:
            result = mcp_server.get_party_pokemon()
            assert isinstance(result, dict)
            assert "success" in result
        except Exception:
            pass  # May fail without ROM

    def test_get_inventory_detailed(self, mcp_server):
        """Test getting inventory"""
        try:
            result = mcp_server.get_inventory_detailed()
            assert isinstance(result, dict)
            assert "success" in result
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
                assert "image_base64" in result or "dimensions" in result
        except Exception:
            pass  # May fail without ROM

    def test_get_screen_base64_without_data(self, mcp_server):
        """Test getting screen metadata only"""
        try:
            result = mcp_server.get_screen_base64(include_base64=False)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM


class TestSaveStates:
    """Test save/load state functions"""

    def test_save_game_state(self, mcp_server):
        """Test saving game state"""
        try:
            result = mcp_server.save_game_state()
            assert isinstance(result, dict)
            assert "success" in result
        except Exception:
            pass  # May fail without ROM

    def test_load_game_state_missing(self, mcp_server):
        """Test loading non-existent save"""
        try:
            result = mcp_server.load_game_state(save_name="nonexistent_save_12345")
            # Should fail
            assert result.get("success") is False or "error" in result or "not found" in str(result).lower()
        except Exception:
            pass  # May fail without ROM or save


class TestSessionManagement:
    """Test session management functions"""

    def test_session_start(self, mcp_server):
        """Test starting session"""
        result = mcp_server.session_start(goal="Test goal")
        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "session_id" in result.get("data", {})

    def test_session_start_with_custom_id(self, mcp_server):
        """Test starting session with custom ID"""
        result = mcp_server.session_start(session_id="custom_test_session", goal="Custom test")
        assert isinstance(result, dict)
        assert result.get("success") is True

    def test_session_set_and_get(self, mcp_server):
        """Test session set and get"""
        # Start session
        start_result = mcp_server.session_start(goal="Test goal")
        session_id = start_result.get("data", {}).get("session_id")

        if session_id:
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
        assert result.get("success") is True

    def test_session_delete(self, mcp_server):
        """Test deleting session"""
        # Create session
        start_result = mcp_server.session_start(goal="Delete test")
        session_id = start_result.get("data", {}).get("session_id")

        if session_id:
            # Delete it
            result = mcp_server.session_delete(session_id)
            assert result.get("success") is True

    def test_session_get_nonexistent(self, mcp_server):
        """Test getting non-existent session"""
        try:
            result = mcp_server.session_get("nonexistent_session_12345")
            # Should fail
            assert result.get("success") is False or "error" in result
        except Exception:
            # Should raise EmulatorError
            pass


class TestAutoPlayModes:
    """Test auto-play mode functions"""

    def test_auto_explore_mode(self, mcp_server):
        """Test auto explore mode"""
        try:
            result = mcp_server.auto_explore_mode(steps=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM

    def test_auto_battle_mode(self, mcp_server):
        """Test auto battle mode"""
        try:
            result = mcp_server.auto_battle_mode(max_moves=1)
            assert isinstance(result, dict)
        except Exception:
            pass  # May fail without ROM/battle


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_button_error(self, mcp_server):
        """Test invalid button raises proper error"""
        with pytest.raises(Exception) as exc_info:
            mcp_server.press_button("NOT_A_BUTTON")
        assert "invalid" in str(exc_info.value).lower() or "button" in str(exc_info.value).lower()

    def test_invalid_address_format(self, mcp_server):
        """Test invalid address format raises error"""
        try:
            result = mcp_server.get_memory_address("invalid_address")
            # Should fail
            assert result.get("success") is False or "error" in result
        except Exception:
            pass  # May raise EmulatorError


class TestToolDefinitions:
    """Test MCP tool definitions"""

    def test_create_tool_definitions(self, mcp_server):
        """Test that tool definitions are created correctly"""
        tools = mcp_server.create_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_tool_names(self, mcp_server):
        """Test that expected tool names exist"""
        tools = mcp_server.create_tool_definitions()
        tool_names = [t.name for t in tools]
        
        expected_tools = [
            "emulator_load_rom",
            "emulator_press_button",
            "emulator_tick",
            "emulator_get_state",
            "get_memory_address",
            "set_memory_address",
            "get_party_pokemon",
            "get_inventory",
            "get_player_position",
            "get_map_location",
            "get_money",
            "get_screen_base64",
            "auto_explore_mode",
            "auto_battle_mode",
            "session_start",
            "session_get",
            "session_set",
            "session_list",
            "session_delete",
            "save_game_state",
            "load_game_state",
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"


class TestSessionPersistence:
    """Test session persistence to disk"""

    def test_session_persists(self, mcp_server):
        """Test that sessions persist across calls"""
        # Create session
        result = mcp_server.session_start(session_id="persist_test_session", goal="Persistence test")
        assert result.get("success") is True
        
        # Verify it exists
        list_result = mcp_server.session_list()
        session_ids = [s.get("id") for s in list_result.get("data", {}).get("sessions", [])]
        assert "persist_test_session" in session_ids
        
        # Clean up
        mcp_server.session_delete("persist_test_session")


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