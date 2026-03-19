#!/usr/bin/env python3
"""
Test script for AI-PyBoy Emulation project
Verifies backend, MCP server, and frontend all work correctly
"""

import sys
import os
import json
import time
import subprocess
import requests

# Configuration
BACKEND_PORT = 5002
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
    
    def add_pass(self, test_name):
        self.passed.append(test_name)
        print(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed.append((test_name, error))
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed)
        print("\n" + "="*50)
        print(f"TEST SUMMARY: {len(self.passed)}/{total} passed")
        print("="*50)
        if self.failed:
            print("\nFailed tests:")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {error}")
        return len(self.failed) == 0


def test_backend_health():
    """Test /health endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        assert data.get("status") == "healthy", f"Status: {data.get('status')}"
        assert "flask" in data.get("checks", {}), "Missing flask check"
        assert "mcp" in data.get("checks", {}), "Missing mcp check"
        assert "pyboy" in data.get("checks", {}), "Missing pyboy check"
        return True, None
    except Exception as e:
        return False, str(e)


def test_backend_status():
    """Test /api/status endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/status", timeout=5)
        assert response.status_code == 200, f"Status code: {response.status_code}"
        data = response.json()
        # Check key fields exist
        assert "ai_providers" in data, "Missing ai_providers"
        assert "active_emulator" in data, "Missing active_emulator"
        assert "rom_loaded" in data, "Missing rom_loaded"
        # Should have mock provider available
        assert data["ai_providers"]["mock"]["available"], "Mock provider not available"
        return True, None
    except Exception as e:
        return False, str(e)


def test_mcp_server_loads():
    """Test that MCP server module loads"""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'ai-game-server'))
        import mcp_server
        assert mcp_server.MCP_AVAILABLE, "MCP not available"
        assert mcp_server.PYBOY_AVAILABLE, "PyBoy not available"
        assert mcp_server.SERVER_VERSION == "3.0.0", f"Wrong version: {mcp_server.SERVER_VERSION}"
        return True, None
    except Exception as e:
        return False, str(e)


def test_frontend_builds():
    """Test that frontend builds successfully"""
    try:
        frontend_dir = os.path.join(PROJECT_ROOT, "ai-game-assistant")
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Build failed: {result.stderr}"
        # Check dist folder exists
        dist_dir = os.path.join(frontend_dir, "dist")
        assert os.path.exists(dist_dir), "dist folder not created"
        assert os.path.exists(os.path.join(dist_dir, "index.html")), "index.html not found"
        return True, None
    except Exception as e:
        return False, str(e)


def check_backend_running():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    print("="*50)
    print("AI-PyBoy Emulation - Test Suite")
    print("="*50)
    print()
    
    results = TestResults()
    
    # Check if backend is running
    if not check_backend_running():
        print("⚠️  Backend not running. Start with:")
        print(f"   cd {PROJECT_ROOT}/ai-game-server/src")
        print(f"   BACKEND_PORT={BACKEND_PORT} python3 main.py")
        print()
        # Try to start it
        print("Attempting to start backend...")
        try:
            subprocess.Popen(
                ["python3", "main.py"],
                cwd=os.path.join(PROJECT_ROOT, "ai-game-server", "src"),
                env={**os.environ, "BACKEND_PORT": str(BACKEND_PORT)},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)
            if not check_backend_running():
                results.add_fail("Backend startup", "Could not start backend")
            else:
                print("✅ Backend started")
        except Exception as e:
            results.add_fail("Backend startup", str(e))
    
    # Run tests
    print("\n--- Backend Tests ---")
    
    success, error = test_backend_health()
    if success:
        results.add_pass("/health endpoint")
    else:
        results.add_fail("/health endpoint", error)
    
    success, error = test_backend_status()
    if success:
        results.add_pass("/api/status endpoint")
    else:
        results.add_fail("/api/status endpoint", error)
    
    print("\n--- MCP Server Tests ---")
    
    success, error = test_mcp_server_loads()
    if success:
        results.add_pass("MCP server loads")
    else:
        results.add_fail("MCP server loads", error)
    
    print("\n--- Frontend Tests ---")
    
    success, error = test_frontend_builds()
    if success:
        results.add_pass("Frontend builds")
    else:
        results.add_fail("Frontend builds", error)
    
    # Print summary
    success = results.summary()
    
    print("\n--- Issues Found ---")
    issues = []
    
    # Check for missing API keys
    try:
        response = requests.get(f"{BACKEND_URL}/api/status", timeout=5)
        data = response.json()
        for provider, info in data.get("ai_providers", {}).items():
            if not info.get("available") and "API key" in info.get("error", ""):
                issues.append(f"AI provider '{provider}' needs API key: {info.get('error')}")
    except:
        pass
    
    # Check for Tetris genetic AI error
    try:
        response = requests.get(f"{BACKEND_URL}/api/status", timeout=5)
        data = response.json()
        if data.get("ai_providers", {}).get("tetris-genetic", {}).get("status") == "error":
            issues.append("Tetris genetic AI has implementation error - missing abstract methods")
    except:
        pass
    
    # Check for missing modules
    log_path = os.path.join(PROJECT_ROOT, "ai-game-server", "src", "backend", "ai_game_server.log")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_content = f.read()
            if "utils" in log_content and "not available" in log_content.lower():
                issues.append("Missing 'utils' module - optimization system disabled")
            if "pygba" in log_content.lower() and "not available" in log_content.lower():
                issues.append("PyGBA not installed - GBA emulation unavailable")
    
    if issues:
        print("\n".join(f"  ⚠️  {issue}" for issue in issues))
    else:
        print("  No issues found!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())