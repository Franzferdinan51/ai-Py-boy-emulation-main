#!/usr/bin/env python3
"""
Comprehensive Bug Hunt Test Suite for AI Game Server
Tests all API endpoints, ROM loading, screen capture, memory reading, and error handling
"""
import requests
import json
import time
import sys
import os
from typing import Dict, Any, Optional

# Configuration
BASE_URL = os.environ.get('SERVER_URL', 'http://localhost:5002')
TIMEOUT = 10  # seconds
TEST_ROM_PATH = os.environ.get('TEST_ROM_PATH', '/tmp/test_rom.gb')

# Test results tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': [],
    'details': []
}

def log_result(test_name: str, passed: bool, details: str = "", error: str = ""):
    """Log test result"""
    if passed:
        test_results['passed'] += 1
        status = "✅ PASS"
    else:
        test_results['failed'] += 1
        status = "❌ FAIL"
        if error:
            test_results['errors'].append({
                'test': test_name,
                'error': error
            })
    
    test_results['details'].append({
        'test': test_name,
        'passed': passed,
        'details': details,
        'error': error
    })
    
    print(f"{status} | {test_name}")
    if details:
        print(f"       {details}")
    if error:
        print(f"       ERROR: {error}")
    print()

def make_request(method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
    """Make HTTP request with error handling"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT
        
        response = requests.request(method, url, **kwargs)
        return response
    except requests.exceptions.ConnectionError as e:
        return None
    except requests.exceptions.Timeout as e:
        return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

def test_health_endpoint():
    """Test /health endpoint responds correctly"""
    response = make_request('GET', '/health')
    
    if response is None:
        log_result("Health Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code != 200:
        log_result("Health Endpoint", False, 
                  details=f"Status: {response.status_code}",
                  error=f"Expected 200, got {response.status_code}")
        return False
    
    try:
        data = response.json()
        if data.get('status') != 'healthy':
            log_result("Health Endpoint", False,
                      details=f"Status in response: {data.get('status')}")
            return False
        
        log_result("Health Endpoint", True,
                  details=f"Service: {data.get('service', 'unknown')}, Version: {data.get('version', 'unknown')}")
        return True
    except json.JSONDecodeError as e:
        log_result("Health Endpoint", False, error=f"Invalid JSON: {e}")
        return False

def test_status_endpoint():
    """Test /api/status endpoint"""
    response = make_request('GET', '/api/status')
    
    if response is None:
        log_result("Status Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code != 200:
        log_result("Status Endpoint", False, error=f"Status code: {response.status_code}")
        return False
    
    try:
        data = response.json()
        required_fields = ['rom_loaded', 'active_emulator', 'ai_providers']
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            log_result("Status Endpoint", False,
                      details=f"Missing fields: {missing}")
            return False
        
        log_result("Status Endpoint", True,
                  details=f"ROM loaded: {data.get('rom_loaded')}, Emulator: {data.get('active_emulator')}")
        return True
    except Exception as e:
        log_result("Status Endpoint", False, error=str(e))
        return False

# ============================================================================
# ROM LOADING TESTS
# ============================================================================

def test_upload_rom_no_file():
    """Test ROM upload without file - should return 400"""
    response = make_request('POST', '/api/upload-rom', files={})
    
    if response is None:
        log_result("ROM Upload (No File)", False, error="Server not reachable")
        return False
    
    if response.status_code == 400:
        log_result("ROM Upload (No File)", True,
                  details="Correctly rejected missing file")
        return True
    else:
        log_result("ROM Upload (No File)", False,
                  details=f"Expected 400, got {response.status_code}")
        return False

def test_upload_rom_invalid_extension():
    """Test ROM upload with invalid extension"""
    # Create a fake ROM file with wrong extension
    test_file = '/tmp/fake_rom.txt'
    with open(test_file, 'w') as f:
        f.write("fake rom content")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'rom_file': ('fake_rom.txt', f, 'text/plain')}
            response = make_request('POST', '/api/upload-rom', files=files)
        
        if response is None:
            log_result("ROM Upload (Invalid Ext)", False, error="Server not reachable")
            return False
        
        if response.status_code == 400:
            log_result("ROM Upload (Invalid Ext)", True,
                      details="Correctly rejected invalid extension")
            return True
        else:
            log_result("ROM Upload (Invalid Ext)", False,
                      details=f"Expected 400, got {response.status_code}")
            return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

def test_load_rom_invalid_path():
    """Test loading ROM from invalid path"""
    payload = {
        'path': '/nonexistent/path/to/rom.gb',
        'emulator_type': 'gb'
    }
    
    response = make_request('POST', '/api/rom/load', json=payload)
    
    if response is None:
        log_result("Load ROM (Invalid Path)", False, error="Server not reachable")
        return False
    
    if response.status_code in [400, 404]:
        log_result("Load ROM (Invalid Path)", True,
                  details=f"Correctly rejected invalid path (status: {response.status_code})")
        return True
    else:
        log_result("Load ROM (Invalid Path)", False,
                  details=f"Expected 400/404, got {response.status_code}")
        return False

# ============================================================================
# SCREEN CAPTURE TESTS
# ============================================================================

def test_screen_no_rom():
    """Test screen capture without ROM loaded - should handle gracefully"""
    response = make_request('GET', '/api/screen')
    
    if response is None:
        log_result("Screen Capture (No ROM)", False, error="Server not reachable")
        return False
    
    # Should return error since no ROM loaded
    if response.status_code in [400, 500]:
        try:
            data = response.json()
            if 'error' in data:
                log_result("Screen Capture (No ROM)", True,
                          details=f"Correctly returned error: {data['error']}")
                return True
        except:
            pass
    
    log_result("Screen Capture (No ROM)", False,
              details=f"Unexpected status: {response.status_code}")
    return False

def test_screen_debug_endpoint():
    """Test /api/screen/debug endpoint"""
    response = make_request('GET', '/api/screen/debug')
    
    if response is None:
        log_result("Screen Debug Endpoint", False, error="Server not reachable")
        return False
    
    # Should respond even without ROM (200 with ROM, 503 without)
    if response.status_code in [200, 503]:
        try:
            data = response.json()
            if 'rom_loaded' in data:
                log_result("Screen Debug Endpoint", True,
                          details=f"Response keys: {list(data.keys())[:5]}, Status: {response.status_code}")
                return True
            else:
                log_result("Screen Debug Endpoint", False, error="Missing rom_loaded field")
                return False
        except Exception as e:
            log_result("Screen Debug Endpoint", False, error=f"JSON parse error: {e}")
            return False
    else:
        log_result("Screen Debug Endpoint", False,
                  details=f"Unexpected status: {response.status_code}")
        return False

# ============================================================================
# MEMORY READING TESTS
# ============================================================================

def test_memory_read_no_rom():
    """Test memory reading without ROM"""
    response = make_request('GET', '/api/memory/0xD058')
    
    if response is None:
        log_result("Memory Read (No ROM)", False, error="Server not reachable")
        return False
    
    # Should handle gracefully
    if response.status_code in [200, 400, 500]:
        try:
            data = response.json()
            if 'error' in data or 'values' in data:
                log_result("Memory Read (No ROM)", True,
                          details="Handled gracefully")
                return True
        except:
            pass
    
    log_result("Memory Read (No ROM)", False,
              details=f"Unexpected response: {response.status_code}")
    return False

def test_memory_read_invalid_address():
    """Test memory reading with invalid address"""
    # Address out of range (> 0xFFFF)
    response = make_request('GET', '/api/memory/999999')
    
    if response is None:
        log_result("Memory Read (Invalid Addr)", False, error="Server not reachable")
        return False
    
    if response.status_code == 400:
        log_result("Memory Read (Invalid Addr)", True,
                  details="Correctly rejected invalid address")
        return True
    else:
        log_result("Memory Read (Invalid Addr)", False,
                  details=f"Expected 400, got {response.status_code}")
        return False

def test_memory_write_no_rom():
    """Test memory writing without ROM"""
    payload = {'value': 0x42}
    response = make_request('POST', '/api/memory/0xD058', json=payload)
    
    if response is None:
        log_result("Memory Write (No ROM)", False, error="Server not reachable")
        return False
    
    # Should handle gracefully
    if response.status_code in [200, 400, 500]:
        log_result("Memory Write (No ROM)", True,
                  details="Handled gracefully")
        return True
    
    log_result("Memory Write (No ROM)", False,
              details=f"Unexpected status: {response.status_code}")
    return False

# ============================================================================
# ACTION EXECUTION TESTS
# ============================================================================

def test_action_no_rom():
    """Test action execution without ROM"""
    payload = {'action': 'A'}
    response = make_request('POST', '/api/action', json=payload)
    
    if response is None:
        log_result("Action (No ROM)", False, error="Server not reachable")
        return False
    
    if response.status_code == 400:
        log_result("Action (No ROM)", True,
                  details="Correctly rejected - no ROM loaded")
        return True
    else:
        log_result("Action (No ROM)", False,
                  details=f"Expected 400, got {response.status_code}")
        return False

def test_action_invalid_button():
    """Test action with invalid button"""
    payload = {'action': 'INVALID_BUTTON'}
    response = make_request('POST', '/api/action', json=payload)
    
    if response is None:
        log_result("Action (Invalid Button)", False, error="Server not reachable")
        return False
    
    if response.status_code == 400:
        log_result("Action (Invalid Button)", True,
                  details="Correctly rejected invalid button")
        return True
    else:
        log_result("Action (Invalid Button)", False,
                  details=f"Expected 400, got {response.status_code}")
        return False

def test_action_valid_buttons():
    """Test all valid button actions"""
    valid_buttons = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT', 'NOOP']
    
    # This test will fail without ROM, but we can check the validation
    for button in valid_buttons[:3]:  # Test first 3
        payload = {'action': button, 'frames': 1}
        response = make_request('POST', '/api/action', json=payload)
        
        if response is None:
            log_result(f"Action ({button})", False, error="Server not reachable")
            continue
        
        # Should at least validate the button name
        if response.status_code in [200, 400, 500]:
            log_result(f"Action ({button})", True,
                      details=f"Status: {response.status_code}")
        else:
            log_result(f"Action ({button})", False,
                      details=f"Unexpected status: {response.status_code}")

# ============================================================================
# AI ACTION TESTS
# ============================================================================

def test_ai_action_no_rom():
    """Test AI action without ROM"""
    payload = {'goal': 'Test goal'}
    response = make_request('POST', '/api/ai-action', json=payload)
    
    if response is None:
        log_result("AI Action (No ROM)", False, error="Server not reachable")
        return False
    
    if response.status_code == 400:
        log_result("AI Action (No ROM)", True,
                  details="Correctly rejected - no ROM loaded")
        return True
    else:
        log_result("AI Action (No ROM)", False,
                  details=f"Expected 400, got {response.status_code}")
        return False

def test_ai_action_invalid_json():
    """Test AI action with invalid JSON"""
    response = make_request('POST', '/api/ai-action', 
                           data='not valid json',
                           headers={'Content-Type': 'application/json'})
    
    if response is None:
        log_result("AI Action (Invalid JSON)", False, error="Server not reachable")
        return False
    
    if response.status_code in [400, 500]:
        log_result("AI Action (Invalid JSON)", True,
                  details="Correctly handled invalid JSON")
        return True
    else:
        log_result("AI Action (Invalid JSON)", False,
                  details=f"Expected 400/500, got {response.status_code}")
        return False

# ============================================================================
# GAME STATE TESTS
# ============================================================================

def test_game_state_endpoint():
    """Test /api/game/state endpoint"""
    response = make_request('GET', '/api/game/state')
    
    if response is None:
        log_result("Game State Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code == 200:
        try:
            data = response.json()
            required = ['running', 'rom_loaded', 'screen_available']
            missing = [f for f in required if f not in data]
            
            if missing:
                log_result("Game State Endpoint", False,
                          details=f"Missing fields: {missing}")
                return False
            
            log_result("Game State Endpoint", True,
                      details=f"Running: {data.get('running')}, ROM: {data.get('rom_loaded')}")
            return True
        except Exception as e:
            log_result("Game State Endpoint", False, error=str(e))
            return False
    else:
        log_result("Game State Endpoint", False,
                  details=f"Status: {response.status_code}")
        return False

def test_agent_status_endpoint():
    """Test /api/agent/status endpoint"""
    response = make_request('GET', '/api/agent/status')
    
    if response is None:
        log_result("Agent Status Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code == 200:
        try:
            data = response.json()
            log_result("Agent Status Endpoint", True,
                      details=f"Mode: {data.get('mode')}, Enabled: {data.get('enabled')}")
            return True
        except Exception as e:
            log_result("Agent Status Endpoint", False, error=str(e))
            return False
    else:
        log_result("Agent Status Endpoint", False,
                  details=f"Status: {response.status_code}")
        return False

# ============================================================================
# PARTY & INVENTORY TESTS
# ============================================================================

def test_party_endpoint():
    """Test /api/party endpoint"""
    response = make_request('GET', '/api/party')
    
    if response is None:
        log_result("Party Endpoint", False, error="Server not reachable")
        return False
    
    # Should handle gracefully without ROM
    if response.status_code in [200, 400, 500]:
        try:
            data = response.json()
            if 'error' in data or 'party' in data:
                log_result("Party Endpoint", True,
                          details="Handled gracefully")
                return True
        except:
            pass
    
    log_result("Party Endpoint", False,
              details=f"Unexpected: {response.status_code}")
    return False

def test_inventory_endpoint():
    """Test /api/inventory endpoint"""
    response = make_request('GET', '/api/inventory')
    
    if response is None:
        log_result("Inventory Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code in [200, 400, 500]:
        try:
            data = response.json()
            if 'error' in data or 'money' in data or 'items' in data:
                log_result("Inventory Endpoint", True,
                          details="Handled gracefully")
                return True
        except:
            pass
    
    log_result("Inventory Endpoint", False,
              details=f"Unexpected: {response.status_code}")
    return False

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

def test_openclaw_config():
    """Test OpenClaw configuration endpoints"""
    # GET config
    response = make_request('GET', '/api/openclaw/config')
    
    if response is None:
        log_result("OpenClaw Config (GET)", False, error="Server not reachable")
        return False
    
    if response.status_code == 200:
        log_result("OpenClaw Config (GET)", True)
    else:
        log_result("OpenClaw Config (GET)", False,
                  details=f"Status: {response.status_code}")
        return False
    
    # POST config update
    payload = {
        'vision_model': 'kimi-k2.5',
        'planning_model': 'glm-5',
        'use_dual_model': True
    }
    response = make_request('POST', '/api/openclaw/config', json=payload)
    
    if response is None:
        log_result("OpenClaw Config (POST)", False, error="Server not reachable")
        return False
    
    if response.status_code in [200, 400]:
        log_result("OpenClaw Config (POST)", True,
                  details=f"Status: {response.status_code}")
        return True
    else:
        log_result("OpenClaw Config (POST)", False,
                  details=f"Unexpected: {response.status_code}")
        return False

def test_ai_runtime_config():
    """Test AI runtime configuration"""
    response = make_request('GET', '/api/ai/runtime')
    
    if response is None:
        log_result("AI Runtime Config", False, error="Server not reachable")
        return False
    
    if response.status_code == 200:
        try:
            data = response.json()
            log_result("AI Runtime Config", True,
                      details=f"Provider: {data.get('provider')}")
            return True
        except Exception as e:
            log_result("AI Runtime Config", False, error=str(e))
            return False
    else:
        log_result("AI Runtime Config", False,
                  details=f"Status: {response.status_code}")
        return False

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_404_endpoint():
    """Test 404 handling"""
    response = make_request('GET', '/nonexistent/endpoint')
    
    if response is None:
        log_result("404 Handling", False, error="Server not reachable")
        return False
    
    if response.status_code == 404:
        try:
            data = response.json()
            if 'error' in data:
                log_result("404 Handling", True,
                          details="Returns proper JSON error")
                return True
        except:
            pass
        log_result("404 Handling", True,
                  details="Returns 404")
        return True
    else:
        log_result("404 Handling", False,
                  details=f"Expected 404, got {response.status_code}")
        return False

def test_method_not_allowed():
    """Test 405 Method Not Allowed"""
    # Try POST on GET-only endpoint
    response = make_request('POST', '/health')
    
    if response is None:
        log_result("405 Handling", False, error="Server not reachable")
        return False
    
    if response.status_code == 405:
        log_result("405 Handling", True,
                  details="Correctly returns 405")
        return True
    else:
        log_result("405 Handling", False,
                  details=f"Expected 405, got {response.status_code}")
        return False

# ============================================================================
# PERFORMANCE & STREAMING TESTS
# ============================================================================

def test_performance_endpoint():
    """Test /api/performance endpoint"""
    response = make_request('GET', '/api/performance')
    
    if response is None:
        log_result("Performance Endpoint", False, error="Server not reachable")
        return False
    
    if response.status_code == 200:
        log_result("Performance Endpoint", True)
        return True
    else:
        log_result("Performance Endpoint", False,
                  details=f"Status: {response.status_code}")
        return False

def test_stream_endpoint_exists():
    """Test that /api/stream endpoint exists (SSE)"""
    # Just check it doesn't 404, don't actually stream
    try:
        response = requests.get(f'{BASE_URL}/api/stream', timeout=2, stream=True)
        # SSE should return 200
        if response.status_code == 200:
            log_result("Stream Endpoint Exists", True)
            return True
        else:
            log_result("Stream Endpoint Exists", False,
                      details=f"Status: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        # Timeout is expected for SSE, means endpoint exists
        log_result("Stream Endpoint Exists", True,
                  details="Endpoint exists (SSE stream)")
        return True
    except Exception as e:
        log_result("Stream Endpoint Exists", False, error=str(e))
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all bug hunt tests"""
    print("=" * 80)
    print("🐛 AI GAME SERVER BUG HUNT TEST SUITE")
    print("=" * 80)
    print(f"Target: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Check if server is reachable
    print("📡 Checking server connectivity...")
    health_response = make_request('GET', '/health')
    if health_response is None:
        print("❌ ERROR: Server is not reachable!")
        print(f"   Tried: {BASE_URL}")
        print("   Make sure the server is running:")
        print("   cd /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server")
        print("   python src/main.py")
        print()
        return
    
    print("✅ Server is reachable!")
    print()
    
    # Run test suites
    print("=" * 80)
    print("🏥 HEALTH CHECK TESTS")
    print("=" * 80)
    test_health_endpoint()
    test_status_endpoint()
    
    print("=" * 80)
    print("💾 ROM LOADING TESTS")
    print("=" * 80)
    test_upload_rom_no_file()
    test_upload_rom_invalid_extension()
    test_load_rom_invalid_path()
    
    print("=" * 80)
    print("📸 SCREEN CAPTURE TESTS")
    print("=" * 80)
    test_screen_no_rom()
    test_screen_debug_endpoint()
    
    print("=" * 80)
    print("🧠 MEMORY READING TESTS")
    print("=" * 80)
    test_memory_read_no_rom()
    test_memory_read_invalid_address()
    test_memory_write_no_rom()
    
    print("=" * 80)
    print("🎮 ACTION EXECUTION TESTS")
    print("=" * 80)
    test_action_no_rom()
    test_action_invalid_button()
    test_action_valid_buttons()
    
    print("=" * 80)
    print("🤖 AI ACTION TESTS")
    print("=" * 80)
    test_ai_action_no_rom()
    test_ai_action_invalid_json()
    
    print("=" * 80)
    print("📊 GAME STATE TESTS")
    print("=" * 80)
    test_game_state_endpoint()
    test_agent_status_endpoint()
    
    print("=" * 80)
    print("🎒 PARTY & INVENTORY TESTS")
    print("=" * 80)
    test_party_endpoint()
    test_inventory_endpoint()
    
    print("=" * 80)
    print("⚙️ CONFIGURATION TESTS")
    print("=" * 80)
    test_openclaw_config()
    test_ai_runtime_config()
    
    print("=" * 80)
    print("⚠️ ERROR HANDLING TESTS")
    print("=" * 80)
    test_404_endpoint()
    test_method_not_allowed()
    
    print("=" * 80)
    print("⚡ PERFORMANCE TESTS")
    print("=" * 80)
    test_performance_endpoint()
    test_stream_endpoint_exists()
    
    # Print summary
    print()
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    total = test_results['passed'] + test_results['failed']
    if total > 0:
        print(f"📈 Success Rate: {test_results['passed']/total*100:.1f}%")
    
    if test_results['errors']:
        print()
        print("=" * 80)
        print("❌ FAILED TESTS DETAIL")
        print("=" * 80)
        for error in test_results['errors']:
            print(f"• {error['test']}: {error['error']}")
    
    print()
    print("=" * 80)
    print("🐛 BUG HUNT COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    run_all_tests()
