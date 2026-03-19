#!/usr/bin/env python3
"""
Test script for OpenClaw Model Discovery

Usage:
    python3 test_model_discovery.py
"""
import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'ai-game-server', 'src', 'backend')
sys.path.insert(0, backend_path)

from ai_apis.openclaw_model_discovery import get_model_discovery

def test_model_discovery():
    """Test model discovery service"""
    print("=" * 60)
    print("OpenClaw Model Discovery Test")
    print("=" * 60)
    
    # Initialize discovery service
    discovery = get_model_discovery("http://localhost:18789")
    
    print("\n1. Testing get_available_models()...")
    try:
        models = discovery.get_available_models()
        print(f"   ✓ Found {len(models)} models")
        
        if models:
            print("\n   Sample models:")
            for model in models[:5]:
                vision_tag = " [VISION]" if model.is_vision_capable else ""
                free_tag = " [FREE]" if model.is_free else ""
                print(f"   - {model.name} ({model.provider}){vision_tag}{free_tag}")
                print(f"     {model.description}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n2. Testing get_vision_models()...")
    try:
        vision_models = discovery.get_vision_models()
        print(f"   ✓ Found {len(vision_models)} vision-capable models")
        
        if vision_models:
            for model in vision_models[:3]:
                print(f"   - {model.name} ({model.provider})")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Testing get_planning_models()...")
    try:
        planning_models = discovery.get_planning_models()
        print(f"   ✓ Found {len(planning_models)} planning models")
        
        if planning_models:
            for model in planning_models[:3]:
                print(f"   - {model.name} ({model.provider})")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n4. Testing recommend_model('vision')...")
    try:
        recommended = discovery.recommend_model('vision')
        if recommended:
            print(f"   ✓ Recommended: {recommended.name} ({recommended.provider})")
            print(f"     {recommended.description}")
        else:
            print(f"   ✗ No recommendation available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n5. Testing recommend_model('planning')...")
    try:
        recommended = discovery.recommend_model('planning')
        if recommended:
            print(f"   ✓ Recommended: {recommended.name} ({recommended.provider})")
            print(f"     {recommended.description}")
        else:
            print(f"   ✗ No recommendation available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n6. Testing recommend_model('free')...")
    try:
        recommended = discovery.recommend_model('free')
        if recommended:
            print(f"   ✓ Recommended: {recommended.name} ({recommended.provider})")
            print(f"     {recommended.description}")
        else:
            print(f"   ✗ No recommendation available")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n7. Testing cache...")
    try:
        # First call (may fetch from OpenClaw)
        models1 = discovery.get_available_models()
        print(f"   First call: {len(models1)} models")
        
        # Second call (should use cache)
        models2 = discovery.get_available_models()
        print(f"   Second call (cached): {len(models2)} models")
        print(f"   Cache valid: {discovery._is_cache_valid()}")
        
        # Force refresh
        models3 = discovery.get_available_models(force_refresh=True)
        print(f"   Third call (refreshed): {len(models3)} models")
        
        print(f"   ✓ Cache working correctly")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_model_discovery()
