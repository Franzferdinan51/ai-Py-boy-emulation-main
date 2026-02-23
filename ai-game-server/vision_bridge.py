#!/usr/bin/env python3
"""
Vision Bridge for PyBoy Emulator
Converts PyBoy frames to OpenClaw-compatible format for vision analysis

Usage:
    python3 vision_bridge.py --save /path/to/screenshot.png
    python3 vision_bridge.py --base64  # Output base64 for API
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime

try:
    from pyboy import PyBoy
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False
    print("⚠️  PyBoy not installed")

from PIL import Image


def capture_frame(emulator: PyBoy, output_path: str = None) -> dict:
    """
    Capture current emulator frame and save/return in multiple formats
    
    Returns:
        dict with keys: path, base64, dimensions, timestamp, frame
    """
    if emulator is None:
        raise ValueError("Emulator not initialized")
    
    # Get screen buffer
    screen = emulator.screen
    if screen is None:
        raise ValueError("No screen buffer available")
    
    # Convert to PIL Image
    img = Image.fromarray(screen)
    
    # Prepare result
    result = {
        'dimensions': f"{img.width}x{img.height}",
        'timestamp': datetime.now().isoformat(),
        'frame': getattr(emulator, '_frame_count', 0)
    }
    
    # Save to file if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, format='PNG')
        result['path'] = output_path
        result['size_bytes'] = os.path.getsize(output_path)
    
    # Convert to base64
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    result['base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return result


def analyze_with_vision(image_base64: str, prompt: str, model: str = "gemini-3-flash") -> str:
    """
    Send frame to vision model for analysis
    
    Args:
        image_base64: Base64-encoded PNG image
        prompt: Question/prompt for vision model
        model: Vision model to use (gemini-3-flash, qwen-vl, etc.)
    
    Returns:
        Analysis text from vision model
    """
    # This would integrate with OpenClaw's vision tools
    # For now, return a placeholder
    return f"[Vision analysis with {model}]: {prompt}"


def main():
    parser = argparse.ArgumentParser(description="PyBoy Vision Bridge")
    parser.add_argument("--rom", required=True, help="ROM file path")
    parser.add_argument("--save", help="Save screenshot to this path")
    parser.add_argument("--base64", action="store_true", help="Output base64 JSON")
    parser.add_argument("--frames", type=int, default=1, help="Advance N frames before capture")
    parser.add_argument("--analyze", help="Analyze with vision model (provide prompt)")
    
    args = parser.parse_args()
    
    if not PYBOY_AVAILABLE:
        print("Error: PyBoy not installed")
        sys.exit(1)
    
    # Initialize emulator
    print(f"Loading ROM: {args.rom}")
    emulator = PyBoy(args.rom, window="null")
    
    # Advance frames if requested
    if args.frames > 1:
        print(f"Advancing {args.frames} frames...")
        for _ in range(args.frames):
            emulator.tick()
    
    # Capture frame
    try:
        result = capture_frame(emulator, args.save)
        
        if args.base64:
            # Output JSON with base64
            print(json.dumps(result, indent=2))
        elif args.save:
            print(f"✅ Screenshot saved: {args.save}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Size: {result['size_bytes']} bytes")
        else:
            print(f"Frame captured: {result['dimensions']}")
        
        # Analyze with vision if requested
        if args.analyze:
            print(f"\n🔍 Analyzing with vision model...")
            analysis = analyze_with_vision(result['base64'], args.analyze)
            print(analysis)
    
    finally:
        emulator.stop()


if __name__ == "__main__":
    main()
