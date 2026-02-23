#!/bin/bash
# OpenClaw Game Agent Launcher
# Model-agnostic: Works with ANY OpenClaw model

set -e

ROM_PATH="$1"
GOAL="$2"
MODEL="$3"

show_help() {
    echo "🎮 OpenClaw Game Agent - Model-Agnostic"
    echo "======================================="
    echo ""
    echo "Usage: $0 <rom_path> [goal] [model]"
    echo ""
    echo "Arguments:"
    echo "  rom_path   Path to Game Boy ROM file"
    echo "  goal       Gameplay objective (optional)"
    echo "  model      Model to use (optional, default: gemini-3-flash)"
    echo ""
    echo "Available Models:"
    echo "  gemini-3-flash   - Gemini 3 Flash (✅ vision, unlimited)"
    echo "  gemini-3-pro     - Gemini 3 Pro (✅ vision, ~100/day)"
    echo "  qwen-3.5-plus    - Qwen 3.5 Plus (✅ vision, 18K/month)"
    echo "  qwen-vl          - Qwen-VL Max (✅ vision, API credits)"
    echo "  glm-5            - GLM-5 (❌ text-only, API credits)"
    echo "  glm-4.7          - GLM-4.7 (❌ text-only, API credits)"
    echo "  minimax          - MiniMax M2.5 (❌ text-only, free)"
    echo "  lmstudio-jan     - LM Studio Jan 4B (❌ text-only, local)"
    echo "  lmstudio-glm     - LM Studio GLM Flash (❌ text-only, local)"
    echo ""
    echo "Examples:"
    echo "  # Use default model (Gemini 3 Flash)"
    echo "  $0 /home/duckets/roms/Pokemon\\ -\\ Red\\ Version.gb"
    echo ""
    echo "  # Specify model"
    echo "  $0 /home/duckets/roms/Pokemon\\ -\\ Red\\ Version.gb \"Get starter Pokemon\" qwen-3.5-plus"
    echo ""
    echo "  # Use local model (no API cost)"
    echo "  $0 /home/duckets/roms/Super\\ Mario\\ Land.gb \"Beat World 1\" lmstudio-jan"
    echo ""
    echo "  # List all models"
    echo "  python3 claude_player/agent/openclaw_agent.py --list-models"
}

if [ -z "$ROM_PATH" ] || [ "$ROM_PATH" = "-h" ] || [ "$ROM_PATH" = "--help" ]; then
    show_help
    exit 0
fi

# Default goal if not provided
if [ -z "$GOAL" ]; then
    GOAL="Explore and progress through the game"
fi

# Default model if not provided
if [ -z "$MODEL" ]; then
    MODEL="google-gemini-cli/gemini-3-flash-preview"
    MODEL_ALIAS="gemini-3-flash (default)"
else
    MODEL_ALIAS="$MODEL"
fi

echo "🎮 OpenClaw Game Agent"
echo "===================="
echo "ROM: $ROM_PATH"
echo "Goal: $GOAL"
echo "Model: $MODEL_ALIAS"
echo ""

# Create config
CONFIG_FILE="/tmp/openclaw-game-config-$(date +%s).json"
cat > "$CONFIG_FILE" << EOF
{
  "ROM_PATH": "$ROM_PATH",
  "STATE_PATH": null,
  "LOG_FILE": "/home/duckets/.openclaw/workspace/logs/openclaw-game-agent.log",
  "EMULATION_MODE": "turn_based",
  "EMULATION_SPEED": 1,
  "ENABLE_WRAPPER": false,
  "ENABLE_SOUND": false,
  "MAX_HISTORY_MESSAGES": 30,
  "MAX_SCREENSHOTS": 5,
  "CUSTOM_INSTRUCTIONS": "",
  "MODEL_DEFAULTS": {
    "MODEL": "$MODEL",
    "THINKING": false,
    "MAX_TOKENS": 4000
  },
  "CURRENT_GOAL": "$GOAL"
}
EOF

echo "📝 Config created: $CONFIG_FILE"
echo ""

# Run the agent
cd /home/duckets/ClaudePlayer
python3 -c "
import sys
sys.path.insert(0, 'claude_player')
from agent.openclaw_agent import OpenClawGameAgent
import json

with open('$CONFIG_FILE') as f:
    config = json.load(f)

agent = OpenClawGameAgent(config)
print(f'✅ Game loaded: {agent.game_name}')
print(f'🤖 Model: {agent.model_name}')
print(f'   Vision: {\"✅ Yes\" if agent.model_info[\"vision\"] else \"❌ No\"}')
print(f'   Cost: {agent.model_info[\"cost\"]}')
print()
print('Ready for AI gameplay!')
print()
print('To run auto-play:')
print(f'  python3 claude_player/agent/openclaw_agent.py --rom \"{config[\"ROM_PATH\"]}\" --model {config[\"MODEL_DEFAULTS\"][\"MODEL\"]} --turns 10')
print()

# Cleanup
agent.stop()
"

# Cleanup config
rm -f "$CONFIG_FILE"

echo ""
echo "✅ Agent test complete!"
