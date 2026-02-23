# ClaudePlayer Integration - OpenClaw Game Agent

**Status:** ✅ Integrated (2026-02-23)  
**Adapted from:** https://github.com/jmurth1234/ClaudePlayer  
**Location:** `ai-game-server/openclaw_agent.py`

---

## Overview

ClaudePlayer has been **fully integrated** into the Py-Boy emulation system. The agent is now model-agnostic and works with ANY OpenClaw model (Gemini, Qwen, GLM, MiniMax, local models).

**Key Change:** Replaced Claude 3.7 (paid API) with OpenClaw Gateway - **NO API COST**.

---

## 🎮 What's Integrated

| Component | Location | Purpose |
|-----------|----------|---------|
| **OpenClaw Agent** | `ai-game-server/openclaw_agent.py` | Model-agnostic game agent (450+ lines) |
| **Launcher** | `run_openclaw_agent.sh` | Easy CLI with model selection |
| **Model Guide** | `MODELS.md` | 9 models compared (cost, performance) |

---

## 📦 Supported Models (9 Total)

### Vision-Capable (Recommended for Gameplay)
| Model | Alias | Cost | Best For |
|-------|-------|------|----------|
| **Gemini 3 Flash** | `gemini-3-flash` | ✅ Unlimited | **Default** - Fast, free |
| **Gemini 3 Pro** | `gemini-3-pro` | ~100/day | Complex strategy |
| **Qwen 3.5 Plus** | `qwen-3.5-plus` | 18K/month | Alternative vision |
| **Qwen-VL Max** | `qwen-vl` | API credits | Advanced vision |

### Text-Only (Budget/Testing)
| Model | Alias | Cost | Best For |
|-------|-------|------|----------|
| **MiniMax M2.5** | `minimax` | ✅ Free | Simple games |
| **GLM-5** | `glm-5` | API credits | Better reasoning |
| **LM Studio Jan 4B** | `lmstudio-jan` | ✅ Local | Offline testing |

---

## 🚀 Quick Start

### List Available Models
```bash
cd /home/duckets/ai-Py-boy-emulation-main
python3 ai-game-server/openclaw_agent.py --list-models
```

### Launch with Default Model (Gemini 3 Flash)
```bash
./run_openclaw_agent.sh "/home/duckets/roms/Pokemon - Red Version.gb" "Get starter Pokemon"
```

### Launch with Specific Model
```bash
# Use Qwen 3.5 Plus (vision)
./run_openclaw_agent.sh "/home/duckets/roms/Pokemon Red.gb" "Beat Brock" qwen-3.5-plus

# Use local model (free, offline)
./run_openclaw_agent.sh "/home/duckets/roms/Super Mario Land.gb" "Beat World 1" lmstudio-jan
```

### Python API
```python
from ai-game-server.openclaw_agent import OpenClawGameAgent

config = {
    'ROM_PATH': '/home/duckets/roms/Pokemon - Red Version.gb',
    'MODEL_DEFAULTS': {
        'MODEL': 'google-gemini-cli/gemini-3-flash-preview'  # Any OpenClaw model
    }
}

agent = OpenClawGameAgent(config)

# Auto-play 10 turns with vision
agent.run_auto(max_turns=10, use_vision=True)

# Or manual control
agent.send_inputs("A B START")
agent.save_screenshot()
```

---

## 🎯 Agent Features

| Feature | Description |
|---------|-------------|
| **Model-Agnostic** | Works with ANY OpenClaw model (9+ supported) |
| **Vision Detection** | Auto-detects if model supports vision |
| **Memory System** | Short-term + long-term game memory |
| **Screenshot Capture** | Auto-saves frames to `frames/` directory |
| **Input Parsing** | Understands `A`, `A2`, `AB`, `R2 A U3`, etc. |
| **Auto-Play Mode** | Run multiple turns autonomously |

---

## 🔧 Configuration

### Full Config Example
```json
{
  "ROM_PATH": "/home/duckets/roms/Pokemon - Red Version.gb",
  "MODEL_DEFAULTS": {
    "MODEL": "google-gemini-cli/gemini-3-flash-preview",
    "THINKING": false,
    "MAX_TOKENS": 4000
  },
  "EMULATION_MODE": "turn_based",
  "CUSTOM_INSTRUCTIONS": "Focus on completing the Pokemon League efficiently",
  "LOG_FILE": "/home/duckets/.openclaw/workspace/logs/openclaw-game-agent.log"
}
```

### Switch Models
Just change the `MODEL` field:
```json
"MODEL_DEFAULTS": {
  "MODEL": "bailian/qwen3.5-plus"  // Switch to Qwen
}
```

---

## 📊 Integration with Py-Boy MCP

The OpenClaw Agent works alongside the existing MCP server:

| Tool | Use Case |
|------|----------|
| **MCP Server** (`mcp_server.py`) | Low-level control (load ROM, press button, get frame) |
| **OpenClaw Agent** (`openclaw_agent.py`) | High-level AI gameplay (vision + decision loop) |

**Use MCP for:**
- Simple automation scripts
- Direct button control
- Frame capture without AI

**Use OpenClaw Agent for:**
- Autonomous AI gameplay
- Vision-based decision making
- Complex strategy with memory

---

## 🎮 Usage Examples

### Pokemon Red - Get Starter
```bash
./run_openclaw_agent.sh \
  "/home/duckets/roms/Pokemon - Red Version.gb" \
  "Choose starter Pokemon, talk to rival, reach Route 1" \
  gemini-3-flash
```

### Super Mario Land - Speedrun
```bash
./run_openclaw_agent.sh \
  "/home/duckets/roms/Super Mario Land.gb" \
  "Beat World 1-1 as fast as possible" \
  gemini-3-flash
```

### Tetris - High Score
```bash
./run_openclaw_agent.sh \
  "/home/duckets/roms/Tetris.gb" \
  "Get 100 lines, maximize score" \
  minimax  # Text-only is fine for Tetris
```

---

## 📁 File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/
│   ├── mcp_server.py           # MCP tools (low-level)
│   ├── openclaw_agent.py       # OpenClaw Agent (AI gameplay) ← NEW
│   ├── vision_bridge.py        # Frame capture utility
│   └── test_emulator.py        # Test script
├── run_openclaw_agent.sh       # Launcher script ← NEW
├── MODELS.md                   # Model selection guide ← NEW
├── OPENCLAW-INTEGRATION.md     # Original integration docs
└── CLAUDEPLAYER-INTEGRATION.md # This file ← NEW
```

---

## 💰 Cost Comparison

### Playing 1 Hour of Pokemon Red
(~60 AI decisions, vision-enabled)

| Model | Cost | Notes |
|-------|------|-------|
| **Gemini 3 Flash** | ✅ $0 | Unlimited on Pro plan |
| **Gemini 3 Pro** | ~$0.01 | Within daily limit |
| **Qwen 3.5 Plus** | ~$0.003 | From 18K/month quota |
| **MiniMax** | ✅ $0 | Free tier |
| **LM Studio** | ✅ $0 | Local (electricity only) |

**Best value:** Gemini 3 Flash (unlimited + vision)

---

## 🛠️ Troubleshooting

### Model Not Found
```bash
# List available models
python3 ai-game-server/openclaw_agent.py --list-models

# Use full model name
./run_openclaw_agent.sh "rom.gb" "goal" "google-gemini-cli/gemini-3-flash-preview"
```

### Vision Not Working
```python
# Check model capability
agent = OpenClawGameAgent(config)
print(agent.model_info['vision'])  # True/False

# Force disable for text-only models
agent.run_auto(max_turns=10, use_vision=False)
```

### PyBoy Import Error
```bash
pip3 install pyboy --break-system-packages
```

---

## 📈 Next Steps

- [ ] Create MCP wrapper for OpenClaw Agent tools
- [ ] Add vision-based gameplay loop examples
- [ ] Create game-specific strategy guides
- [ ] Integrate memory with OpenClaw MEMORY.md

---

**Integration Date:** 2026-02-23  
**Author:** DuckBot (OpenClaw Agent)  
**License:** MIT (same as ClaudePlayer)
