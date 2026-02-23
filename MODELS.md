# OpenClaw Game Agent - Available Models

**Updated:** 2026-02-23

The OpenClaw Game Agent is **model-agnostic** - it works with ANY model available in OpenClaw.

---

## 📦 Recommended Models

### Best for Gameplay (Vision-Capable)

| Model | Alias | Vision | Cost | Best For |
|-------|-------|--------|------|----------|
| **Gemini 3 Flash** | `gemini-3-flash` | ✅ | Unlimited | **Default** - Fast, unlimited |
| **Gemini 3 Pro** | `gemini-3-pro` | ✅ | ~100/day | Complex strategy |
| **Qwen 3.5 Plus** | `qwen-3.5-plus` | ✅ | 18K/month | Good alternative |
| **Qwen-VL Max** | `qwen-vl` | ✅ | API credits | Advanced vision |

### Budget Options (Text-Only)

| Model | Alias | Vision | Cost | Best For |
|-------|-------|--------|------|----------|
| **MiniMax M2.5** | `minimax` | ❌ | Free | Simple games |
| **GLM-4.7** | `glm-4.7` | ❌ | API credits | Budget play |
| **GLM-5** | `glm-5` | ❌ | API credits | Better reasoning |

### Local Models (Free, Offline)

| Model | Alias | Vision | Cost | Best For |
|-------|-------|--------|------|----------|
| **Jan 4B** | `lmstudio-jan` | ❌ | Free (local) | Testing, offline |
| **GLM Flash** | `lmstudio-glm` | ❌ | Free (local) | Fast local play |

---

## 🎮 Usage

### List Available Models
```bash
cd /home/duckets/ClaudePlayer
python3 claude_player/agent/openclaw_agent.py --list-models
```

### Quick Launch with Specific Model
```bash
# Use Gemini 3 Flash (default, unlimited)
./run_openclaw_agent.sh "/path/to/Pokemon Red.gb" "Get starter Pokemon" gemini-3-flash

# Use Qwen 3.5 Plus (alternative vision model)
./run_openclaw_agent.sh "/path/to/Pokemon Red.gb" "Get starter Pokemon" qwen-3.5-plus

# Use local model (free, offline, no vision)
./run_openclaw_agent.sh "/path/to/Super Mario Land.gb" "Beat World 1" lmstudio-jan
```

### Python API
```python
from claude_player.agent.openclaw_agent import OpenClawGameAgent

# Use any model
config = {
    'ROM_PATH': '/path/to/rom.gb',
    'MODEL_DEFAULTS': {
        'MODEL': 'google-gemini-cli/gemini-3-flash-preview'  # Any OpenClaw model
    }
}

agent = OpenClawGameAgent(config)
print(f"Model: {agent.model_name}")
print(f"Vision: {agent.model_info['vision']}")
print(f"Cost: {agent.model_info['cost']}")

# Run with vision
agent.run_auto(max_turns=10, use_vision=True)

# Run without vision (for text-only models)
agent.run_auto(max_turns=10, use_vision=False)
```

---

## 🎯 Model Selection Guide

### For Pokemon/RPG Games
**Recommended:** Gemini 3 Flash or Qwen 3.5 Plus

Why:
- Vision needed to read text boxes, menus, maps
- Unlimited usage for long play sessions
- Good reasoning for inventory management, battles

```bash
./run_openclaw_agent.sh "Pokemon Red.gb" "Complete Pokemon League" gemini-3-flash
```

### For Action Games (Mario, Kirby)
**Recommended:** Gemini 3 Flash (fast) or MiniMax (budget)

Why:
- Fast response time needed
- Vision helpful for platforming
- Simple objectives don't need advanced reasoning

```bash
./run_openclaw_agent.sh "Super Mario Land.gb" "Beat World 1-4" gemini-3-flash
```

### For Testing/Development
**Recommended:** Local models (lmstudio-jan, lmstudio-glm)

Why:
- Free, unlimited testing
- No API calls
- Fast iteration

```bash
./run_openclaw_agent.sh "Tetris.gb" "Get 100 lines" lmstudio-jan
```

### For Complex Strategy
**Recommended:** Gemini 3 Pro or Qwen-VL

Why:
- Advanced reasoning
- Better long-term planning
- Superior vision analysis

```bash
./run_openclaw_agent.sh "Pokemon Red.gb" "Optimize team for Elite Four" gemini-3-pro
```

---

## 💰 Cost Comparison

### Playing 1 Hour of Pokemon Red
(Assumes ~60 decisions, vision-enabled)

| Model | Decisions | Cost |
|-------|-----------|------|
| Gemini 3 Flash | 60 | ✅ $0 (unlimited) |
| Gemini 3 Pro | 60 | ~$0.01 (within daily limit) |
| Qwen 3.5 Plus | 60 | ~$0.003 (from monthly quota) |
| Qwen-VL | 60 | ~$0.06 (API credits) |
| MiniMax | 60 | ✅ $0 (free) |
| LM Studio (local) | 60 | ✅ $0 (electricity only) |

**Best value:** Gemini 3 Flash (unlimited, vision-capable)

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
  "CUSTOM_INSTRUCTIONS": "Focus on completing the Pokemon League efficiently"
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

## 📊 Performance Comparison

| Model | Speed | Vision | Reasoning | Cost Efficiency |
|-------|-------|--------|-----------|-----------------|
| Gemini 3 Flash | ⚡⚡⚡ | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Gemini 3 Pro | ⚡⚡ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Qwen 3.5 Plus | ⚡⚡⚡ | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Qwen-VL | ⚡⚡ | ✅ | ⭐⭐⭐⭐ | ⭐⭐ |
| MiniMax | ⚡⚡⚡ | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| LM Studio | ⚡⚡⚡⚡ | ❌ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Overall Best:** Gemini 3 Flash (fast, vision, unlimited)

---

## 🛠️ Troubleshooting

### Model Not Found
```bash
# Check available models
python3 claude_player/agent/openclaw_agent.py --list-models

# Use full model name
./run_openclaw_agent.sh "rom.gb" "goal" "google-gemini-cli/gemini-3-flash-preview"
```

### Vision Not Working
```python
# Check model capability
agent = OpenClawGameAgent(config)
print(agent.model_info['vision'])  # True/False

# Force disable vision for text-only models
agent.run_auto(max_turns=10, use_vision=False)
```

### Slow Performance
- Use faster model (Gemini 3 Flash, lmstudio-jan)
- Disable vision if not needed
- Reduce max_tokens in config

---

**Model selection is flexible!** Start with Gemini 3 Flash (default), then experiment based on your needs. 🦆
