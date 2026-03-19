---
name: openclaw-integration
description: OpenClaw agent integration for PyBoy Game Boy emulation - spawn patterns, tool configuration, and best practices for autonomous gaming agents
metadata: {"openclaw": {"emoji": "🎮", "os": ["darwin", "linux", "win32"]}}
---

# 🎮 OpenClaw Integration Skill

**Status:** ✅ Ready for Integration  
**Version:** 1.0.0  
**Last Updated:** March 19, 2026

---

## 🎯 What This Skill Provides

This skill provides OpenClaw-specific integration patterns for PyBoy Game Boy emulation, including:
- **Agent spawn patterns** - How to spawn sub-agents for gaming tasks
- **Tool configuration** - MCP server registration and management
- **Session management** - Using session tools for game state
- **Best practices** - OpenClaw-native workflows for autonomous agents

---

## 🚀 Agent Structure (OpenClaw Pattern)

### Sub-Agent Spawn Pattern

Use `sessions_spawn` to create isolated gaming agent sessions:

```bash
# Spawn a sub-agent for specific gaming task
sessions_spawn --task "Battle the Elite 4 in Pokemon Red" --model bailian/MiniMax-M2.5
```

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `task` | Task description for the sub-agent | "Battle Brock" |
| `model` | Override default model | `bailian/kimi-k2.5` |
| `label` | Optional session label | "gym-battle" |
| `runTimeoutSeconds` | Max runtime (0=unlimited) | 300 |
| `cleanup` | Delete/keep session after | `delete` |

### Multi-Agent Routing

Route gaming tasks to specialized models:

| Task Type | Model | Why |
|-----------|-------|-----|
| **Screen Analysis** | `bailian/kimi-k2.5` | FREE vision + image understanding |
| **Battle Strategy** | `bailian/qwen3.5-plus` | Best reasoning (83.2% MMLU) |
| **Exploration** | `bailian/MiniMax-M2.5` | FREE unlimited, fast |
| **Complex Puzzles** | `bailian/qwen3.5-plus` | Complex reasoning |

---

## 🛠️ MCP Server Configuration

### Register MCP Server (Required)

```bash
# Add PyBoy MCP server to OpenClaw
mcporter add duckbot-emulator --stdio "python3 /Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"
```

### Verify Registration

```bash
# List registered MCP servers
mcporter list | grep duckbot

# Test connection
mcporter call duckbot-emulator emulator_get_state
```

---

## 🔧 Tool Patterns That Work

### Session Tools (For Game State)

```bash
# List active gaming sessions
sessions_list

# Get session history
sessions_history sessionKey="game-session-123"

# Send task to another gaming session
sessions_send sessionKey="game-session-123" message="Continue grinding on Route 1"
```

### Spawned Agent Patterns

```bash
# Spawn gaming agent with vision
sessions_spawn \
  --task "Play Pokemon Red - explore Viridian City and buy supplies" \
  --model bailian/kimi-k2.5 \
  --label "exploration-agent" \
  --runTimeoutSeconds 600

# Spawn battle specialist
sessions_spawn \
  --task "Battle Brock at Pewter City Gym. Use type advantage to win." \
  --model bailian/qwen3.5-plus \
  --label "battle-agent"
```

---

## 📋 SKILL.md Format (AgentSkills Spec)

OpenClaw skills use AgentSkills-compatible format:

```markdown
---
name: skill-name
description: What this skill provides
metadata: {"openclaw": {"emoji": "🎮", "os": ["darwin", "linux", "win32"]}}
---

# Skill Content

Your skill documentation here...
```

**Frontmatter fields:**
| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Skill identifier |
| `description` | Yes | Short description |
| `metadata` | No | JSON object with OpenClaw specifics |

**Metadata fields:**
| Field | Type | Description |
|-------|------|-------------|
| `emoji` | string | Skill icon for UI |
| `os` | array | Supported platforms |
| `requires.bins` | array | Required CLI tools |
| `requires.env` | array | Required env variables |
| `install` | array | Installation instructions |

---

## 🎮 Gaming Agent Workflow

### Standard OpenClaw Gaming Loop

```
1. START SESSION
   sessions_spawn --task "Play Pokemon Red" --model bailian/kimi-k2.5

2. LOAD ROM (via MCP)
   duckbot-emulator.emulator_load_rom({"rom_path": "pokemon-red.gb"})

3. GET VISUAL CONTEXT
   duckbot-emulator.emulator_get_frame({"include_base64": true})

4. ANALYZE (with vision model)
   Use bailian/kimi-k2.5 to understand screen

5. DECIDE ACTION
   Based on game state + vision analysis

6. EXECUTE
   duckbot-emulator.emulator_press_sequence({"sequence": "A"})

7. SAVE PROGRESS
   duckbot-emulator.emulator_save_state({"save_name": "checkpoint"})

8. ANNOUNCE (auto via sessions_spawn)
   Results delivered to main chat
```

### Vision-Only Gaming (No Emulator Control)

For pure analysis without direct control:

```bash
# Get screenshot
duckbot-emulator.emulator_get_frame({"include_base64": true})

# Analyze with vision model
# Prompt: "Analyze this Pokemon game screen and recommend actions"
```

---

## 🔗 Integration Points

### With DuckBot Skill

This skill works alongside `duckbot` skill:
- `duckbot` - Gaming persona and gameplay strategies
- `openclaw` - OpenClaw-specific integration patterns

### MCP Tools Available

| Tool | Source | Purpose |
|------|--------|---------|
| `emulator_*` | duckbot-emulator | Game control |
| `sessions_spawn` | OpenClaw core | Create sub-agents |
| `sessions_list` | OpenClaw core | List agents |
| `sessions_send` | OpenClaw core | Message agents |

---

## 📁 File Locations

```
ai-Py-boy-emulation-main/
├── skills/openclaw/
│   └── SKILL.md                    # This file
├── skills/duckbot/
│   └── SKILL.md                    # DuckBot persona
├── skills/pyboy/
│   └── SKILL.md                    # PyBoy reference
├── ai-game-server/
│   ├── mcp_server.py               # MCP server
│   └── requirements.txt
└── AGENTS.md                       # Agent guide
```

---

## 🔧 Configuration (openclaw.json)

Add to your `~/.openclaw/openclaw.json`:

```json
{
  "mcpServers": {
    "duckbot-emulator": {
      "command": "python3",
      "args": ["/Users/duckets/.openclaw/workspace/ai-Py-boy-emulation-main/ai-game-server/mcp_server.py"],
      "env": {}
    }
  }
}
```

---

## 🏆 Best Practices

### Agent Spawning
1. **Use descriptive tasks** - "Battle Brock" not "do stuff"
2. **Set appropriate timeouts** - 300-600s for exploration, 60s for battles
3. **Choose right model** - Vision for screen analysis, reasoning for strategy
4. **Label sessions** - Makes debugging easier

### MCP Server
1. **Verify registration** before first use
2. **Test connection** with simple tool calls
3. **Handle errors gracefully** - MCP can disconnect

### Gaming Sessions
1. **Save often** - Use save states liberally
2. **Check game state** before important decisions
3. **Use vision** for complex situations
4. **Announce results** - Let main session know progress

---

## 🔗 Related Documentation

- [OpenClaw Skills Docs](https://docs.openclaw.ai/tools/skills) - Official skill format
- [Session Tools](https://docs.openclaw.ai/concepts/session-tool) - sessions_spawn, sessions_list
- [AGENTS.md](../../AGENTS.md) - Agent-first gaming guide
- [duckbot/SKILL.md](../duckbot/SKILL.md) - DuckBot persona

---

**This skill enables OpenClaw-native integration for autonomous Game Boy gameplay!**

*Powered by OpenClaw + Bailian AI*