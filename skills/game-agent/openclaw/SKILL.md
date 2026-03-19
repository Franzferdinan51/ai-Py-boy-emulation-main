# OpenClaw Integration Guide

**Status:** ✅ Ready  
**Version:** 1.0  
**Last Updated:** March 19, 2026

---

## Overview

This guide covers OpenClaw-specific setup, MCP server registration, and spawn examples for the Game Agent skill. The MCP server enables OpenClaw agents to control Game Boy emulation programmatically.

---

## MCP Server Registration

### Quick Register (Pre-configured)

Two MCP servers are already configured:

```bash
# Persistent HTTP server (recommended for agent workflows)
mcporter add pyboy --url "http://127.0.0.1:8000/mcp"

# Stdio server (spawns new process each call)
mcporter add pyboy-stdio --stdio "python3 -m mcp_server.server"
```

### Verify Registration

```bash
mcporter list | grep pyboy
```

Expected output:
```
pyboy — PyBoy GameBoy emulator MCP server (persistent HTTP) (13 tools)
pyboy-stdio — PyBoy GameBoy emulator MCP server (stdio) (13 tools)
```

### Server URLs

| Server | Type | URL/Command | Best For |
|--------|------|-------------|----------|
| `pyboy` | HTTP | `http://127.0.0.1:8000/mcp` | Persistent sessions, many calls |
| `pyboy-stdio` | stdio | Spawns new process | Isolated calls, testing |

---

## Manual Registration

If servers aren't pre-configured, register them manually:

### Option 1: HTTP Server

```bash
mcporter add pyboy \
  --url "http://127.0.0.1:8000/mcp" \
  --description "PyBoy GameBoy emulator MCP server"
```

### Option 2: stdio Server

```bash
mcporter add pyboy-stdio \
  --command "/Users/duckets/.openclaw/workspace/.venv-pyboy/bin/python" \
  --args "-m" "mcp_server.server" \
  --cwd "/Users/duckets/.openclaw/workspace/mcp-pyboy/src" \
  --description "PyBoy GameBoy emulator MCP server (stdio)"
```

### Option 3: Custom ROM Path

```bash
# Create custom launcher with specific ROM
cat > /usr/local/bin/pyboy-pokemon << 'EOF'
#!/bin/bash
cd /Users/duckets/.openclaw/workspace/mcp-pyboy/src
exec python3 -m mcp_server.server "$@"
EOF

chmod +x /usr/local/bin/pyboy-pokemon

mcporter add pyboy-pokemon \
  --stdio "pyboy-pokemon"
```

---

## Starting the MCP Server

### For HTTP Server (Persistent)

```bash
# Option 1: Using the built-in persistent server
cd /Users/duckets/.openclaw/workspace/mcp-pyboy/src
source ../.venv/bin/activate
python3 -m mcp_server.run_persistent

# Option 2: Background process
nohup python3 -m mcp_server.run_persistent > /tmp/pyboy-mcp.log 2>&1 &
```

### Verify Server is Running

```bash
# Health check
curl -s http://127.0.0.1:8000/health
# Should return: {"status": "ok"}

# Or via mcporter
mcporter call pyboy.health_check
```

---

## All Available Tools

The MCP server provides 13 tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `health_check` | Verify server is running | (none) |
| `get_server_info` | Get server version/capabilities | (none) |
| `load_rom` | Load a .gb/.gbc ROM file | `rom_path: string` |
| `get_screen` | Get screen as base64 PNG | (none) |
| `get_session_info` | Get current session state | (none) |
| `press_button` | Press a button | `button: string`, `hold_duration?: number` |
| `wait_frames` | Wait N frames | `frames?: number` |
| `get_memory` | Read RAM at address | `address: number`, `length?: number` |
| `get_player_info` | Get player position, money, badges | (none) |
| `get_inventory` | Get bag items | (none) |
| `save_game_state` | Save to file | `save_path?: string` |
| `load_game_state` | Load from file | `save_path: string` |
| `get_party_pokemon` | Get party Pokemon details | (none) |

---

## Spawn Examples

### Basic Sub-Agent Spawn

```bash
# Spawn agent to play Pokemon
sessions_spawn \
  --task "Load pokemon-red.gb and start a new game. Choose Charmander as your starter." \
  --model bailian/kimi-k2.5
```

### With Specific Tool Access

```bash
# Spawn with explicit MCP tool access
sessions_spawn \
  --task "Play Pokemon Red: load the ROM, check player info, explore Pallet Town" \
  --model bailian/kimi-k2.5 \
  --mcp-servers pyboy
```

### Using mcporter Directly

```bash
# Load a ROM
mcporter call pyboy.load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"

# Check server info
mcporter call pyboy.get_server_info

# Get current screen
mcporter call pyboy.get_screen

# Press buttons
mcporter call pyboy.press_button button="START"
mcporter call pyboy.press_button button="A" hold_duration=10
```

### Complete Gameplay Workflow

```bash
# 1. Start new game
mcporter call pyboy.load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"

# 2. Press START to begin
mcporter call pyboy.press_button button="START"

# 3. Wait for text
mcporter call pyboy.wait_frames frames=60

# 4. Press A to confirm options
mcporter call pyboy.press_button button="A"

# 5. Get screen for vision analysis
mcporter call pyboy.get_screen

# 6. Get player info
mcporter call pyboy.get_player_info

# 7. Save progress
mcporter call pyboy.save_game_state save_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/saves/duckbot-start.state"
```

---

## OpenClaw Agent Configuration

### Model Selection

| Model | Use Case | Recommendation |
|-------|----------|----------------|
| `bailian/kimi-k2.5` | Vision + text analysis | ✅ Recommended |
| `bailian/MiniMax-M2.5` | General tasks | Good backup |
| `bailian/glm-5` | Fast coding | For automation scripts |

### Spawning via sessions_spawn

```bash
# Minimal spawn
sessions_spawn --task "Play Pokemon Red" --model bailian/kimi-k2.5

# Full spawn with options
sessions_spawn \
  --task "Complete Pokemon Red: beat the Elite 4" \
  --model bailian/kimi-k2.5 \
  --max-steps 1000 \
  --timeout 3600
```

### Direct mcporter Usage in Agents

```json
{
  "tool": "pyboy.load_rom",
  "args": {
    "rom_path": "/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"
  }
}
```

```json
{
  "tool": "pyboy.press_button",
  "args": {
    "button": "A",
    "hold_duration": 5
  }
}
```

---

## Example: Complete Game Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    GAME AGENT LOOP                         │
├─────────────────────────────────────────────────────────────┤
│  1. GET SCREEN                                             │
│     → pyboy.get_screen() → base64 PNG                      │
│                                                             │
│  2. ANALYZE (Vision Model)                                  │
│     → kimi-k2.5: "What should I do?"                       │
│                                                             │
│  3. DECIDE                                                  │
│     → Based on game state + vision                         │
│                                                             │
│  4. ACT                                                     │
│     → pyboy.press_button(button="A")                        │
│                                                             │
│  5. WAIT                                                    │
│     → pyboy.wait_frames(frames=30)                          │
│                                                             │
│  6. CHECK STATE                                             │
│     → pyboy.get_player_info()                              │
│                                                             │
│  7. SAVE (as needed)                                        │
│     → pyboy.save_game_state(save_path="checkpoint.state")   │
│                                                             │
│  8. REPEAT                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Server Not Running

```bash
# Check if server is running
curl -s http://127.0.0.1:8000/health || echo "Server not running"

# Start server
cd /Users/duckets/.openclaw/workspace/mcp-pyboy/src
python3 -m mcp_server.run_persistent &

# Or use the script
./start-mcp-server.sh
```

### Tools Not Available

```bash
# Verify registration
mcporter list

# Re-add if needed
mcporter remove pyboy
mcporter add pyboy --url "http://127.0.0.1:8000/mcp"
```

### ROM Loading Fails

```bash
# Verify ROM exists
ls -la /Users/duckets/.openclaw/workspace/mcp-pyboy/roms/*.gb

# Try with full path
mcporter call pyboy.load_rom rom_path="/Users/duckets/.openclaw/workspace/mcp-pyboy/roms/pokemon-red.gb"
```

### Memory Read Errors

```bash
# Check if ROM is loaded
mcporter call pyboy.get_session_info

# Memory addresses vary by game
# For Pokemon Red:
# - 0xD361: Player Y position
# - 0xD362: Player X position
# - 0xD35E: Map ID
mcporter call pyboy.get_memory address=0xD361 length=1
```

---

## File Locations

```
/Users/duckets/.openclaw/workspace/
├── mcp-pyboy/
│   ├── src/mcp_server/
│   │   ├── server.py          # Main MCP server
│   │   └── run_persistent.py  # HTTP server launcher
│   ├── saves/                 # Save states
│   ├── roms/                  # ROM files
│   └── .venv/                 # Python environment
│
└── ai-Py-boy-emulation-main/
    ├── skills/game-agent/
    │   ├── SKILL.md           # Main skill docs
    │   ├── openclaw/SKILL.md  # This file
    │   └── examples/          # Game-specific guides
    └── ai-game-server/
        └── mcp_server.py      # Alternative MCP server
```

---

## Related Documentation

- [game-agent/SKILL.md](../SKILL.md) - Main skill documentation
- [examples/pokemon_red.md](../examples/pokemon_red.md) - Pokemon Red guide
- [examples/generic.md](../examples/generic.md) - Generic game guide
- [mcp-pyboy README](../../mcp-pyboy/README.md) - MCP PyBoy docs
- [OPENCLAW-INTEGRATION](../../OPENCLAW-INTEGRATION.md) - Original integration guide

---

**Author:** Game Agent  
**Last Updated:** March 19, 2026