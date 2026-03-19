# 🎮 AI GameBoy Emulator

OpenClaw Agent-powered emulation for Game Boy, Game Boy Color, and Game Boy Advance games. Now with **MCP v4.0.0** and autonomous AI gameplay!

## What This Is

An MCP server + web interface that lets AI agents control Game Boy emulation. Agents can:
- Read game memory (position, inventory, HP, etc.)
- Press buttons autonomously
- Analyze screens via vision AI
- Make intelligent decisions with auto-play modes
- Manage persistent gaming sessions

## ✨ New in v4.0.0

| Feature | Description |
|---------|-------------|
| **Smart Session Management** | Persistent sessions with TTL, goal tracking, and data storage |
| **Auto-Catch** | AI automatically catches wild Pokemon |
| **Auto-Item Use** | AI uses items strategically |
| **Auto-NPC Talk** | AI interacts with NPCs autonomously |
| **Enhanced Memory Access** | Read memory ranges and single bytes |
| **Error Codes** | Programmatic error handling with error codes |
| **Better Type Hints** | Improved code quality |

## Supported Systems

| System | Formats |
|--------|---------|
| Game Boy | .gb |
| Game Boy Color | .gbc |
| Game Boy Advance | .gba |

## Requirements

- Python 3.10+
- Node.js 18+ (for web UI)
- PyBoy: `pip install pyboy`
- MCP: `pip install mcp`

## Installation

```bash
# Clone repo
git clone https://github.com/Franzferdinan51/ai-Py-boy-emulation-main
cd ai-Py-boy-emulation-main

# Install Python dependencies
cd ai-game-server
pip install -r requirements.txt

# Install Node dependencies (for web UI)
cd ../ai-game-assistant
npm install
```

## Quick Start

### 1. Configure AI Provider

Create `ai-game-server/.env` with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# OR Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OR Local (LM Studio, Ollama)
OPENAI_API_BASE=http://localhost:1234/v1
```

### 2. Start Backend

```bash
cd ai-game-server/src
BACKEND_PORT=5002 python3 main.py
```

### 3. Start Web UI (optional)

```bash
cd ../ai-game-assistant
npx vite
```

### 4. Register MCP Server

```bash
mcporter add gameboy --stdio "python3 /path/to/ai-game-server/mcp_server.py"
```

## MCP Tools

### Core Controls

| Tool | Description |
|------|-------------|
| `emulator_load_rom` | Load a ROM file |
| `emulator_press_button` | Press GB button (A/B/START/SELECT/UP/DOWN/LEFT/RIGHT) |
| `emulator_press_sequence` | Press multiple buttons in sequence |
| `emulator_tick` | Advance N frames |
| `emulator_get_state` | Get emulator state |

### Vision & Screen

| Tool | Description |
|------|-------------|
| `get_screen_base64` | Get screen as base64 for AI vision |
| `emulator_get_frame` | Get current game frame |
| `emulator_save_screenshot` | Save screenshot to file |

### Memory Reading

| Tool | Description |
|------|-------------|
| `get_player_position` | Get player X,Y coordinates |
| `get_party_info` | Get all Pokemon in party |
| `get_inventory` | Get bag items |
| `get_map_location` | Get current location |
| `get_money` | Get player money |
| `emulator_read_memory` | Read raw RAM at address |
| `get_memory_range` | Read memory range |
| `get_memory_byte` | Read single byte |
| `read_game_state` | Get full game state snapshot |

### Save States

| Tool | Description |
|------|-------------|
| `save_game_state` | Save game progress |
| `load_game_state` | Load saved game |
| `emulator_list_saves` | List all saves |

### Auto-Play Modes 🤖

| Tool | Description |
|------|-------------|
| `auto_battle` | AI fights Pokemon automatically |
| `auto_explore` | AI explores the world |
| `auto_grind` | AI grinds for XP/money |
| `auto_catch` | AI catches wild Pokemon |
| `auto_item_use` | AI uses items strategically |
| `auto_npc_talk` | AI talks to NPCs |

### Session Management

| Tool | Description |
|------|-------------|
| `session_start` | Create new gaming session |
| `session_get` | Get session data |
| `session_set` | Store session data |
| `session_list` | List all sessions |
| `session_delete` | Delete session |

## HTTP API Endpoints

### Core
- `GET /health` - Health check
- `GET /api/status` - Server status
- `GET /api/config` - Configuration

### Emulator
- `POST /api/load_rom` - Load ROM
- `POST /api/action` - Send button input
- `POST /api/ai-action` - Get AI action
- `GET /api/screen` - Get screen image

### Save/Load
- `POST /api/save_state` - Save state
- `POST /api/load_state` - Load state

### Memory
- `POST /memory` - Read memory
- `GET /characters` - Get sprites
- `GET /tilemap` - Get tilemap
- `GET /sprites` - Get sprites

### AI
- `POST /api/chat` - Chat with AI
- `GET /api/providers/status` - AI provider status

## Example: Autonomous Pokemon Battle

```python
import json
import subprocess

def call_mcp(tool, args):
    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool, "arguments": args}
    }
    result = subprocess.run(
        ["python3", "mcp_server.py"],
        input=json.dumps(request),
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout).get("result", {})

# Start a gaming session
session = call_mcp("session_start", {"goal": "Beat Elite 4"})

# Get party info
party = call_mcp("get_party_info", {})

# Let AI battle automatically
battle_result = call_mcp("auto_battle", {"max_moves": 10})
print(f"Battle result: {battle_result}")
```

## Web UI

The web UI provides:
- Manual game control
- Agent status monitoring  
- Screen viewing
- Settings configuration

Access at http://localhost:5173 (default)

## File Structure

```
ai-Py-boy-emulation-main/
├── ai-game-server/         # Python backend + MCP
│   ├── mcp_server.py       # MCP server (v4.0.0)
│   └── src/                # Flask API
├── ai-game-assistant/     # React web UI
├── skills/                 # OpenClaw skills
├── tools/                  # Agent utilities
├── guides/                 # Documentation
│   ├── API_REFERENCE.md   # Full API docs
│   ├── DECISION_TREE.md  # AI decision making
│   └── VISION_GUIDE.md   # Vision integration
└── tests/                  # Test suites
    ├── test_api.py        # HTTP API tests
    └── test_mcp.py        # MCP tools tests
```

## Documentation

- [📖 API Reference](guides/API_REFERENCE.md) - Complete API documentation
- [🚀 Quick Start](QUICKSTART.md) - New user guide
- [🧠 Decision Tree](guides/DECISION_TREE.md) - How AI makes decisions
- [👁️ Vision Guide](guides/VISION_GUIDE.md) - Vision AI integration
- [📝 Examples](EXAMPLES.md) - Example prompts and sessions
- [🔧 Troubleshooting](TROUBLESHOOTING.md) - Common issues
- [📦 AGENTS.md](AGENTS.md) - Autonomous agent setup

## Testing

```bash
# Install pytest
pip install pytest requests

# Run HTTP API tests
pytest tests/test_api.py -v

# Run MCP tools tests
pytest tests/test_mcp.py -v

# Run all tests
pytest tests/ -v
```

## Troubleshooting

### Backend won't start
- Check Python version (3.10+)
- Verify dependencies installed
- Check port not in use

### MCP tools not responding
- Verify backend running
- Check mcporter registration
- Try restarting backend

### ROM won't load
- Verify ROM file exists and is readable
- Check file format supported

## License

MIT

---

**DuckBot is currently playing Pokemon Red!** 🎮🦆